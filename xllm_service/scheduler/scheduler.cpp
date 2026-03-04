/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "scheduler/scheduler.h"

#include "common/xllm/status.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "tokenizer/tokenizer_factory.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>
#include <limits>
#include <thread>
#include <unordered_set>

static constexpr int kHeartbeatInterval = 3;  // in seconds
static constexpr int kQueueProcessThreadNum = 4;
static std::string ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";

namespace xllm_service {

Scheduler::Scheduler(const Options& options) : options_(options) {
  tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                  &tokenizer_args_);
  chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);

  etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr());
  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  }

  instance_mgr_ =
      std::make_unique<InstanceMgr>(options, etcd_client_, is_master_service_);

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else if (options.load_balance_policy() == "LST_IMH") {
    // LST-IMH mode: dispatch coordinator thread replaces processing threads
    dispatch_coordinator_thread_ = std::make_unique<std::thread>(
        &Scheduler::dispatch_coordinator, this);
    LOG(INFO) << "LST-IMH scheduling mode enabled.";
  } else {
    lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
  }

  if (is_master_service_) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  } else {
    auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
    etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  }
}

Scheduler::~Scheduler() {
  exited_ = true;

  // Wake up dispatch coordinator if in LST_IMH mode
  {
    std::lock_guard<std::mutex> lock(dispatch_mutex_);
    dispatch_signal_ = true;
  }
  dispatch_cv_.notify_one();
  if (dispatch_coordinator_thread_ && dispatch_coordinator_thread_->joinable()) {
    dispatch_coordinator_thread_->join();
  }

  // Drain pending queue - notify clients of remaining requests
  {
    std::lock_guard<std::mutex> lock(pending_queue_mutex_);
    for (auto& req : pending_queue_) {
      if (req && req->timeout_callback) {
        req->timeout_callback();
      }
    }
    pending_queue_.clear();
  }

  for (auto& queue_pair : request_queues_) {
    queue_pair.second->emplace(nullptr);  // unblock queue
  }
  for (auto& thread_pair : processing_threads_) {
    for (auto& thread : thread_pair.second) {
      if (thread->joinable()) {
        thread->join();
      }
    }
  }
  etcd_client_->stop_watch();
}

bool Scheduler::schedule(std::shared_ptr<Request> request) {
  // apply chat template
  if (request->messages.size() > 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }

    auto prompt = chat_template_->apply(request->messages);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages";
      return false;
    }
    request->prompt = prompt.value();
  }

  // encode prompt
  if (request->prompt.size() != 0) {
    if (!get_tls_tokenizer()->encode(request->prompt, &request->token_ids)) {
      LOG(ERROR) << "Encode prompt failed: " << request->prompt;
      return false;
    }
  }

  // Update model heat
  if (request->prompt.size() != 0) {
    instance_mgr_->update_model_heat(request->model, request->token_ids.size());
  }

  // LST-IMH mode: add to pending queue, signal coordinator
  if (options_.load_balance_policy() == "LST_IMH") {
    // Compute estimated processing time
    if (!request->token_ids.empty()) {
      request->estimated_processing_time_ms = static_cast<int64_t>(
          instance_mgr_->predict_ttft_any_instance(request->model,
                                                    request->token_ids.size()));
    }

    {
      std::lock_guard<std::mutex> lock(pending_queue_mutex_);
      pending_queue_.push_back(request);
      // Remember model name for coordinator (single model assumption)
      lst_imh_model_name_ = request->model;
    }

    signal_dispatch();
    return true;
  }

  // Push request to queue (existing path for RR/CAR/SLO_AWARE)
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (request_queues_.find(request->model) == request_queues_.end()) {
      request_queues_[request->model] =
          std::make_shared<ConcurrentQueue<std::shared_ptr<Request>>>();
      for (int i = 0; i < kQueueProcessThreadNum; ++i) {
        processing_threads_[request->model].emplace_back(
            std::make_unique<std::thread>(&Scheduler::process_request_queue,
                                          this,
                                          request->model));
      }
    }
    request_queues_[request->model]->push(request);
  }

  return true;
}

void Scheduler::process_request_queue(const std::string& model_name) {
  while (!exited_) {
    std::shared_ptr<ConcurrentQueue<std::shared_ptr<Request>>> queue;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (request_queues_.find(model_name) == request_queues_.end()) {
        break;
      }
      queue = request_queues_[model_name];
    }

    auto request = queue->pop();
    if (request == nullptr) {
      continue;
    }

    // Dual-pool routing
    PoolType pool = instance_mgr_->get_model_pool(request->model);

    // New model: assign to pool (blocking if steady wakeup needed)
    if (pool == PoolType::NONE) {
      instance_mgr_->assign_model_to_pool(request->model);
      pool = instance_mgr_->get_model_pool(request->model);
    }

    // Steady pool: check for upgrade to elastic
    if (pool == PoolType::STEADY) {
      if (instance_mgr_->steady_part_check_upgrading(request->model)) {
        pool = PoolType::ELASTIC;
      }
    }

    if (pool == PoolType::STEADY) {
      // Route to steady pool instance
      if (!instance_mgr_->route_to_steady_instance(request)) {
        LOG(ERROR) << "Failed to route to steady instance for " << request->model;
        continue;
      }
    } else {
      // Elastic pool: set prefill_only mode
      request->prefill_only = true;

      int32_t model_count = instance_mgr_->get_wakeup_count(request->model);
      if (model_count == 0) {
        // Cold elastic model: blocking wakeup
        instance_mgr_->dynamic_part_auto_scaling();

        auto awake = instance_mgr_->get_awake_instances(request->model);
        if (awake.empty()) {
          LOG(ERROR) << "dynamic_part_auto_scaling failed to wake model " << request->model;
          continue;
        }
        request->routing.prefill_name = awake[0];
        request->routing.decode_name = awake[0];
      } else {
        // Warm elastic model: route first, then trigger scaling adjustment
        lb_policy_->select_instances_pair(request);
        instance_mgr_->dynamic_part_auto_scaling();
      }
    }

    DLOG(INFO) << request->routing.debug_string();

    // update request metrics
    if (request->prompt.size() != 0) {
      instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
    }

    if (request->dispatch_callback) {
      std::thread([request]() { request->dispatch_callback(); }).detach();
    }
  }
}

std::shared_ptr<brpc::Channel> Scheduler::get_channel(
    const std::string& target_name) {
  return instance_mgr_->get_channel(target_name);
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    global_kvcache_mgr_->upload_kvcache();

    instance_mgr_->upload_load_metrics();
  }
}

void Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_) {
    return;
  }
  instance_mgr_->on_heartbeat(req->name());
  global_kvcache_mgr_->record_updated_kvcaches(req->name(), req->cache_event());
  instance_mgr_->record_load_metrics_update(req->name(), req->load_metrics());
  instance_mgr_->update_latency_metrics(req->name(), req->latency_metrics());

  // Update XTensor info if present
  if (req->has_xtensor_info()) {
    instance_mgr_->update_xtensor_info(req->name(), req->xtensor_info());
  }
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        options_.service_name(),
                        kHeartbeatInterval)) {
    is_master_service_ = true;

    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);

    global_kvcache_mgr_->set_as_master();
    instance_mgr_->set_as_master();
  }
}

InstanceMetaInfo Scheduler::get_instance_info(
    const std::string& instance_name) {
  return instance_mgr_->get_instance_info(instance_name);
}

std::vector<std::string> Scheduler::get_static_decode_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_decode_list(instance_name);
}

std::vector<std::string> Scheduler::get_static_prefill_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_prefill_list(instance_name);
}

Tokenizer* Scheduler::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         first_message_sent = std::unordered_set<size_t>(),
         service_request_id = request->service_request_id,
         created_time = absl::ToUnixSeconds(absl::Now())](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(call_data,
                                                      &first_message_sent,
                                                      include_usage,
                                                      service_request_id,
                                                      created_time,
                                                      model,
                                                      req_output);
      }

      return response_handler_.send_result_to_client(
          call_data, service_request_id, created_time, model, req_output);
    };
    requests_[request->service_request_id] = request;
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->service_request_id] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }
    request->output_callback =
        [this,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         service_request_id = request->service_request_id,
         created_time = absl::ToUnixSeconds(absl::Now())](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(call_data,
                                                      include_usage,
                                                      service_request_id,
                                                      created_time,
                                                      model,
                                                      req_output);
      }

      return response_handler_.send_result_to_client(
          call_data, service_request_id, created_time, model, req_output);
    };
    requests_[request->service_request_id] = request;
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[request->service_request_id] =
        next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

void Scheduler::finish_request(const std::string& service_request_id,
                               bool error) {
  std::string prefill_instance;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it != requests_.end()) {
      prefill_instance = it->second->routing.prefill_name;
      // update instance request metrics for finished request
      if (error) {
        instance_mgr_->update_request_metrics(it->second,
                                              RequestAction::CANCEL);
      } else {
        instance_mgr_->update_request_metrics(it->second,
                                              RequestAction::FINISH_DECODE);
      }

      requests_.erase(it);
    }
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
  }

}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  const std::string& service_request_id = request_output.service_request_id;
  OutputCallback cb;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it == requests_.end()) {
      LOG(ERROR) << "Can not found the callback for the received request "
                    "output, request id is: "
                 << service_request_id;
      return false;
    }
    cb = it->second->output_callback;

    // update instance request metrics
    it->second->num_generated_tokens += 1;
    instance_mgr_->update_request_metrics(it->second, RequestAction::GENERATE);
  }

  size_t req_thread_idx = -1;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    auto it = remote_requests_output_thread_map_.find(service_request_id);
    if (it == remote_requests_output_thread_map_.end()) {
      LOG(ERROR) << "Can not found the thread for the received request output, "
                    "request id is: "
                 << service_request_id;
      return false;
    }
    req_thread_idx = it->second;
  }

  output_threadpools_[req_thread_idx].schedule(
      [this,
       service_request_id,
       cb,
       request_output = std::move(request_output)]() mutable {
        if (!cb(request_output) || request_output.finished) {
          finish_request(service_request_id);
        }
      });

  return true;
}

void Scheduler::update_request_metrics_for_prefill(
    const std::string& service_request_id) {
  std::string prefill_instance;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it != requests_.end()) {
      prefill_instance = it->second->routing.prefill_name;
      it->second->num_generated_tokens += 1;
      // update instance request metrics for prefill finished request
      instance_mgr_->update_request_metrics(it->second,
                                            RequestAction::FINISH_PREFILL);
    }
  }

  // Signal dispatch coordinator on PREFILL_DONE so it can re-evaluate availability
  if (!prefill_instance.empty() &&
      options_.load_balance_policy() == "LST_IMH") {
    LOG(INFO) << "[LST-IMH] PREFILL_DONE: instance=" << prefill_instance
              << " request=" << service_request_id;
    signal_dispatch();
  }

}

// === LST-IMH Implementation ===

void Scheduler::signal_dispatch() {
  {
    std::lock_guard<std::mutex> lock(dispatch_mutex_);
    dispatch_signal_ = true;
  }
  dispatch_cv_.notify_one();
}

std::vector<SchedulingJob> Scheduler::moore_hodgson(
    const std::vector<SchedulingJob>& jobs, int64_t T) {
  // Sort jobs by EDD (earliest due date first)
  std::vector<SchedulingJob> sorted_jobs = jobs;
  std::sort(sorted_jobs.begin(), sorted_jobs.end(),
            [](const SchedulingJob& a, const SchedulingJob& b) {
              return a.deadline_ms < b.deadline_ms;
            });

  // on_time_indices tracks which jobs (by index in sorted_jobs) are on-time
  std::vector<size_t> on_time_indices;
  // Track removed indices for lazy heap deletion
  std::unordered_set<size_t> removed_indices;

  // Max-heap: (processing_time, index_in_sorted_jobs)
  auto cmp = [](const std::pair<int64_t, size_t>& a,
                const std::pair<int64_t, size_t>& b) {
    return a.first < b.first;
  };
  std::priority_queue<std::pair<int64_t, size_t>,
                      std::vector<std::pair<int64_t, size_t>>,
                      decltype(cmp)> max_heap(cmp);

  int64_t current_time = T;

  for (size_t i = 0; i < sorted_jobs.size(); ++i) {
    on_time_indices.push_back(i);
    max_heap.push({sorted_jobs[i].processing_time_ms, i});
    current_time += sorted_jobs[i].processing_time_ms;

    if (current_time > sorted_jobs[i].deadline_ms) {
      // Skip stale heap entries (already removed in a previous iteration)
      while (!max_heap.empty() &&
             removed_indices.count(max_heap.top().second)) {
        max_heap.pop();
      }
      if (max_heap.empty()) break;

      // Remove job with largest processing time
      auto [p_max, idx_max] = max_heap.top();
      max_heap.pop();

      // Mark as removed
      removed_indices.insert(idx_max);

      // Remove idx_max from on_time_indices
      on_time_indices.erase(
          std::remove(on_time_indices.begin(), on_time_indices.end(), idx_max),
          on_time_indices.end());

      current_time -= p_max;
    }
  }

  // Build result from on_time_indices (already in EDD order)
  std::vector<SchedulingJob> result;
  result.reserve(on_time_indices.size());
  for (size_t idx : on_time_indices) {
    result.push_back(sorted_jobs[idx]);
  }
  return result;
}

std::unordered_map<std::string, std::vector<SchedulingJob>>
Scheduler::run_lst_imh(std::vector<SchedulingJob>& jobs,
                       std::vector<MachineInfo>& machines) {
  // Sort machines by availability time ascending (earliest available first)
  // Tie-break by instance name for deterministic ordering
  std::sort(machines.begin(), machines.end(),
            [](const MachineInfo& a, const MachineInfo& b) {
              if (a.availability_time_ms != b.availability_time_ms)
                return a.availability_time_ms < b.availability_time_ms;
              return a.instance_name < b.instance_name;
            });

  std::unordered_map<std::string, std::vector<SchedulingJob>> assignment;
  std::vector<SchedulingJob> remaining = jobs;

  for (const auto& machine : machines) {
    if (remaining.empty()) break;

    // Run Moore-Hodgson on remaining jobs for this machine
    auto on_time = moore_hodgson(remaining, machine.availability_time_ms);

    if (!on_time.empty()) {
      // Only assign the most urgent job (first in EDD order) to this machine.
      // The dispatch loop dispatches at most 1 job per ready instance per cycle,
      // so assigning more would starve subsequent machines of work.
      assignment[machine.instance_name] = {on_time[0]};

      // Remove only the assigned job from remaining
      const auto& assigned_id = on_time[0].request->service_request_id;
      remaining.erase(
          std::remove_if(remaining.begin(), remaining.end(),
                         [&assigned_id](const SchedulingJob& j) {
                           return j.request->service_request_id == assigned_id;
                         }),
          remaining.end());
    }
  }

  return assignment;
}

void Scheduler::dispatch_coordinator() {
  const int64_t pre_pull_ms =
      static_cast<int64_t>(options_.lst_imh_pre_pull_ms());
  LOG(INFO) << "[LST-IMH] dispatch_coordinator started, pre_pull_ms="
            << pre_pull_ms;

  while (!exited_) {
    // Compute next wake time from EPDT (estimated prefill done time)
    int64_t now_ms = absl::ToUnixMillis(absl::Now());
    int64_t next_wake_ms = std::numeric_limits<int64_t>::max();

    {
      std::string model_name;
      {
        std::lock_guard<std::mutex> lock(pending_queue_mutex_);
        model_name = lst_imh_model_name_;
      }
      if (!model_name.empty()) {
        auto instances = instance_mgr_->get_awake_instances(model_name);
        for (const auto& inst : instances) {
          int64_t epdt = instance_mgr_->get_estimated_prefill_done_time(inst);
          if (epdt > now_ms) {
            int64_t wake_at = epdt - pre_pull_ms;
            if (wake_at > now_ms) {
              next_wake_ms = std::min(next_wake_ms, wake_at);
            }
          }
        }
      }
    }

    // Wait for signal or timer
    {
      std::unique_lock<std::mutex> lock(dispatch_mutex_);
      if (next_wake_ms == std::numeric_limits<int64_t>::max()) {
        // No active prefills, wait indefinitely for signal
        dispatch_cv_.wait(lock, [this] {
          return dispatch_signal_ || exited_;
        });
      } else {
        auto wait_ms = std::max(int64_t(0), next_wake_ms - now_ms);
        dispatch_cv_.wait_for(lock, std::chrono::milliseconds(wait_ms),
                              [this] {
                                return dispatch_signal_ || exited_;
                              });
      }
      dispatch_signal_ = false;
    }

    if (exited_) break;

    now_ms = absl::ToUnixMillis(absl::Now());

    // Get awake instances for the model
    std::string model_name;
    {
      std::lock_guard<std::mutex> lock(pending_queue_mutex_);
      model_name = lst_imh_model_name_;
    }
    if (model_name.empty()) continue;
    auto awake_instances =
        instance_mgr_->get_awake_instances(model_name);
    if (awake_instances.empty()) continue;

    // Build machine list with real-time availability estimates using EPDT
    std::vector<MachineInfo> machines;
    std::vector<std::string> ready_instances;

    for (const auto& inst : awake_instances) {
      int64_t epdt = instance_mgr_->get_estimated_prefill_done_time(inst);
      int64_t T_i = std::max(int64_t(0), epdt - now_ms);
      machines.push_back({inst, T_i});
      if (T_i <= pre_pull_ms) {
        ready_instances.push_back(inst);
      }
    }

    if (ready_instances.empty()) continue;

    // Purge expired requests and build job list
    std::vector<SchedulingJob> jobs;
    {
      std::lock_guard<std::mutex> lock(pending_queue_mutex_);
      auto it = pending_queue_.begin();
      while (it != pending_queue_.end()) {
        int64_t deadline = (*it)->arrival_time_ms + (*it)->ttft_slo_ms;
        if (deadline <= now_ms) {
          // Expired - discard
          LOG(WARNING) << "Request " << (*it)->service_request_id
                       << " expired (TTFT SLO exceeded), discarding.";
          if ((*it)->timeout_callback) {
            (*it)->timeout_callback();
          }
          it = pending_queue_.erase(it);
        } else {
          jobs.push_back({*it, (*it)->estimated_processing_time_ms, deadline});
          ++it;
        }
      }
    }

    if (jobs.empty()) continue;

    // Run LST-IMH algorithm
    auto assignment = run_lst_imh(jobs, machines);

    // Dispatch to each ready instance
    for (const auto& instance : ready_instances) {
      auto assign_it = assignment.find(instance);
      if (assign_it == assignment.end() || assign_it->second.empty()) {
        LOG(INFO) << "[LST-IMH] dispatch: instance " << instance
                  << " is ready but got no assignment";
        continue;
      }

      // Take first job (already in EDD order from Moore-Hodgson)
      auto& job = assign_it->second[0];
      auto request = job.request;

      // Set routing
      request->routing.prefill_name = instance;
      request->routing.decode_name = instance;
      request->estimated_ttft = job.processing_time_ms;

      // Remove from pending queue
      {
        std::lock_guard<std::mutex> lock(pending_queue_mutex_);
        auto before_size = pending_queue_.size();
        pending_queue_.erase(
            std::remove_if(
                pending_queue_.begin(), pending_queue_.end(),
                [&request](const std::shared_ptr<Request>& r) {
                  return r->service_request_id ==
                         request->service_request_id;
                }),
            pending_queue_.end());
      }

      // Update request metrics
      if (!request->prompt.empty()) {
        instance_mgr_->update_request_metrics(request,
                                              RequestAction::SCHEDULE);
      }

      // Note: EPDT (estimated_prefill_done_time) is already updated in
      // update_request_metrics(SCHEDULE) above, no separate dispatch state needed.

      // Execute dispatch callback
      if (request->dispatch_callback) {
        std::thread([request]() { request->dispatch_callback(); }).detach();
      }
    }
  }

}

}  // namespace xllm_service