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
#include "loadbalance_policy/lst_imh_policy.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "scheduler/prism/prism_instance_mgr.h"
#include "scheduler/prism/prism_request_tracker.h"
#include "scheduler/serverless_llm/serverless_llm_instance_mgr.h"
#include "scheduler/llumnix/llumnix_instance_mgr.h"
#include "tokenizer/tokenizer_factory.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>
#include <thread>

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

  prism_mode_ = (options.baseline_type() == "PRISM");
  serverless_llm_mode_ = (options.baseline_type() == "SERVERLESS_LLM");
  llumnix_mode_ = (options.baseline_type() == "LLUMNIX");

  if (prism_mode_) {
    PrismConfig prism_config;
    prism_config.schedule_interval_s = options.prism_schedule_interval_s();
    prism_config.memory_pool_budget_gb = options.prism_memory_pool_budget_gb();
    prism_config.model_idle_threshold_s = options.prism_idle_threshold_s();
    prism_config.migrate_policy = options.prism_migrate_policy();
    prism_config.max_models_per_instance = options.prism_max_models_per_instance();
    prism_config.backend_queue_threshold = options.prism_backend_queue_threshold();

    auto prism_mgr = std::make_shared<PrismInstanceMgr>(
        options, etcd_client_, is_master_service_, prism_config);
    prism_mgr->start_global_scheduler();
    instance_mgr_ = prism_mgr;  // polymorphic assignment

    LOG(INFO) << "Scheduler: Prism mode enabled";
  } else if (serverless_llm_mode_) {
    ServerlessLLMConfig sllm_config;
    sllm_config.schedule_interval_s = options.sllm_schedule_interval_s();
    sllm_config.model_idle_threshold_s = options.sllm_idle_threshold_s();
    sllm_config.d2d_speed_gbps = options.sllm_d2d_speed_gbps();
    sllm_config.h2d_speed_gbps = options.sllm_h2d_speed_gbps();
    sllm_config.drain_alpha = options.sllm_drain_alpha();
    sllm_config.drain_beta = options.sllm_drain_beta();
    sllm_config.target_ongoing_requests = options.sllm_target_ongoing_requests();
    sllm_config.min_instances_per_model = options.sllm_min_instances();
    sllm_config.max_instances_per_model = options.sllm_max_instances();
    sllm_config.enable_knapsack_migration = options.sllm_enable_knapsack();
    sllm_config.max_models_per_instance = options.sllm_max_models_per_instance();

    auto sllm_mgr = std::make_shared<ServerlessLLMInstanceMgr>(
        options, etcd_client_, is_master_service_, sllm_config);
    sllm_mgr->start_global_scheduler();
    instance_mgr_ = sllm_mgr;  // polymorphic assignment

    LOG(INFO) << "Scheduler: ServerlessLLM mode enabled";
  } else if (llumnix_mode_) {
    LlumnixConfig llumnix_config;
    llumnix_config.schedule_interval_s = options.llumnix_schedule_interval_s();
    llumnix_config.model_idle_threshold_s = options.llumnix_idle_threshold_s();
    llumnix_config.migrate_out_load_threshold = options.llumnix_migrate_out_load_threshold();
    llumnix_config.topk_random_dispatch = options.llumnix_topk_random_dispatch();
    llumnix_config.max_models_per_instance = options.llumnix_max_models_per_instance();
    llumnix_config.min_instances_per_model = options.llumnix_min_instances();
    llumnix_config.max_instances_per_model = options.llumnix_max_instances();
    llumnix_config.dispatch_load_metric = options.llumnix_dispatch_load_metric();
    llumnix_config.migration_load_metric = options.llumnix_migration_load_metric();
    llumnix_config.dispatch_policy = options.llumnix_dispatch_policy();
    llumnix_config.migration_policy = options.llumnix_migration_policy();
    llumnix_config.dispatch_busy_threshold = options.llumnix_dispatch_busy_threshold();
    llumnix_config.dispatch_busy_threshold_remaining_steps =
        options.llumnix_dispatch_busy_threshold_remaining_steps();

    auto llumnix_mgr = std::make_shared<LlumnixInstanceMgr>(
        options, etcd_client_, is_master_service_, llumnix_config);
    llumnix_mgr->start_global_scheduler();
    instance_mgr_ = llumnix_mgr;  // polymorphic assignment

    LOG(INFO) << "Scheduler: Llumnix mode enabled";
  } else {
    instance_mgr_ =
        std::make_shared<InstanceMgr>(options, etcd_client_, is_master_service_);
  }

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else if (options.load_balance_policy() == "LST_IMH") {
    lb_policy_ = std::make_unique<LstImhPolicy>(options, instance_mgr_);
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

  // Shut down lb_policy first (unblocks any threads waiting in
  // select_instances_pair, e.g. LstImhPolicy's coordinator).
  if (lb_policy_) {
    lb_policy_->shutdown();
  }

  // Push nullptr sentinels to unblock process_request_queue threads.
  for (auto& queue_pair : request_queues_) {
    queue_pair.second->emplace(nullptr);
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

  // Push request to queue (all policies go through process_request_queue)
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

    // Prism mode: bypass dual-pool, use PrismInstanceMgr dispatch
    if (prism_mode_) {
      process_prism_request(request);
      continue;
    }

    // ServerlessLLM mode: bypass dual-pool, use ServerlessLLMInstanceMgr dispatch
    if (serverless_llm_mode_) {
      process_serverless_llm_request(request);
      continue;
    }

    // Llumnix mode: bypass dual-pool, use LlumnixInstanceMgr dispatch
    if (llumnix_mode_) {
      process_llumnix_request(request);
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
      // Pull-based dispatch: queue in LST-IMH coordinator (Moore-Hodgson with 1 instance)
      if (!lb_policy_->select_instances_pair(request)) {
        LOG(WARNING) << "LB policy failed to assign instance for steady request "
                     << request->service_request_id
                     << " model=" << request->model;
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
          LOG(ERROR) << "dynamic_part_auto_scaling failed to wake model " << request->model
                     << " (request silently dropped!)";
          continue;
        }
        request->routing.prefill_name = awake[0];
        request->routing.decode_name = awake[0];
      } else {
        // Warm elastic model: route first, then trigger scaling adjustment
        if (!lb_policy_->select_instances_pair(request)) {
          LOG(WARNING) << "LB policy failed to assign instance for request "
                       << request->service_request_id
                       << " model=" << request->model;
          continue;
        }
        instance_mgr_->dynamic_part_auto_scaling();
      }
    }

    DLOG(INFO) << request->routing.debug_string();

    // update request metrics (skip if already updated by LstImhPolicy coordinator)
    if (request->prompt.size() != 0 && !request->metrics_already_updated) {
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

void Scheduler::process_prism_request(std::shared_ptr<Request> request) {
  auto prism_mgr = std::static_pointer_cast<PrismInstanceMgr>(instance_mgr_);

  // 1. Register request in Prism tracker
  auto prism_req = std::make_shared<PrismReq>();
  prism_req->rid = request->service_request_id;
  prism_req->model = request->model;
  prism_req->arrival_time = PrismRequestTracker::now_seconds();
  prism_req->slo = (request->ttft_slo_ms > 0)
                       ? request->ttft_slo_ms / 1000.0
                       : 30.0;
  prism_req->prompt_len = static_cast<int32_t>(request->token_ids.size());
  prism_req->state = PrismReqState::WAITING;
  prism_mgr->enqueue_prism_request(request->model, prism_req);

  // 2. Wait for an available instance using CV-based blocking (no busy-poll)
  bool dispatched = prism_mgr->dispatch_prism_request_blocking(request, 30.0);

  if (!dispatched) {
    LOG(ERROR) << "Prism: timeout waiting for model " << request->model
               << " request_id=" << request->service_request_id;
    prism_mgr->finish_prism_request(prism_req->rid);
    return;
  }

  // 3. Mark as running
  prism_mgr->start_prism_running(prism_req->rid, request->routing.prefill_name);

  DLOG(INFO) << "Prism: dispatched " << request->service_request_id
             << " to " << request->routing.prefill_name;

  // 4. Update request metrics
  if (request->prompt.size() != 0 && !request->metrics_already_updated) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  if (request->dispatch_callback) {
    std::thread([request]() { request->dispatch_callback(); }).detach();
  }
}

void Scheduler::process_serverless_llm_request(
    std::shared_ptr<Request> request) {
  auto sllm_mgr =
      std::static_pointer_cast<ServerlessLLMInstanceMgr>(instance_mgr_);

  // 1. Register request in ServerlessLLM tracker
  sllm_mgr->enqueue_request(request->model, request->service_request_id);

  // 2. Wait for an available instance (global scheduler will activate models)
  bool dispatched = false;
  for (int retry = 0; retry < 300 && !exited_; ++retry) {
    if (sllm_mgr->dispatch_request(request)) {
      dispatched = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  if (!dispatched) {
    LOG(ERROR) << "ServerlessLLM: timeout waiting for model " << request->model
               << " request_id=" << request->service_request_id;
    sllm_mgr->finish_request(request->service_request_id);
    return;
  }

  // 3. Mark as running
  sllm_mgr->start_running(request->service_request_id,
                           request->routing.prefill_name);

  DLOG(INFO) << "ServerlessLLM: dispatched " << request->service_request_id
             << " to " << request->routing.prefill_name;

  // 4. Update request metrics
  if (request->prompt.size() != 0 && !request->metrics_already_updated) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  if (request->dispatch_callback) {
    std::thread([request]() { request->dispatch_callback(); }).detach();
  }
}

void Scheduler::process_llumnix_request(std::shared_ptr<Request> request) {
  auto llumnix_mgr =
      std::static_pointer_cast<LlumnixInstanceMgr>(instance_mgr_);

  // 1. Register request in Llumnix tracker
  llumnix_mgr->enqueue_request(request->model, request->service_request_id);

  // 2. Wait for an available instance (global scheduler will activate models)
  bool dispatched = false;
  for (int retry = 0; retry < 300 && !exited_; ++retry) {
    if (llumnix_mgr->dispatch_request(request)) {
      dispatched = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  if (!dispatched) {
    LOG(ERROR) << "Llumnix: timeout waiting for model " << request->model
               << " request_id=" << request->service_request_id;
    llumnix_mgr->finish_request(request->service_request_id);
    return;
  }

  // 3. Mark as running
  llumnix_mgr->start_running(request->service_request_id,
                              request->routing.prefill_name);

  DLOG(INFO) << "Llumnix: dispatched " << request->service_request_id
             << " to " << request->routing.prefill_name;

  // 4. Update request metrics
  if (request->prompt.size() != 0 && !request->metrics_already_updated) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  if (request->dispatch_callback) {
    std::thread([request]() { request->dispatch_callback(); }).detach();
  }
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

  // Notify Prism tracker of request completion
  if (prism_mode_) {
    auto prism_mgr = std::static_pointer_cast<PrismInstanceMgr>(instance_mgr_);
    prism_mgr->finish_prism_request(service_request_id);
  }

  // Notify ServerlessLLM tracker of request completion
  if (serverless_llm_mode_) {
    auto sllm_mgr =
        std::static_pointer_cast<ServerlessLLMInstanceMgr>(instance_mgr_);
    sllm_mgr->finish_request(service_request_id);
  }

  // Notify Llumnix tracker of request completion
  if (llumnix_mode_) {
    auto llumnix_mgr =
        std::static_pointer_cast<LlumnixInstanceMgr>(instance_mgr_);
    llumnix_mgr->finish_request(service_request_id);
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

  // Notify lb_policy on PREFILL_DONE (polymorphic; LstImhPolicy uses this to
  // wake its coordinator, other policies ignore it).
  if (!prefill_instance.empty() && lb_policy_) {
    lb_policy_->on_prefill_done(prefill_instance);
  }

}

}  // namespace xllm_service