#include "scheduler/blitzscale/blitzscale_instance_mgr.h"

#include "common/coro.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace xllm_service {

namespace {

bool contains_instance(const std::vector<std::string>& instances,
                       const std::string& instance_name) {
  return std::find(instances.begin(), instances.end(), instance_name) !=
         instances.end();
}

}  // namespace

BlitzScaleInstanceMgr::BlitzScaleInstanceMgr(
    const Options& options,
    const std::shared_ptr<EtcdClient>& etcd_client,
    bool is_master_service,
    const BlitzScaleConfig& config)
    : InstanceMgr(options, etcd_client, is_master_service), config_(config) {
  LOG(INFO) << "BlitzScaleInstanceMgr created with"
            << " schedule_interval=" << config_.schedule_interval_s << "s"
            << " scale_down_threshold_ms=" << config_.scale_down_threshold_ms
            << " tokens_prefilled_per_sec="
            << config_.tokens_prefilled_per_sec
            << " tokens_transferred_per_sec="
            << config_.tokens_transferred_per_sec
            << " max_blocks_per_replica=" << config_.max_blocks_per_replica
            << " min_prefill=" << config_.min_prefill_instances
            << " max_prefill=" << config_.max_prefill_instances
            << " min_decode=" << config_.min_decode_instances
            << " max_decode=" << config_.max_decode_instances;

  // Pre-initialize per-model entries so map structure never changes at runtime.
  for (const auto& [model_id, _] : MODELS) {
    pending_queues_[model_id] =
        std::make_shared<ConcurrentQueue<std::shared_ptr<Request>>>();
    idle_prefill_queues_[model_id] =
        std::make_shared<ConcurrentQueue<std::string>>();
    pending_tokens_[model_id] =
        std::make_shared<std::atomic<int32_t>>(0);
  }
}

BlitzScaleInstanceMgr::~BlitzScaleInstanceMgr() {
  stop_global_scheduler();
}

void BlitzScaleInstanceMgr::enqueue_request(const std::string& model,
                                            const std::string& request_id,
                                            int32_t prompt_tokens,
                                            double enqueue_time) {
  std::unique_lock<std::shared_mutex> lock(req_mutex_);
  auto info = std::make_shared<ReqInfo>();
  info->rid = request_id;
  info->model = model;
  info->prompt_tokens = prompt_tokens;
  info->prompt_blocks = get_request_blocks(prompt_tokens);
  info->enqueue_time = enqueue_time;
  requests_[request_id] = info;
  model_requests_[model].push_back(request_id);
}

void BlitzScaleInstanceMgr::add_pending_request(
    std::shared_ptr<Request> request) {
  const double enqueue_time = now_seconds();
  const int32_t token_count =
      static_cast<int32_t>(request->token_ids.size());

  pending_tokens_[request->model]->fetch_add(token_count,
                                             std::memory_order_relaxed);
  pending_queues_[request->model]->push(request);

  enqueue_request(request->model,
                  request->service_request_id,
                  token_count,
                  enqueue_time);

  dispatch_notify_.push(request->model);
}

void BlitzScaleInstanceMgr::enqueue_idle_prefill(
    const std::string& model_id, const std::string& instance_name) {
  idle_prefill_queues_[model_id]->push(instance_name);
  dispatch_notify_.push(model_id);
}

void BlitzScaleInstanceMgr::notify_prefill_done(
    const std::string& prefill_instance) {
  // Find which model(s) this instance serves as prefill, then re-enqueue it
  // as idle so the dispatch thread can pair it with the next pending request.
  std::vector<std::string> models;
  {
    std::lock_guard<std::mutex> lk(role_mutex_);
    for (const auto& [model_id, instances] : prefill_instances_) {
      if (instances.count(prefill_instance)) {
        models.push_back(model_id);
      }
    }
  }
  for (const auto& model_id : models) {
    enqueue_idle_prefill(model_id, prefill_instance);
  }
}

void BlitzScaleInstanceMgr::start_running(const std::string& request_id,
                                          const std::string& prefill_instance,
                                          const std::string& decode_instance) {
  std::string model;
  {
    std::unique_lock<std::shared_mutex> lock(req_mutex_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
      return;
    }
    it->second->state = ReqInfo::RUNNING;
    it->second->prefill_instance = prefill_instance;
    it->second->decode_instance = decode_instance;
    model = it->second->model;

    // Maintain per-instance counters (atomic, under exclusive lock for
    // map-structural safety; value updates are relaxed atomics).
    auto& pc = instance_prefill_count_[prefill_instance];
    if (!pc) pc = std::make_unique<std::atomic<int32_t>>(0);
    pc->fetch_add(1, std::memory_order_relaxed);

    auto& dc = instance_decode_count_[decode_instance];
    if (!dc) dc = std::make_unique<std::atomic<int32_t>>(0);
    dc->fetch_add(1, std::memory_order_relaxed);
  }
  // Update last_used without holding req_mutex_ (eliminates nested lock).
  const double now = now_seconds();
  std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
  model_last_used_[model][prefill_instance] = now;
  model_last_used_[model][decode_instance] = now;
}

void BlitzScaleInstanceMgr::finish_request(const std::string& request_id) {
  std::string model;
  std::string prefill_instance;
  std::string decode_instance;
  {
    std::unique_lock<std::shared_mutex> lock(req_mutex_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
      return;
    }

    model = it->second->model;
    prefill_instance = it->second->prefill_instance;
    decode_instance = it->second->decode_instance;

    // Decrement per-instance counters.
    if (!prefill_instance.empty()) {
      auto cit = instance_prefill_count_.find(prefill_instance);
      if (cit != instance_prefill_count_.end()) {
        cit->second->fetch_sub(1, std::memory_order_relaxed);
      }
    }
    if (!decode_instance.empty()) {
      auto cit = instance_decode_count_.find(decode_instance);
      if (cit != instance_decode_count_.end()) {
        cit->second->fetch_sub(1, std::memory_order_relaxed);
      }
    }

    auto& model_reqs = model_requests_[model];
    model_reqs.erase(
        std::remove(model_reqs.begin(), model_reqs.end(), request_id),
        model_reqs.end());
    if (model_reqs.empty()) {
      model_requests_.erase(model);
    }

    requests_.erase(it);
  }

  // Update last_used without holding req_mutex_ (eliminates nested lock).
  if (!prefill_instance.empty() || !decode_instance.empty()) {
    const double now = now_seconds();
    std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
    if (!prefill_instance.empty()) {
      model_last_used_[model][prefill_instance] = now;
    }
    if (!decode_instance.empty()) {
      model_last_used_[model][decode_instance] = now;
    }
  }

  if (!model.empty()) {
    dispatch_notify_.push(model);
  }
}

bool BlitzScaleInstanceMgr::dispatch_request(std::shared_ptr<Request> request) {
  auto prefill = select_prefill_instance(request->model);
  if (!prefill.has_value()) {
    return false;
  }

  auto decode = select_decode_instance(request->model,
                                       get_request_blocks(request->token_ids.size()));
  if (!decode.has_value()) {
    decode = prefill;
  }

  if (!decode.has_value()) {
    return false;
  }

  request->routing.prefill_name = *prefill;
  request->routing.decode_name = *decode;

  std::unique_lock<std::shared_mutex> lock(req_mutex_);
  auto it = requests_.find(request->service_request_id);
  if (it != requests_.end()) {
    it->second->prefill_instance = *prefill;
    it->second->decode_instance = *decode;
  }
  return true;
}

bool BlitzScaleInstanceMgr::dispatch_with_prefill(
    std::shared_ptr<Request> request,
    const std::string& prefill_instance) {
  // Reject dispatch if the instance hasn't finished waking up yet.
  // execute_activate enqueues the instance immediately but send_model_wakeup
  // is async; the instance will be re-enqueued via notify_prefill_done once
  // it actually processes its first request.
  std::optional<std::string> decode;
  if (request->max_tokens == 1) {
    decode = prefill_instance;  // Prefill-only request; decode is a no-op. Skip checks.
  }
  if (!decode.has_value()) {
    auto awake = get_awake_instances(request->model);
    if (!contains_instance(awake, prefill_instance)) {
      return false;
    }

    decode = select_decode_instance(
        request->model, get_request_blocks(request->token_ids.size()));
    if (!decode.has_value()) {
      decode = prefill_instance;
    }
  }

  request->routing.prefill_name = prefill_instance;
  request->routing.decode_name = *decode;

  std::unique_lock<std::shared_mutex> lock(req_mutex_);
  auto it = requests_.find(request->service_request_id);
  if (it != requests_.end()) {
    it->second->prefill_instance = prefill_instance;
    it->second->decode_instance = *decode;
  }
  return true;
}

void BlitzScaleInstanceMgr::start_global_scheduler() {
  if (running_.exchange(true)) {
    return;
  }
  sched_thread_ =
      std::make_unique<std::thread>(&BlitzScaleInstanceMgr::scheduling_loop,
                                    this);
  dispatch_thread_ =
      std::make_unique<std::thread>(&BlitzScaleInstanceMgr::dispatch_loop,
                                    this);
  LOG(INFO) << "BlitzScale global scheduler started";
}

void BlitzScaleInstanceMgr::stop_global_scheduler() {
  if (!running_.exchange(false)) {
    return;
  }
  dispatch_notify_.push("");  // unblock dispatch_loop's blocking pop
  if (sched_thread_ && sched_thread_->joinable()) {
    sched_thread_->join();
  }
  if (dispatch_thread_ && dispatch_thread_->joinable()) {
    dispatch_thread_->join();
  }
  LOG(INFO) << "BlitzScale global scheduler stopped";
}

void BlitzScaleInstanceMgr::scheduling_loop() {
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(config_.schedule_interval_s * 1000)));
    if (!running_.load()) {
      break;
    }

    sync_role_state();
    auto actions = gen_actions();
    if (!actions.empty()) {
      LOG(INFO) << "BlitzScale scheduler: executing " << actions.size()
                << " actions";
      execute_actions(actions);
    }
    // New capacity may have been created; wake dispatch thread for all models
    for (const auto& [model_id, _] : MODELS) {
      dispatch_notify_.push(model_id);
    }
  }
}

void BlitzScaleInstanceMgr::dispatch_loop() {
  while (running_.load()) {
    std::string model = dispatch_notify_.pop();  // blocks until notified
    if (!running_.load()) break;

    // Drain additional pending notifications to avoid redundant dispatch calls
    std::string extra;
    while (dispatch_notify_.try_pop(extra)) {
      if (extra.empty()) break;  // shutdown sentinel re-encountered
    }

    dispatch_pending_requests();
  }
}

void BlitzScaleInstanceMgr::dispatch_pending_requests() {
  for (auto& [model, idle_q] : idle_prefill_queues_) {
    auto& pq = pending_queues_[model];
    while (true) {
      auto prefill_opt = idle_q->try_front();
      if (!prefill_opt) break;
      auto front_opt = pq->try_front();
      if (!front_opt) break;

      const std::string& prefill_instance = *prefill_opt;
      const auto& front = *front_opt;

      // Check timeout via enqueue_time stored in ReqInfo (shared read lock).
      double enqueue_time = 0.0;
      {
        std::shared_lock<std::shared_mutex> lk(req_mutex_);
        auto it = requests_.find(front->service_request_id);
        if (it != requests_.end()) {
          enqueue_time = it->second->enqueue_time;
        }
      }
      if (enqueue_time > 0.0 && now_seconds() - enqueue_time > 30.0) {
        // Timeout: remove request, leave idle instance for next request
        std::shared_ptr<Request> timed_out;
        if (pq->try_pop(timed_out)) {
          pending_tokens_[model]->fetch_sub(
              static_cast<int32_t>(timed_out->token_ids.size()),
              std::memory_order_relaxed);
        }
        LOG(ERROR) << "BlitzScale: timeout waiting for model " << model
                   << " request_id=" << front->service_request_id;
        finish_request(front->service_request_id);
        continue;
      }

      // Pull model: dispatch the front request to the specific idle instance
      if (!dispatch_with_prefill(front, prefill_instance)) {
        break;  // No decode capacity; wait for next wakeup
      }

      // Successfully dispatched — consume the idle instance and the request.
      // pop_front_if may no-op if execute_deactivate erased it concurrently.
      idle_q->pop_front_if(prefill_instance);
      // Only dispatch thread pops from pq, so front is stable.
      std::shared_ptr<Request> consumed;
      if (pq->try_pop(consumed)) {
        pending_tokens_[model]->fetch_sub(
            static_cast<int32_t>(consumed->token_ids.size()),
            std::memory_order_relaxed);
      }

      start_running(front->service_request_id,
                    front->routing.prefill_name,
                    front->routing.decode_name);

      DLOG(INFO) << "BlitzScale: dispatched " << front->service_request_id
                 << " prefill=" << front->routing.prefill_name
                 << " decode=" << front->routing.decode_name;

      if (!front->prompt.empty() && !front->metrics_already_updated) {
        update_request_metrics(front, RequestAction::SCHEDULE);
      }

      if (front->dispatch_callback) {
        Coroutine(front->dispatch_callback(), /*detach=*/true);
      }
    }
  }
}

void BlitzScaleInstanceMgr::sync_role_state() {
  // Collect instances newly assigned to prefill so we can enqueue them as idle
  // after releasing role_mutex_ (enqueue_idle_prefill must not be called under it).
  std::vector<std::pair<std::string, std::string>> new_prefill;

  {
    std::lock_guard<std::mutex> lock(role_mutex_);
    instance_to_models_.clear();
    model_to_instances_.clear();

    for (const auto& [model_id, _] : MODELS) {
      auto awake = get_awake_instances(model_id);
      std::unordered_set<std::string> awake_set(awake.begin(), awake.end());

      auto& prefill_set = prefill_instances_[model_id];
      auto& decode_set = decode_instances_[model_id];

      for (auto it = prefill_set.begin(); it != prefill_set.end();) {
        if (!awake_set.count(*it)) {
          it = prefill_set.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = decode_set.begin(); it != decode_set.end();) {
        if (!awake_set.count(*it)) {
          it = decode_set.erase(it);
        } else {
          ++it;
        }
      }

      for (const auto& instance_name : awake) {
        const bool in_prefill = prefill_set.count(instance_name) > 0;
        const bool in_decode = decode_set.count(instance_name) > 0;

        if (in_prefill && in_decode) {
          if (prefill_set.size() <= decode_set.size()) {
            decode_set.erase(instance_name);
          } else {
            prefill_set.erase(instance_name);
          }
          continue;
        }

        if (!in_prefill && !in_decode) {
          const bool need_prefill =
              static_cast<int32_t>(prefill_set.size()) <
              config_.min_prefill_instances;
          const bool need_decode =
              static_cast<int32_t>(decode_set.size()) <
              config_.min_decode_instances;

          if (need_prefill && !need_decode) {
            prefill_set.insert(instance_name);
            new_prefill.emplace_back(model_id, instance_name);
          } else if (!need_prefill && need_decode) {
            decode_set.insert(instance_name);
          } else if (prefill_set.size() <= decode_set.size()) {
            prefill_set.insert(instance_name);
            new_prefill.emplace_back(model_id, instance_name);
          } else {
            decode_set.insert(instance_name);
          }
        }
      }

      for (const auto& instance_name : prefill_set) {
        instance_to_models_[instance_name].insert(model_id);
        model_to_instances_[model_id].insert(instance_name);
      }
      for (const auto& instance_name : decode_set) {
        instance_to_models_[instance_name].insert(model_id);
        model_to_instances_[model_id].insert(instance_name);
      }
    }
  }

  for (const auto& [model_id, instance_name] : new_prefill) {
    enqueue_idle_prefill(model_id, instance_name);
  }
}

std::vector<BlitzScaleInstanceMgr::BlitzAction>
BlitzScaleInstanceMgr::gen_actions() {
  std::vector<BlitzAction> actions;
  for (const auto& [model_id, _] : MODELS) {
    auto model_actions = gen_model_actions(model_id);
    actions.insert(actions.end(), model_actions.begin(), model_actions.end());
  }

  std::stable_sort(actions.begin(), actions.end(),
                   [](const BlitzAction& lhs, const BlitzAction& rhs) {
                     auto priority = [](BlitzAction::Type type) {
                       switch (type) {
                         case BlitzAction::FLIP_TO_PREFILL:
                         case BlitzAction::FLIP_TO_DECODE:
                           return 0;
                         case BlitzAction::DEACTIVATE_PREFILL:
                         case BlitzAction::DEACTIVATE_DECODE:
                           return 1;
                         case BlitzAction::ACTIVATE_PREFILL:
                         case BlitzAction::ACTIVATE_DECODE:
                           return 2;
                       }
                       return 3;
                     };
                     return priority(lhs.type) < priority(rhs.type);
                   });
  return actions;
}

std::vector<BlitzScaleInstanceMgr::BlitzAction>
BlitzScaleInstanceMgr::gen_model_actions(const std::string& model_id) {
  std::vector<BlitzAction> actions;

  const int32_t waiting_count = get_waiting_count(model_id);
  const auto prefill_instances = get_role_instances(model_id, Role::PREFILL);
  const auto decode_instances = get_role_instances(model_id, Role::DECODE);
  const int32_t current_prefill = static_cast<int32_t>(prefill_instances.size());
  const int32_t current_decode = static_cast<int32_t>(decode_instances.size());

  if (!engaged_model(model_id, waiting_count, current_prefill, current_decode)) {
    return actions;
  }

  

  // const int32_t active_count = get_active_count(model_id);
  const int32_t waiting_prefill_tokens = get_waiting_prefill_tokens(model_id);
  const int32_t waiting_decode_blocks =
      /* (active_count > 0) ?  */compute_waiting_decode_blocks(model_id)/*  : 0 */;
  const int32_t prefill_tokens =
     /*  (active_count > 0) ?  */compute_prefill_tokens(model_id)/*  : 0 */;

  LOG(INFO) << "BlitzScale: planning actions for model " << model_id
            << " waiting_count=" << waiting_count
            << " current_prefill=" << current_prefill
            << " current_decode=" << current_decode
            << " waiting_prefill_tokens=" << waiting_prefill_tokens
            << " waiting_decode_blocks=" << waiting_decode_blocks
            << " prefill_tokens=" << prefill_tokens;

  auto [delta_prefill, delta_decode] =
      compute_scale_plan(model_id,
                         current_prefill,
                         current_decode,
                         waiting_prefill_tokens,
                         waiting_decode_blocks,
                         prefill_tokens);

  LOG(INFO) << "BlitzScale: computed scale plan for model " << model_id
            << " delta_prefill=" << delta_prefill
            << " delta_decode=" << delta_decode;

  while (delta_prefill > 0 && delta_decode < 0) {
    auto flip = select_flip_instance(model_id, Role::DECODE);
    if (!flip.has_value()) {
      break;
    }
    actions.push_back(
        {BlitzAction::FLIP_TO_PREFILL, model_id, flip.value()});
    {
      std::lock_guard<std::mutex> lock(role_mutex_);
      assign_role_locked(model_id, flip.value(), Role::PREFILL);
    }
    --delta_prefill;
    ++delta_decode;
  }

  while (delta_decode > 0 && delta_prefill < 0) {
    auto flip = select_flip_instance(model_id, Role::PREFILL);
    if (!flip.has_value()) {
      break;
    }
    actions.push_back(
        {BlitzAction::FLIP_TO_DECODE, model_id, flip.value()});
    {
      std::lock_guard<std::mutex> lock(role_mutex_);
      assign_role_locked(model_id, flip.value(), Role::DECODE);
    }
    --delta_decode;
    ++delta_prefill;
  }

  for (int32_t i = 0; i < delta_prefill; ++i) {
    auto instance_name = select_activation_instance(model_id);
    if (!instance_name.has_value()) {
      break;
    }
    actions.push_back(
        {BlitzAction::ACTIVATE_PREFILL, model_id, instance_name.value()});
    {
      std::lock_guard<std::mutex> lock(role_mutex_);
      assign_role_locked(model_id, instance_name.value(), Role::PREFILL);
      instance_to_models_[instance_name.value()].insert(model_id);
      model_to_instances_[model_id].insert(instance_name.value());
    }
  }
  for (int32_t i = 0; i < delta_decode; ++i) {
    auto instance_name = select_activation_instance(model_id);
    if (!instance_name.has_value()) {
      break;
    }
    actions.push_back(
        {BlitzAction::ACTIVATE_DECODE, model_id, instance_name.value()});
    {
      std::lock_guard<std::mutex> lock(role_mutex_);
      assign_role_locked(model_id, instance_name.value(), Role::DECODE);
      instance_to_models_[instance_name.value()].insert(model_id);
      model_to_instances_[model_id].insert(instance_name.value());
    }
  }

  for (int32_t i = 0; i < -delta_prefill; ++i) {
    auto instance_name = select_scale_down_instance(model_id, Role::PREFILL);
    if (!instance_name.has_value()) {
      break;
    }
    actions.push_back(
        {BlitzAction::DEACTIVATE_PREFILL, model_id, instance_name.value()});
  }
  for (int32_t i = 0; i < -delta_decode; ++i) {
    auto instance_name = select_scale_down_instance(model_id, Role::DECODE);
    if (!instance_name.has_value()) {
      break;
    }
    actions.push_back(
        {BlitzAction::DEACTIVATE_DECODE, model_id, instance_name.value()});
  }

  return actions;
}

void BlitzScaleInstanceMgr::execute_actions(
    const std::vector<BlitzAction>& actions) {
  for (const auto& action : actions) {
    switch (action.type) {
      case BlitzAction::ACTIVATE_PREFILL:
        execute_activate(action.model_id, action.instance_name, Role::PREFILL);
        break;
      case BlitzAction::ACTIVATE_DECODE:
        execute_activate(action.model_id, action.instance_name, Role::DECODE);
        break;
      case BlitzAction::DEACTIVATE_PREFILL:
        execute_deactivate(action.model_id,
                           action.instance_name,
                           Role::PREFILL);
        break;
      case BlitzAction::DEACTIVATE_DECODE:
        execute_deactivate(action.model_id,
                           action.instance_name,
                           Role::DECODE);
        break;
      case BlitzAction::FLIP_TO_PREFILL:
        execute_flip(action.model_id, action.instance_name, Role::PREFILL);
        break;
      case BlitzAction::FLIP_TO_DECODE:
        execute_flip(action.model_id, action.instance_name, Role::DECODE);
        break;
    }
  }
}

void BlitzScaleInstanceMgr::execute_activate(const std::string& model_id,
                                             const std::string& instance_name,
                                             Role role) {
  LOG(INFO) << "BlitzScale: activating " << model_id << " on "
            << instance_name << " as "
            << (role == Role::PREFILL ? "prefill" : "decode");

  uint64_t model_bytes = get_model_size_bytes(model_id);
  deduct_free_pages(instance_name, model_bytes);
  // Transition model state from SLEEP to ALLOCATED before wakeup
  auto model_mgr = get_model_instance_mgr(model_id);
  model_mgr->set_model_state(instance_name, ModelState::ALLOCATED);

  // Use inherited send_model_wakeup (D2D with H2D fallback)
  send_model_wakeup(instance_name, model_id,
                    /*memory_increased_in_advance=*/false);

  {
    std::lock_guard<std::mutex> lock(role_mutex_);
    assign_role_locked(model_id, instance_name, role);
    instance_to_models_[instance_name].insert(model_id);
    model_to_instances_[model_id].insert(instance_name);
  }

  {
    std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
    model_last_used_[model_id][instance_name] = now_seconds();
  }

  if (role == Role::PREFILL) {
    enqueue_idle_prefill(model_id, instance_name);
  }
}

void BlitzScaleInstanceMgr::execute_deactivate(const std::string& model_id,
                                               const std::string& instance_name,
                                               Role role) {
  LOG(INFO) << "BlitzScale: deactivating " << model_id << " on "
            << instance_name << " from "
            << (role == Role::PREFILL ? "prefill" : "decode");

  {
    std::lock_guard<std::mutex> lock(role_mutex_);
    remove_role_locked(model_id, instance_name, role);
    auto inst_it = instance_to_models_.find(instance_name);
    if (inst_it != instance_to_models_.end()) {
      inst_it->second.erase(model_id);
      if (inst_it->second.empty()) {
        instance_to_models_.erase(inst_it);
      }
    }
    auto model_it = model_to_instances_.find(model_id);
    if (model_it != model_to_instances_.end()) {
      model_it->second.erase(instance_name);
      if (model_it->second.empty()) {
        model_to_instances_.erase(model_it);
      }
    }
  }

  {
    std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
    auto model_it = model_last_used_.find(model_id);
    if (model_it != model_last_used_.end()) {
      model_it->second.erase(instance_name);
      if (model_it->second.empty()) {
        model_last_used_.erase(model_it);
      }
    }
  }

  if (role == Role::PREFILL) {
    auto it = idle_prefill_queues_.find(model_id);
    if (it != idle_prefill_queues_.end()) {
      it->second->erase_first(instance_name);
    }
  }

  std::thread([this, model_id, instance_name]() {
    wait_for_model_drain(instance_name, model_id);
    auto mgr = get_model_instance_mgr(model_id);
    if (mgr && !mgr->can_sleep(instance_name)) {
      LOG(WARNING) << "BlitzScale: skipping sleep of " << model_id << " on "
                   << instance_name << " (D2D locked)";
      return;
    }
    send_model_sleep(instance_name, model_id);
  }).detach();
}

void BlitzScaleInstanceMgr::execute_flip(const std::string& model_id,
                                         const std::string& instance_name,
                                         Role role) {
  LOG(INFO) << "BlitzScale: flipping " << model_id << " on " << instance_name
            << " to " << (role == Role::PREFILL ? "prefill" : "decode");

  {
    std::lock_guard<std::mutex> lock(role_mutex_);
    assign_role_locked(model_id, instance_name, role);
    instance_to_models_[instance_name].insert(model_id);
    model_to_instances_[model_id].insert(instance_name);
  }

  if (role == Role::PREFILL) {
    enqueue_idle_prefill(model_id, instance_name);
  }
}

std::vector<std::string> BlitzScaleInstanceMgr::get_all_instance_names() {
  std::vector<std::string> result;
  for (const auto& [model_id, _] : MODELS) {
    auto mgr = get_model_instance_mgr(model_id);
    if (!mgr) {
      continue;
    }
    auto names = mgr->get_all_instance_names();
    for (const auto& name : names) {
      if (!contains_instance(result, name)) {
        result.push_back(name);
      }
    }
  }
  return result;
}

int32_t BlitzScaleInstanceMgr::get_waiting_count(const std::string& model_id) {
  auto it = pending_queues_.find(model_id);
  if (it == pending_queues_.end()) {
    return 0;
  }
  return static_cast<int32_t>(it->second->size());
}

int32_t BlitzScaleInstanceMgr::get_waiting_prefill_tokens(
    const std::string& model_id) const {
  auto it = pending_tokens_.find(model_id);
  if (it == pending_tokens_.end()) {
    return 0;
  }
  return it->second->load(std::memory_order_relaxed);
}

int32_t BlitzScaleInstanceMgr::get_active_count(const std::string& model_id) {
  std::shared_lock<std::shared_mutex> lock(req_mutex_);
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) {
    return 0;
  }
  return static_cast<int32_t>(it->second.size());
}

int32_t BlitzScaleInstanceMgr::get_request_blocks(int32_t prompt_tokens) const {
  const int32_t block_size = std::max<int32_t>(1, config_.block_size);
  return std::max<int32_t>(1, (prompt_tokens + block_size - 1) / block_size);
}

int32_t BlitzScaleInstanceMgr::estimate_used_blocks(
    const std::string& instance_name) {
  auto lm = get_instance_load_metrics(instance_name);
  if (!lm.has_value()) {
    return 0;
  }
  if (lm->num_total_gpu_blocks > 0 && lm->num_used_gpu_blocks > 0) {
    return static_cast<int32_t>(lm->num_used_gpu_blocks);
  }
  if (lm->num_total_gpu_blocks > 0) {
    return static_cast<int32_t>(std::round(
        lm->gpu_cache_usage_perc * lm->num_total_gpu_blocks));
  }
  return static_cast<int32_t>(std::round(
      lm->gpu_cache_usage_perc * config_.max_blocks_per_replica));
}

int32_t BlitzScaleInstanceMgr::get_prefill_load(
    const std::string& instance_name) {
  // Shared lock only to protect map-structural access; value is an atomic.
  std::shared_lock<std::shared_mutex> lock(req_mutex_);
  auto it = instance_prefill_count_.find(instance_name);
  if (it == instance_prefill_count_.end()) return 0;
  return it->second->load(std::memory_order_relaxed);
}

int32_t BlitzScaleInstanceMgr::get_decode_load(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(req_mutex_);
  auto it = instance_decode_count_.find(instance_name);
  if (it == instance_decode_count_.end()) return 0;
  return it->second->load(std::memory_order_relaxed);
}

std::vector<std::string> BlitzScaleInstanceMgr::get_role_instances(
    const std::string& model_id, Role role) {
  std::lock_guard<std::mutex> lock(role_mutex_);
  const auto& table =
      (role == Role::PREFILL) ? prefill_instances_ : decode_instances_;
  auto it = table.find(model_id);
  if (it == table.end()) {
    return {};
  }
  return std::vector<std::string>(it->second.begin(), it->second.end());
}

std::vector<std::string> BlitzScaleInstanceMgr::get_dispatch_candidates(
    const std::string& model_id, Role role) {
  auto candidates = get_role_instances(model_id, role);
  auto awake = get_awake_instances(model_id);

  std::vector<std::string> active;
  for (const auto& instance_name : candidates) {
    if (contains_instance(awake, instance_name)) {
      active.push_back(instance_name);
    }
  }
  if (!active.empty()) {
    return active;
  }

  auto fallback = get_role_instances(
      model_id, role == Role::PREFILL ? Role::DECODE : Role::PREFILL);
  for (const auto& instance_name : fallback) {
    if (contains_instance(awake, instance_name)) {
      active.push_back(instance_name);
    }
  }
  return active;
}

std::optional<std::string> BlitzScaleInstanceMgr::select_prefill_instance(
    const std::string& model_id) {
  auto candidates = get_dispatch_candidates(model_id, Role::PREFILL);
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::sort(candidates.begin(), candidates.end(),
            [this](const std::string& lhs, const std::string& rhs) {
              int64_t lhs_done = get_estimated_prefill_done_time(lhs);
              int64_t rhs_done = get_estimated_prefill_done_time(rhs);
              if (lhs_done != rhs_done) {
                return lhs_done < rhs_done;
              }
              return estimate_used_blocks(lhs) < estimate_used_blocks(rhs);
            });
  return candidates.front();
}

std::optional<std::string> BlitzScaleInstanceMgr::select_decode_instance(
    const std::string& model_id, int32_t request_blocks) {
  auto candidates = get_dispatch_candidates(model_id, Role::DECODE);
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::sort(candidates.begin(), candidates.end(),
            [this](const std::string& lhs, const std::string& rhs) {
              return estimate_used_blocks(lhs) < estimate_used_blocks(rhs);
            });

  const int32_t capacity_limit = static_cast<int32_t>(
      std::floor(config_.decode_upper_bound * config_.max_blocks_per_replica));
  for (const auto& instance_name : candidates) {
    if (estimate_used_blocks(instance_name) + request_blocks <= capacity_limit) {
      return instance_name;
    }
  }
  return candidates.front();
}

std::optional<std::string> BlitzScaleInstanceMgr::select_activation_instance(
    const std::string& model_id) {
  auto all_instances = get_all_instance_names();
  if (all_instances.empty()) {
    return std::nullopt;
  }

  std::vector<std::tuple<int32_t, int32_t, std::string>> candidates;
  {
    std::lock_guard<std::mutex> lock(role_mutex_);
    for (const auto& instance_name : all_instances) {
      if (!has_valid_xtensor_info(instance_name)) {
        continue;
      }

      auto active_models_it = instance_to_models_.find(instance_name);
      int32_t colocated_models =
          active_models_it == instance_to_models_.end()
              ? 0
              : static_cast<int32_t>(active_models_it->second.size());
      if (active_models_it != instance_to_models_.end() &&
          active_models_it->second.count(model_id) > 0) {
        continue;
      }
      if (colocated_models >= config_.max_models_per_instance) {
        continue;
      }
      if (!has_enough_space_for_model(instance_name, model_id)) {
        continue;
      }

      candidates.emplace_back(colocated_models,
                              estimate_used_blocks(instance_name),
                              instance_name);
    }
  }

  if (candidates.empty()) {
    return std::nullopt;
  }

  std::sort(candidates.begin(), candidates.end());
  return std::get<2>(candidates.front());
}

std::optional<std::string> BlitzScaleInstanceMgr::select_scale_down_instance(
    const std::string& model_id, Role role) {
  auto candidates = get_role_instances(model_id, role);
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::sort(candidates.begin(), candidates.end(),
            [this, role](const std::string& lhs, const std::string& rhs) {
              const int32_t lhs_blocks = estimate_used_blocks(lhs);
              const int32_t rhs_blocks = estimate_used_blocks(rhs);
              if (lhs_blocks != rhs_blocks) {
                return lhs_blocks < rhs_blocks;
              }
              if (role == Role::PREFILL) {
                return get_prefill_load(lhs) < get_prefill_load(rhs);
              }
              return get_decode_load(lhs) < get_decode_load(rhs);
            });
  return candidates.front();
}

std::optional<std::string> BlitzScaleInstanceMgr::select_flip_instance(
    const std::string& model_id, Role from_role) {
  return select_scale_down_instance(model_id, from_role);
}

int32_t BlitzScaleInstanceMgr::compute_waiting_decode_blocks(
    const std::string& model_id) {
  std::set<std::string> active_instances;
  {
    std::lock_guard<std::mutex> lock(role_mutex_);
    auto prefill_it = prefill_instances_.find(model_id);
    if (prefill_it != prefill_instances_.end()) {
      active_instances.insert(prefill_it->second.begin(), prefill_it->second.end());
    }
    auto decode_it = decode_instances_.find(model_id);
    if (decode_it != decode_instances_.end()) {
      active_instances.insert(decode_it->second.begin(), decode_it->second.end());
    }
  }

  int32_t total = 0;
  for (const auto& instance_name : active_instances) {
    total += estimate_used_blocks(instance_name);
  }
  return total;
}

int32_t BlitzScaleInstanceMgr::compute_prefill_tokens(
    const std::string& model_id) {
  int32_t total = 0;
  auto prefill_instances = get_role_instances(model_id, Role::PREFILL);
  for (const auto& instance_name : prefill_instances) {
    total += estimate_used_blocks(instance_name) * config_.block_size;
  }
  return total;
}

std::pair<int32_t, int32_t> BlitzScaleInstanceMgr::compute_scale_plan(
    const std::string& model_id,
    int32_t current_prefill,
    int32_t current_decode,
    int32_t waiting_prefill_tokens,
    int32_t waiting_decode_blocks,
    int32_t prefill_tokens) {
  const int32_t effective_prefill = std::max(1, current_prefill);
  const int32_t effective_decode = std::max(1, current_decode);

  const double decode_high = std::max(0.01, config_.decode_upper_bound);
  const double decode_low = std::max(0.01, config_.decode_lower_bound);
  const double migration_high = std::max(0.01, config_.migration_upper_bound);
  const double migration_low = std::max(0.01, config_.migration_lower_bound);
  const double prefill_high = std::max(0.01, config_.prefill_upper_bound);
  const double prefill_low = std::max(0.01, config_.prefill_lower_bound);

  const int32_t min_decode_mem = static_cast<int32_t>(std::ceil(
      waiting_decode_blocks /
      (decode_high * std::max<uint32_t>(1, config_.max_blocks_per_replica))));
  const int32_t max_decode_mem = static_cast<int32_t>(std::ceil(
      waiting_decode_blocks /
      (decode_low * std::max<uint32_t>(1, config_.max_blocks_per_replica))));

  const uint32_t tokens_consumed_per_sec = std::max<uint32_t>(
      1, effective_decode * config_.tokens_transferred_per_sec);
  const int32_t min_decode_kv = static_cast<int32_t>(std::ceil(
      prefill_tokens / (migration_high * tokens_consumed_per_sec)));
  const int32_t max_decode_kv = static_cast<int32_t>(std::ceil(
      prefill_tokens / (migration_low * tokens_consumed_per_sec)));

  const int32_t min_prefill_kv = static_cast<int32_t>(std::ceil(
      prefill_tokens / (migration_high * tokens_consumed_per_sec)));
  const int32_t max_prefill_kv = static_cast<int32_t>(std::ceil(
      prefill_tokens / (migration_low * tokens_consumed_per_sec)));

  const int32_t min_prefill_thpt = static_cast<int32_t>(std::ceil(
      waiting_prefill_tokens /
      (prefill_high * std::max<uint32_t>(1, config_.tokens_prefilled_per_sec))));
  const int32_t max_prefill_thpt = static_cast<int32_t>(std::ceil(
      waiting_prefill_tokens /
      (prefill_low * std::max<uint32_t>(1, config_.tokens_prefilled_per_sec))));

  int32_t desired_decode = effective_decode;
  int32_t min_decode = std::max(min_decode_mem, min_decode_kv);
  int32_t max_decode = std::max({max_decode_mem, max_decode_kv, min_decode});
  if (current_decode == 0) {
    desired_decode = std::max(1, config_.min_decode_instances);
  } else if (min_decode > current_decode) {
    desired_decode = (min_decode + max_decode + 1) / 2;
  } else if (max_decode < current_decode) {
    desired_decode = max_decode;
  }

  int32_t desired_prefill = effective_prefill;
  int32_t min_prefill = std::max(min_prefill_thpt, min_prefill_kv);
  int32_t max_prefill = std::max({max_prefill_thpt, max_prefill_kv, min_prefill});
  if (current_prefill == 0) {
    desired_prefill = std::max(1, config_.min_prefill_instances);
  } else if (min_prefill > current_prefill) {
    desired_prefill = (min_prefill + max_prefill + 1) / 2;
  } else if (max_prefill < current_prefill) {
    desired_prefill = max_prefill;
  }

  if (waiting_prefill_tokens > 0 || waiting_decode_blocks > 0 ||
      get_active_count(model_id) > 0) {
    desired_prefill =
        std::clamp(desired_prefill,
                   std::max(1, config_.min_prefill_instances),
                   std::max(config_.min_prefill_instances,
                            config_.max_prefill_instances));
    desired_decode =
        std::clamp(desired_decode,
                   std::max(1, config_.min_decode_instances),
                   std::max(config_.min_decode_instances,
                            config_.max_decode_instances));
  }

  int32_t delta_prefill = desired_prefill - current_prefill;
  int32_t delta_decode = desired_decode - current_decode;

  delta_prefill = std::max(delta_prefill, -1);
  delta_decode = std::max(delta_decode, -1);

  if (current_prefill > 0) {
    delta_prefill =
        std::max(delta_prefill,
                 -current_prefill + std::max(1, config_.min_prefill_instances));
  }
  if (current_decode > 0) {
    delta_decode =
        std::max(delta_decode,
                 -current_decode + std::max(1, config_.min_decode_instances));
  }

  update_overprovision_state(model_id,
                             current_prefill,
                             current_decode,
                             desired_prefill,
                             desired_decode,
                             &delta_prefill,
                             &delta_decode);

  return {delta_prefill, delta_decode};
}

void BlitzScaleInstanceMgr::update_overprovision_state(
    const std::string& model_id,
    int32_t current_prefill,
    int32_t current_decode,
    int32_t desired_prefill,
    int32_t desired_decode,
    int32_t* delta_prefill,
    int32_t* delta_decode) {
  const double now_s = now_seconds();
  const double threshold_s = config_.scale_down_threshold_ms / 1000.0;

  std::lock_guard<std::mutex> lock(overprovision_mutex_);
  auto& state = overprovision_state_[model_id];

  if (desired_prefill < current_prefill) {
    if (!state.prefill_since_s.has_value()) {
      state.prefill_since_s = now_s;
      *delta_prefill = 0;
    } else if ((now_s - *state.prefill_since_s) < threshold_s) {
      *delta_prefill = 0;
    }
  } else {
    state.prefill_since_s.reset();
  }

  if (desired_decode < current_decode) {
    if (!state.decode_since_s.has_value()) {
      state.decode_since_s = now_s;
      *delta_decode = 0;
    } else if ((now_s - *state.decode_since_s) < threshold_s) {
      *delta_decode = 0;
    }
  } else {
    state.decode_since_s.reset();
  }
}

void BlitzScaleInstanceMgr::assign_role_locked(const std::string& model_id,
                                               const std::string& instance_name,
                                               Role role) {
  auto& prefill_set = prefill_instances_[model_id];
  auto& decode_set = decode_instances_[model_id];
  if (role == Role::PREFILL) {
    decode_set.erase(instance_name);
    prefill_set.insert(instance_name);
  } else {
    prefill_set.erase(instance_name);
    decode_set.insert(instance_name);
  }
}

void BlitzScaleInstanceMgr::remove_role_locked(const std::string& model_id,
                                               const std::string& instance_name,
                                               Role role) {
  auto& table =
      (role == Role::PREFILL) ? prefill_instances_[model_id]
                              : decode_instances_[model_id];
  table.erase(instance_name);
}

bool BlitzScaleInstanceMgr::engaged_model(const std::string& model_id,
                                          int32_t waiting_count,
                                          int32_t prefill_count,
                                          int32_t decode_count) const {
  return waiting_count > 0 || prefill_count > 0 || decode_count > 0 ||
         const_cast<BlitzScaleInstanceMgr*>(this)->get_active_count(model_id) > 0;
}

double BlitzScaleInstanceMgr::now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace xllm_service
