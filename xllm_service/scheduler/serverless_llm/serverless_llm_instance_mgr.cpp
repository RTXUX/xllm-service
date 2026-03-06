#include "scheduler/serverless_llm/serverless_llm_instance_mgr.h"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <thread>

namespace xllm_service {

// ============================================================================
// Constructor / Destructor
// ============================================================================

ServerlessLLMInstanceMgr::ServerlessLLMInstanceMgr(
    const Options& options,
    const std::shared_ptr<EtcdClient>& etcd_client,
    bool is_master_service,
    const ServerlessLLMConfig& config)
    : InstanceMgr(options, etcd_client, is_master_service), config_(config) {
  LOG(INFO) << "ServerlessLLMInstanceMgr created with"
            << " schedule_interval=" << config_.schedule_interval_s << "s"
            << " idle_threshold=" << config_.model_idle_threshold_s << "s"
            << " d2d_speed=" << config_.d2d_speed_gbps << "GB/s"
            << " h2d_speed=" << config_.h2d_speed_gbps << "GB/s"
            << " target_ongoing=" << config_.target_ongoing_requests
            << " knapsack=" << config_.enable_knapsack_migration
            << " max_models_per_instance=" << config_.max_models_per_instance;
}

ServerlessLLMInstanceMgr::~ServerlessLLMInstanceMgr() {
  stop_global_scheduler();
}

// ============================================================================
// Helpers
// ============================================================================

double ServerlessLLMInstanceMgr::now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<std::string> ServerlessLLMInstanceMgr::get_all_instance_names() {
  std::vector<std::string> result;
  for (const auto& [model_id, _] : MODELS) {
    auto mgr = get_model_instance_mgr(model_id);
    if (!mgr) continue;
    auto names = mgr->get_all_instance_names();
    for (const auto& name : names) {
      if (std::find(result.begin(), result.end(), name) == result.end()) {
        result.push_back(name);
      }
    }
  }
  return result;
}

double ServerlessLLMInstanceMgr::get_model_weight_gb(
    const std::string& model_id) {
  uint64_t bytes = get_model_size_bytes(model_id);
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// ============================================================================
// Request Tracking
// ============================================================================

void ServerlessLLMInstanceMgr::enqueue_request(const std::string& model,
                                                const std::string& request_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto info = std::make_shared<ReqInfo>();
  info->rid = request_id;
  info->model = model;
  info->state = ReqInfo::WAITING;
  requests_[request_id] = info;
  model_requests_[model].push_back(request_id);
}

void ServerlessLLMInstanceMgr::start_running(
    const std::string& request_id, const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = requests_.find(request_id);
  if (it != requests_.end()) {
    it->second->state = ReqInfo::RUNNING;
    it->second->instance_name = instance_name;
  }
  // Touch model in store for LRU tracking
  store_mgr_.touch_model(requests_[request_id]->model, instance_name);
}

void ServerlessLLMInstanceMgr::finish_request(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = requests_.find(request_id);
  if (it == requests_.end()) return;

  const std::string& model = it->second->model;
  const std::string& instance = it->second->instance_name;

  // Touch model for LRU
  if (!instance.empty()) {
    store_mgr_.touch_model(model, instance);
  }

  // Remove from model_requests_
  auto& model_reqs = model_requests_[model];
  model_reqs.erase(
      std::remove(model_reqs.begin(), model_reqs.end(), request_id),
      model_reqs.end());
  if (model_reqs.empty()) {
    model_requests_.erase(model);
  }

  requests_.erase(it);
}

int32_t ServerlessLLMInstanceMgr::get_waiting_count(
    const std::string& model_id) {
  // req_mutex_ must be held by caller
  int32_t count = 0;
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) return 0;
  for (const auto& rid : it->second) {
    auto req_it = requests_.find(rid);
    if (req_it != requests_.end() &&
        req_it->second->state == ReqInfo::WAITING) {
      count++;
    }
  }
  return count;
}

int32_t ServerlessLLMInstanceMgr::get_running_count(
    const std::string& model_id) {
  // req_mutex_ must be held by caller
  int32_t count = 0;
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) return 0;
  for (const auto& rid : it->second) {
    auto req_it = requests_.find(rid);
    if (req_it != requests_.end() &&
        req_it->second->state == ReqInfo::RUNNING) {
      count++;
    }
  }
  return count;
}

int32_t ServerlessLLMInstanceMgr::get_total_active_count(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) return 0;
  return static_cast<int32_t>(it->second.size());
}

int32_t ServerlessLLMInstanceMgr::get_reqs_on_instance(
    const std::string& instance_name) {
  // req_mutex_ must be held by caller
  int32_t count = 0;
  for (const auto& [rid, info] : requests_) {
    if (info->instance_name == instance_name &&
        info->state == ReqInfo::RUNNING) {
      count++;
    }
  }
  return count;
}

// ============================================================================
// Request Dispatch
// ============================================================================

bool ServerlessLLMInstanceMgr::dispatch_request(
    std::shared_ptr<Request> request) {
  const std::string& model_id = request->model;

  // Get WAKEUP instances for this model
  auto awake = get_awake_instances(model_id);
  if (awake.empty()) {
    return false;
  }

  // Select instance with lowest load (fewest running requests)
  std::string best_instance;
  int32_t min_reqs = std::numeric_limits<int32_t>::max();

  std::lock_guard<std::mutex> lock(req_mutex_);
  for (const auto& inst : awake) {
    int32_t reqs = get_reqs_on_instance(inst);
    if (reqs < min_reqs) {
      min_reqs = reqs;
      best_instance = inst;
    }
  }

  if (best_instance.empty()) {
    return false;
  }

  request->routing.prefill_name = best_instance;
  request->routing.decode_name = best_instance;
  return true;
}

// ============================================================================
// Global Scheduler Lifecycle
// ============================================================================

void ServerlessLLMInstanceMgr::start_global_scheduler() {
  running_ = true;
  sched_thread_ = std::make_unique<std::thread>(
      &ServerlessLLMInstanceMgr::scheduling_loop, this);
  LOG(INFO) << "ServerlessLLM global scheduler started (interval="
            << config_.schedule_interval_s << "s)";
}

void ServerlessLLMInstanceMgr::stop_global_scheduler() {
  running_ = false;
  if (sched_thread_ && sched_thread_->joinable()) {
    sched_thread_->join();
  }
}

// ============================================================================
// Background Scheduling Loop
// ============================================================================

void ServerlessLLMInstanceMgr::scheduling_loop() {
  while (running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(config_.schedule_interval_s * 1000)));

    if (!running_) break;

    try {
      // 1. Sync storage state from InstanceMgr
      sync_store_state();

      // 2. Generate and execute actions
      auto actions = gen_actions();
      if (!actions.empty()) {
        LOG(INFO) << "ServerlessLLM scheduler: executing " << actions.size()
                  << " actions";
        execute_actions(actions);
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "ServerlessLLM scheduling_loop exception: " << e.what();
    }
  }
}

void ServerlessLLMInstanceMgr::sync_store_state() {
  auto all_instances = get_all_instance_names();

  // Build instance -> awake models map
  std::unordered_map<std::string, std::vector<std::string>>
      instance_to_awake;
  std::unordered_set<std::string> all_awake_models;

  for (const auto& [model_id, _] : MODELS) {
    auto awake = get_awake_instances(model_id);
    for (const auto& inst : awake) {
      instance_to_awake[inst].push_back(model_id);
      all_awake_models.insert(model_id);
    }
  }

  store_mgr_.update_awake_models(instance_to_awake);

  // A model has D2D if it's awake on at least one instance
  // (another instance could use D2D to transfer from there)
  store_mgr_.update_d2d_sources(all_awake_models);

  // Sync placement mapping
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_.clear();
    model_to_instances_.clear();
    for (const auto& [inst, models] : instance_to_awake) {
      for (const auto& model : models) {
        instance_to_models_[inst].insert(model);
        model_to_instances_[model].insert(inst);
      }
    }
  }
}

// ============================================================================
// Core Scheduling Algorithm — gen_actions
// ============================================================================

std::vector<ServerlessLLMInstanceMgr::SLLMAction>
ServerlessLLMInstanceMgr::gen_actions() {
  std::vector<SLLMAction> all_actions;

  // Phase 1: Activate models that have waiting requests but no WAKEUP instance
  auto activate_actions = activate_needed_models();
  all_actions.insert(all_actions.end(), activate_actions.begin(),
                     activate_actions.end());

  // Phase 2: Evict idle models (LRU)
  auto evict_actions = evict_idle_models();
  all_actions.insert(all_actions.end(), evict_actions.begin(),
                     evict_actions.end());

  // Phase 3: Auto-scale based on concurrency
  auto scale_actions = auto_scale_models();
  all_actions.insert(all_actions.end(), scale_actions.begin(),
                     scale_actions.end());

  // Sort: DEACTIVATE before ACTIVATE (free space first)
  std::stable_sort(all_actions.begin(), all_actions.end(),
                   [](const SLLMAction& a, const SLLMAction& b) {
                     return (a.type == SLLMAction::DEACTIVATE) &&
                            (b.type != SLLMAction::DEACTIVATE);
                   });

  return all_actions;
}

// ============================================================================
// Phase 1: Activate Needed Models (Storage-Aware)
// ============================================================================

std::vector<ServerlessLLMInstanceMgr::SLLMAction>
ServerlessLLMInstanceMgr::activate_needed_models() {
  std::vector<SLLMAction> actions;

  // Find models with waiting requests but no WAKEUP instance
  std::vector<std::string> models_needing_activation;
  {
    std::lock_guard<std::mutex> lock(req_mutex_);
    for (const auto& [model_id, req_ids] : model_requests_) {
      if (req_ids.empty()) continue;

      // Check if any request is waiting
      bool has_waiting = false;
      for (const auto& rid : req_ids) {
        auto it = requests_.find(rid);
        if (it != requests_.end() &&
            it->second->state == ReqInfo::WAITING) {
          has_waiting = true;
          break;
        }
      }
      if (!has_waiting) continue;

      auto awake = get_awake_instances(model_id);
      if (awake.empty()) {
        models_needing_activation.push_back(model_id);
      }
    }
  }

  // For each model, use storage-aware allocation
  for (const auto& model_id : models_needing_activation) {
    auto plan = find_best_allocation(model_id);
    if (!plan.has_value()) {
      LOG(WARNING) << "ServerlessLLM: no allocation found for model "
                   << model_id;
      continue;
    }

    LOG(INFO) << "ServerlessLLM: activating model " << model_id
              << " on instance " << plan->instance_name
              << " (tier=" << plan->storage_tier
              << " latency=" << plan->total_latency << "s)";

    // Add eviction actions first
    for (const auto& eviction : plan->evictions) {
      actions.push_back(eviction);
    }

    // Add activation action
    SLLMAction action;
    action.type = SLLMAction::ACTIVATE;
    action.model_id = model_id;
    action.instance_name = plan->instance_name;
    actions.push_back(action);
  }

  return actions;
}

// ============================================================================
// Phase 2: Evict Idle Models (LRU)
// ============================================================================

std::vector<ServerlessLLMInstanceMgr::SLLMAction>
ServerlessLLMInstanceMgr::evict_idle_models() {
  std::vector<SLLMAction> actions;

  std::lock_guard<std::mutex> lock(placement_mutex_);
  for (const auto& [instance_name, models] : instance_to_models_) {
    for (const auto& model_id : models) {
      double idle = store_mgr_.get_idle_duration(model_id, instance_name);

      // Check if model has zero active requests
      int32_t active = get_total_active_count(model_id);

      // Check minimum instance constraint
      auto inst_it = model_to_instances_.find(model_id);
      int32_t current_instances =
          inst_it != model_to_instances_.end()
              ? static_cast<int32_t>(inst_it->second.size())
              : 0;

      if (idle > config_.model_idle_threshold_s && active == 0 &&
          current_instances > config_.min_instances_per_model) {
        LOG(INFO) << "ServerlessLLM: LRU evicting idle model " << model_id
                  << " from instance " << instance_name
                  << " (idle " << idle << "s)";
        SLLMAction action;
        action.type = SLLMAction::DEACTIVATE;
        action.model_id = model_id;
        action.instance_name = instance_name;
        actions.push_back(action);
      }
    }
  }

  return actions;
}

// ============================================================================
// Phase 3: Auto-Scale Based on Concurrency
// ============================================================================

std::vector<ServerlessLLMInstanceMgr::SLLMAction>
ServerlessLLMInstanceMgr::auto_scale_models() {
  std::vector<SLLMAction> actions;

  // Collect models with active requests
  std::unordered_map<std::string, int32_t> model_active_reqs;
  {
    std::lock_guard<std::mutex> lock(req_mutex_);
    for (const auto& [model_id, req_ids] : model_requests_) {
      model_active_reqs[model_id] = static_cast<int32_t>(req_ids.size());
    }
  }

  for (const auto& [model_id, active_reqs] : model_active_reqs) {
    if (active_reqs == 0) continue;

    // desired = ceil(active_reqs / target_ongoing_requests)
    int32_t desired = static_cast<int32_t>(
        std::ceil(static_cast<double>(active_reqs) /
                  config_.target_ongoing_requests));
    desired = std::max(desired, config_.min_instances_per_model);
    desired = std::min(desired, config_.max_instances_per_model);

    auto awake = get_awake_instances(model_id);
    int32_t current = static_cast<int32_t>(awake.size());

    if (desired > current) {
      // Scale up: activate on more instances
      int32_t to_add = desired - current;
      for (int32_t i = 0; i < to_add; ++i) {
        auto plan = find_best_allocation(model_id);
        if (!plan.has_value()) {
          LOG(WARNING) << "ServerlessLLM: auto-scale cannot find instance for "
                       << model_id << " (need " << to_add
                       << " more, got " << i << ")";
          break;
        }

        for (const auto& eviction : plan->evictions) {
          actions.push_back(eviction);
        }

        SLLMAction action;
        action.type = SLLMAction::ACTIVATE;
        action.model_id = model_id;
        action.instance_name = plan->instance_name;
        actions.push_back(action);
      }
    } else if (desired < current && desired >= 0) {
      // Scale down: deactivate excess instances (pick least loaded)
      int32_t to_remove = current - desired;

      // Sort awake instances by load (fewest requests first — deactivate those)
      std::vector<std::pair<std::string, int32_t>> inst_load;
      {
        std::lock_guard<std::mutex> lock(req_mutex_);
        for (const auto& inst : awake) {
          inst_load.push_back({inst, get_reqs_on_instance(inst)});
        }
      }
      std::sort(inst_load.begin(), inst_load.end(),
                [](const auto& a, const auto& b) {
                  return a.second < b.second;
                });

      for (int32_t i = 0; i < to_remove && i < static_cast<int32_t>(inst_load.size()); ++i) {
        // Only deactivate if no running requests
        if (inst_load[i].second > 0) continue;

        SLLMAction action;
        action.type = SLLMAction::DEACTIVATE;
        action.model_id = model_id;
        action.instance_name = inst_load[i].first;
        actions.push_back(action);
      }
    }
  }

  return actions;
}

// ============================================================================
// Storage-Aware Allocation
// ============================================================================

std::optional<ServerlessLLMInstanceMgr::AllocationPlan>
ServerlessLLMInstanceMgr::find_best_allocation(const std::string& model_id) {
  auto all_instances = get_all_instance_names();

  std::optional<AllocationPlan> best_plan;
  double best_latency = std::numeric_limits<double>::max();

  for (const auto& instance_name : all_instances) {
    // Skip instances without valid xtensor info
    if (!has_valid_xtensor_info(instance_name)) continue;

    // Check model count limit
    {
      std::lock_guard<std::mutex> lock(placement_mutex_);
      auto it = instance_to_models_.find(instance_name);
      if (it != instance_to_models_.end() &&
          static_cast<int32_t>(it->second.size()) >=
              config_.max_models_per_instance) {
        continue;
      }
      // Skip if model already active on this instance
      if (it != instance_to_models_.end() &&
          it->second.find(model_id) != it->second.end()) {
        continue;
      }
    }

    // Estimate loading latency
    auto estimate = estimate_loading_latency(model_id, instance_name);
    double io_wait = store_mgr_.get_io_queue_wait(instance_name);
    double total_latency = estimate.latency_s + io_wait;

    if (has_enough_space_for_model(instance_name, model_id)) {
      // Instance has enough space — direct allocation
      if (total_latency < best_latency) {
        best_latency = total_latency;
        AllocationPlan plan;
        plan.instance_name = instance_name;
        plan.total_latency = total_latency;
        plan.storage_tier = estimate.storage_tier;
        best_plan = plan;
      }
    } else if (config_.enable_knapsack_migration) {
      // Try knapsack eviction
      uint64_t model_size = get_model_size_bytes(model_id);
      uint64_t free_bytes = get_instance_free_bytes(instance_name);
      uint64_t needed = (model_size > free_bytes) ? (model_size - free_bytes) : 0;

      auto knapsack_result = knapsack_eviction(instance_name, needed);
      if (knapsack_result.has_value()) {
        double eviction_cost = knapsack_result->total_cost;
        double total_with_eviction = total_latency + eviction_cost;

        if (total_with_eviction < best_latency) {
          best_latency = total_with_eviction;
          AllocationPlan plan;
          plan.instance_name = instance_name;
          plan.total_latency = total_with_eviction;
          plan.storage_tier = estimate.storage_tier;

          // Convert eviction candidates to DEACTIVATE actions
          for (const auto& ev : knapsack_result->evicted) {
            SLLMAction action;
            action.type = SLLMAction::DEACTIVATE;
            action.model_id = ev.model_id;
            action.instance_name = ev.instance_name;
            plan.evictions.push_back(action);
          }

          best_plan = plan;
        }
      }
    }
  }

  return best_plan;
}

ServerlessLLMInstanceMgr::LoadingEstimate
ServerlessLLMInstanceMgr::estimate_loading_latency(
    const std::string& model_id, const std::string& instance_name) {
  int tier = store_mgr_.get_storage_tier(model_id, instance_name);
  double model_size_gb = get_model_weight_gb(model_id);

  LoadingEstimate estimate;
  estimate.storage_tier = tier;

  switch (tier) {
    case 0:
      // Already loaded on GPU — zero latency
      estimate.latency_s = 0.0;
      break;
    case 1:
      // D2D transfer available
      estimate.latency_s = model_size_gb / config_.d2d_speed_gbps;
      break;
    case 2:
    default:
      // H2D only
      estimate.latency_s = model_size_gb / config_.h2d_speed_gbps;
      break;
  }

  return estimate;
}

// ============================================================================
// 0/1 Knapsack Eviction (DP)
// ============================================================================

std::vector<ServerlessLLMInstanceMgr::EvictionCandidate>
ServerlessLLMInstanceMgr::get_eviction_candidates(
    const std::string& instance_name) {
  std::vector<EvictionCandidate> candidates;

  std::lock_guard<std::mutex> lock(placement_mutex_);
  auto it = instance_to_models_.find(instance_name);
  if (it == instance_to_models_.end()) return candidates;

  for (const auto& model_id : it->second) {
    // Only consider models with no active requests
    int32_t active = get_total_active_count(model_id);
    if (active > 0) continue;

    // Check minimum instance constraint
    auto inst_it = model_to_instances_.find(model_id);
    int32_t current_instances =
        inst_it != model_to_instances_.end()
            ? static_cast<int32_t>(inst_it->second.size())
            : 0;
    if (current_instances <= config_.min_instances_per_model) continue;

    EvictionCandidate candidate;
    candidate.model_id = model_id;
    candidate.instance_name = instance_name;
    candidate.freed_bytes = get_model_size_bytes(model_id);

    // Cost = drain_time + reload_time
    // drain_time = 0 (no active requests)
    // reload_time = model_size / h2d_speed (worst case: reload from H2D)
    double model_gb = get_model_weight_gb(model_id);
    double reload_time = model_gb / config_.h2d_speed_gbps;
    candidate.cost = reload_time;

    candidates.push_back(candidate);
  }

  return candidates;
}

std::optional<ServerlessLLMInstanceMgr::KnapsackResult>
ServerlessLLMInstanceMgr::knapsack_eviction(
    const std::string& instance_name, uint64_t needed_bytes) {
  auto candidates = get_eviction_candidates(instance_name);
  if (candidates.empty()) return std::nullopt;

  // Convert to page-based DP
  static constexpr uint64_t kPageSize = kXTensorPageSizeBytes;  // 2MB
  int32_t needed_pages = static_cast<int32_t>(
      (needed_bytes + kPageSize - 1) / kPageSize);
  int32_t n = static_cast<int32_t>(candidates.size());

  // Each candidate's freed pages
  std::vector<int32_t> freed_pages(n);
  for (int32_t i = 0; i < n; ++i) {
    freed_pages[i] = static_cast<int32_t>(
        (candidates[i].freed_bytes + kPageSize - 1) / kPageSize);
  }

  int32_t W = needed_pages;

  // dp[j] = minimum cost to free exactly j pages (or more, capped at W)
  // Initialize with infinity
  std::vector<double> dp(W + 1, std::numeric_limits<double>::max());
  dp[0] = 0.0;

  // Track which items are selected
  // selected[j] = bitmask (use vector<bool> per capacity for simplicity)
  std::vector<std::vector<bool>> selected(W + 1, std::vector<bool>(n, false));

  for (int32_t i = 0; i < n; ++i) {
    // Process in reverse to avoid using item twice
    for (int32_t j = W; j >= 0; --j) {
      if (dp[j] == std::numeric_limits<double>::max()) continue;

      int32_t new_j = std::min(W, j + freed_pages[i]);
      double new_cost = dp[j] + candidates[i].cost;

      if (new_cost < dp[new_j]) {
        dp[new_j] = new_cost;
        selected[new_j] = selected[j];
        selected[new_j][i] = true;
      }
    }
  }

  // Check if we can free enough pages
  if (dp[W] == std::numeric_limits<double>::max()) {
    return std::nullopt;
  }

  KnapsackResult result;
  result.total_cost = dp[W];
  for (int32_t i = 0; i < n; ++i) {
    if (selected[W][i]) {
      result.evicted.push_back(candidates[i]);
    }
  }

  LOG(INFO) << "ServerlessLLM: knapsack eviction on " << instance_name
            << " needs " << needed_pages << " pages, evicting "
            << result.evicted.size() << " models, cost=" << result.total_cost;

  return result;
}

// ============================================================================
// Action Execution
// ============================================================================

void ServerlessLLMInstanceMgr::execute_actions(
    const std::vector<SLLMAction>& actions) {
  for (const auto& action : actions) {
    switch (action.type) {
      case SLLMAction::ACTIVATE:
        execute_activate(action.model_id, action.instance_name);
        break;
      case SLLMAction::DEACTIVATE:
        execute_deactivate(action.model_id, action.instance_name);
        break;
    }
  }
}

void ServerlessLLMInstanceMgr::execute_activate(
    const std::string& model_id, const std::string& instance_name) {
  LOG(INFO) << "ServerlessLLM: executing ACTIVATE model=" << model_id
            << " instance=" << instance_name;

  store_mgr_.record_io_start(instance_name);

  // Use inherited send_model_wakeup (D2D with H2D fallback)
  send_model_wakeup(instance_name, model_id,
                    /*memory_increased_in_advance=*/false);

  store_mgr_.record_io_complete(instance_name);

  // Update placement mapping
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_[instance_name].insert(model_id);
    model_to_instances_[model_id].insert(instance_name);
  }

  // Touch model for LRU
  store_mgr_.touch_model(model_id, instance_name);
}

void ServerlessLLMInstanceMgr::execute_deactivate(
    const std::string& model_id, const std::string& instance_name) {
  LOG(INFO) << "ServerlessLLM: executing DEACTIVATE model=" << model_id
            << " instance=" << instance_name;

  // Update placement mapping first
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_[instance_name].erase(model_id);
    model_to_instances_[model_id].erase(instance_name);
    if (model_to_instances_[model_id].empty()) {
      model_to_instances_.erase(model_id);
    }
  }

  // Async drain + sleep (non-blocking)
  std::thread([this, instance_name, model_id]() {
    wait_for_model_drain(instance_name, model_id);
    send_model_sleep(instance_name, model_id);
  }).detach();
}

}  // namespace xllm_service
