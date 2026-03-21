#include "scheduler/llumnix/llumnix_instance_mgr.h"

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

LlumnixInstanceMgr::LlumnixInstanceMgr(
    const Options& options,
    const std::shared_ptr<EtcdClient>& etcd_client,
    bool is_master_service,
    const LlumnixConfig& config)
    : InstanceMgr(options, etcd_client, is_master_service),
      config_(config),
      dispatch_policy_(parse_dispatch_policy(config.dispatch_policy)),
      migration_policy_(parse_migration_policy(config.migration_policy)),
      dispatch_load_metric_(parse_load_metric(config.dispatch_load_metric)),
      migration_load_metric_(parse_load_metric(config.migration_load_metric)),
      rng_(std::random_device{}()) {
  LOG(INFO) << "LlumnixInstanceMgr created with"
            << " schedule_interval=" << config_.schedule_interval_s << "s"
            << " idle_threshold=" << config_.model_idle_threshold_s << "s"
            << " migrate_out_load_threshold=" << config_.migrate_out_load_threshold
            << " topk_random_dispatch=" << config_.topk_random_dispatch
            << " dispatch_policy=" << config_.dispatch_policy
            << " migration_policy=" << config_.migration_policy
            << " dispatch_load_metric=" << config_.dispatch_load_metric
            << " migration_load_metric=" << config_.migration_load_metric
            << " max_models_per_instance=" << config_.max_models_per_instance;
}

LlumnixInstanceMgr::~LlumnixInstanceMgr() {
  stop_global_scheduler();
}

// ============================================================================
// Helpers
// ============================================================================

double LlumnixInstanceMgr::now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<std::string> LlumnixInstanceMgr::get_all_instance_names() {
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

double LlumnixInstanceMgr::get_model_weight_gb(const std::string& model_id) {
  uint64_t bytes = get_model_size_bytes(model_id);
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// ============================================================================
// Request Tracking
// ============================================================================

void LlumnixInstanceMgr::enqueue_request(const std::string& model,
                                          const std::string& request_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto info = std::make_shared<ReqInfo>();
  info->rid = request_id;
  info->model = model;
  info->state = ReqInfo::WAITING;
  requests_[request_id] = info;
  model_requests_[model].push_back(request_id);
}

void LlumnixInstanceMgr::start_running(const std::string& request_id,
                                        const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = requests_.find(request_id);
  if (it != requests_.end()) {
    it->second->state = ReqInfo::RUNNING;
    it->second->instance_name = instance_name;
  }

  // Update last-used time for the model on this instance
  {
    std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
    if (it != requests_.end()) {
      model_last_used_[it->second->model][instance_name] = now_seconds();
    }
  }
}

void LlumnixInstanceMgr::finish_request(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = requests_.find(request_id);
  if (it == requests_.end()) return;

  const std::string& model = it->second->model;
  const std::string& instance = it->second->instance_name;

  // Update last-used time
  if (!instance.empty()) {
    std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
    model_last_used_[model][instance] = now_seconds();
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

// ============================================================================
// Request Count Helpers
// ============================================================================

int32_t LlumnixInstanceMgr::get_waiting_count(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) return 0;
  int32_t count = 0;
  for (const auto& rid : it->second) {
    auto req_it = requests_.find(rid);
    if (req_it != requests_.end() && req_it->second->state == ReqInfo::WAITING) {
      ++count;
    }
  }
  return count;
}

int32_t LlumnixInstanceMgr::get_running_count(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) return 0;
  int32_t count = 0;
  for (const auto& rid : it->second) {
    auto req_it = requests_.find(rid);
    if (req_it != requests_.end() && req_it->second->state == ReqInfo::RUNNING) {
      ++count;
    }
  }
  return count;
}

int32_t LlumnixInstanceMgr::get_total_active_count(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  auto it = model_requests_.find(model_id);
  if (it == model_requests_.end()) return 0;
  return static_cast<int32_t>(it->second.size());
}

int32_t LlumnixInstanceMgr::get_reqs_on_instance(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(req_mutex_);
  int32_t count = 0;
  for (const auto& [_, req] : requests_) {
    if (req->instance_name == instance_name &&
        req->state == ReqInfo::RUNNING) {
      ++count;
    }
  }
  return count;
}

// ============================================================================
// Load Computation
// ============================================================================

double LlumnixInstanceMgr::compute_instance_load(
    const std::string& instance_name, LlumnixLoadMetric metric) {
  auto lm = get_instance_load_metrics(instance_name);

  if (metric == LlumnixLoadMetric::KV_BLOCKS_RATIO) {
    // Source: KvBlocksRatioLoad.compute_instance_load (load_computation.py:81-86)
    //   all_wanted_blocks = num_used_gpu_blocks + num_blocks_all_waiting_requests
    //   demand_factor = all_wanted_blocks / num_total_gpu_blocks
    if (!lm.has_value()) return 0.0;
    // Source: KvBlocksRatioLoad returns np.inf when num_total_gpu_blocks == 0
    if (lm->num_total_gpu_blocks == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double all_wanted = static_cast<double>(lm->num_used_gpu_blocks) +
                        static_cast<double>(lm->num_blocks_all_waiting_requests);
    return all_wanted / static_cast<double>(lm->num_total_gpu_blocks);
  } else {
    // Source: RemainingStepsLoad.compute_instance_load (load_computation.py:107-120)
    //   num_available = num_total - num_used
    //   num_requests = num_running + num_waiting (non-defrag mode)
    //   num_available -= num_blocks_all_waiting_requests
    //   remaining_steps = num_available / num_requests  (higher = less loaded)
    //
    // Note: Source's RemainingStepsLoad uses __lt__ as >=, meaning higher remaining_steps
    // = less loaded. We normalize to [0,1] where higher = more loaded for consistent
    // sorting (ascending = least loaded first).
    if (!lm.has_value()) return 0.0;
    if (lm->num_total_gpu_blocks == 0) return 0.0;

    int64_t num_available = static_cast<int64_t>(lm->num_total_gpu_blocks) -
                            static_cast<int64_t>(lm->num_used_gpu_blocks);
    // Non-defrag mode: subtract all waiting request blocks
    num_available -= static_cast<int64_t>(lm->num_blocks_all_waiting_requests);
    // Source allows negative num_available to reflect over-allocation (load > 1.0)

    uint64_t num_requests = lm->num_running_requests + lm->waiting_requests_num;
    // R3-2: Source returns RemainingStepsLoad(np.inf) when num_requests == 0.
    // We return 0.0 (least loaded). Both produce the same sort order (first in
    // ascending) and same filtering behavior (not busy, valid destination).
    if (num_requests == 0) return 0.0;

    double remaining_steps = static_cast<double>(num_available) /
                             static_cast<double>(num_requests);
    // Normalize: load = 1.0 - (remaining_steps / num_total_gpu_blocks)
    // Negative remaining_steps → load > 1.0, correctly reflecting over-allocation
    double load = 1.0 - remaining_steps /
                            static_cast<double>(lm->num_total_gpu_blocks);
    return load;
  }
}

double LlumnixInstanceMgr::compute_load_after_migrate(
    const std::string& instance_name, LlumnixLoadMetric metric,
    bool is_migrate_in) {
  // Source: InstanceLoadCalculator._compute_load_after_migrate (instance_info.py:147-158)
  // Deep-copies instance info, adjusts num_running_requests and num_available_gpu_blocks
  // by num_blocks_last_running_request, then recomputes load.
  // R3-1: Source modifies num_available_gpu_blocks, NOT num_used_gpu_blocks.
  // KvBlocksRatioLoad uses num_used_gpu_blocks (unchanged), so simulated load == current
  // load for KV_BLOCKS_RATIO. RemainingStepsLoad uses num_available (= total - used),
  // so we apply the delta there.
  auto lm = get_instance_load_metrics(instance_name);
  if (!lm.has_value()) return 0.0;

  LoadMetrics simulated = lm.value();
  int64_t blocks_last_running =
      static_cast<int64_t>(simulated.num_blocks_last_running_request);

  // Track available blocks delta separately (matches Python's modification target)
  int64_t available_delta = 0;
  if (is_migrate_in) {
    simulated.num_running_requests += 1;
    available_delta = -blocks_last_running;
  } else {
    if (simulated.num_running_requests > 0) simulated.num_running_requests -= 1;
    available_delta = blocks_last_running;
  }

  if (metric == LlumnixLoadMetric::KV_BLOCKS_RATIO) {
    // Source: KvBlocksRatioLoad.compute_instance_load uses num_used_gpu_blocks
    // which is NOT modified by _compute_load_after_migrate. So simulated load
    // equals current load (the migration simulation has no effect on this metric).
    if (simulated.num_total_gpu_blocks == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double all_wanted = static_cast<double>(simulated.num_used_gpu_blocks) +
                        static_cast<double>(simulated.num_blocks_all_waiting_requests);
    return all_wanted / static_cast<double>(simulated.num_total_gpu_blocks);
  } else {
    if (simulated.num_total_gpu_blocks == 0) return 0.0;
    // Apply available_delta: equivalent to Python modifying num_available_gpu_blocks
    int64_t num_available = static_cast<int64_t>(simulated.num_total_gpu_blocks) -
                            static_cast<int64_t>(simulated.num_used_gpu_blocks) +
                            available_delta;
    num_available -= static_cast<int64_t>(simulated.num_blocks_all_waiting_requests);

    uint64_t num_requests = simulated.num_running_requests + simulated.waiting_requests_num;
    if (num_requests == 0) return 0.0;

    double remaining_steps = static_cast<double>(num_available) /
                             static_cast<double>(num_requests);
    return 1.0 - remaining_steps /
                    static_cast<double>(simulated.num_total_gpu_blocks);
  }
}

bool LlumnixInstanceMgr::is_instance_busy(
    const std::string& instance_name, LlumnixLoadMetric metric,
    double threshold) {
  if (metric == LlumnixLoadMetric::REMAINING_STEPS) {
    // R3-4: Source: RemainingStepsLoad.is_busy() → remaining_steps < BUSY_THRESHOLD (10.0)
    // Use raw remaining_steps to match Python semantics (threshold in raw scale).
    // The normalized load approach can't reproduce the raw threshold behavior.
    auto lm = get_instance_load_metrics(instance_name);
    if (!lm.has_value()) return false;
    if (lm->num_total_gpu_blocks == 0) return false;

    int64_t num_available = static_cast<int64_t>(lm->num_total_gpu_blocks) -
                            static_cast<int64_t>(lm->num_used_gpu_blocks);
    num_available -= static_cast<int64_t>(lm->num_blocks_all_waiting_requests);

    uint64_t num_requests = lm->num_running_requests + lm->waiting_requests_num;
    if (num_requests == 0) return false;  // np.inf remaining_steps → not busy

    double remaining_steps = static_cast<double>(num_available) /
                             static_cast<double>(num_requests);
    return remaining_steps < config_.dispatch_busy_threshold_remaining_steps;
  }
  // KV_BLOCKS_RATIO: use normalized load >= threshold
  double load = compute_instance_load(instance_name, metric);
  return load >= threshold;
}

// ============================================================================
// Dispatch Policies
// ============================================================================

bool LlumnixInstanceMgr::dispatch_request(std::shared_ptr<Request> request) {
  switch (dispatch_policy_) {
    case LlumnixDispatchPolicy::LOAD:
      return dispatch_load(request);
    case LlumnixDispatchPolicy::BALANCED:
      return dispatch_balanced(request);
    case LlumnixDispatchPolicy::QUEUE:
      return dispatch_queue(request);
    case LlumnixDispatchPolicy::ROUND_ROBIN:
      return dispatch_round_robin(request);
    default:
      return dispatch_load(request);
  }
}

bool LlumnixInstanceMgr::dispatch_load(std::shared_ptr<Request> request) {
  auto awake = get_awake_instances(request->model);
  if (awake.empty()) return false;

  // Source: Load.filter() uses MetricBasedFilter to exclude busy instances
  // (dispatch_policy.py:121-127, dispatch_filter.py:30-43)
  std::vector<std::string> candidates;
  for (const auto& inst : awake) {
    if (!is_instance_busy(inst, dispatch_load_metric_,
                          config_.dispatch_busy_threshold)) {
      candidates.push_back(inst);
    }
  }

  // Source: dispatch_scheduler.py:102-105 - fallback to all primary instances
  // if no candidates found and early_reject is disabled
  if (candidates.empty()) {
    candidates = awake;
  }

  // Compute load per instance
  std::vector<std::pair<double, std::string>> load_instances;
  for (const auto& inst : candidates) {
    double load = compute_instance_load(inst, dispatch_load_metric_);
    load_instances.emplace_back(load, inst);
  }

  // Sort ascending (least loaded first)
  std::sort(load_instances.begin(), load_instances.end());

  // Pick randomly from top-K
  int k = std::min(config_.topk_random_dispatch,
                   static_cast<int32_t>(load_instances.size()));
  int idx = 0;
  if (k > 1) {
    std::uniform_int_distribution<int> dist(0, k - 1);
    idx = dist(rng_);
  }

  const std::string& selected = load_instances[idx].second;
  request->routing.prefill_name = selected;
  request->routing.decode_name = selected;

  DLOG(INFO) << "Llumnix load dispatch: " << request->service_request_id
             << " -> " << selected << " (load=" << load_instances[idx].first << ")";
  return true;
}

bool LlumnixInstanceMgr::dispatch_balanced(std::shared_ptr<Request> request) {
  auto awake = get_awake_instances(request->model);
  if (awake.empty()) return false;

  // Pick instance with fewest total requests
  // R3-3: Python's Balanced.select() uses instance_num_requests (a cumulative dispatch
  // counter that doesn't decrease). C++ uses get_reqs_on_instance() which counts
  // currently RUNNING requests. The C++ approach better reflects real-time load.
  std::string best;
  int32_t best_count = std::numeric_limits<int32_t>::max();
  for (const auto& inst : awake) {
    int32_t count = get_reqs_on_instance(inst);
    if (count < best_count) {
      best_count = count;
      best = inst;
    }
  }

  request->routing.prefill_name = best;
  request->routing.decode_name = best;
  return true;
}

bool LlumnixInstanceMgr::dispatch_queue(std::shared_ptr<Request> request) {
  auto awake = get_awake_instances(request->model);
  if (awake.empty()) return false;

  // Source: Queue.dispatch (dispatch_policy.py:154-164)
  // Sort by waiting_requests_num ascending (fewest first), then top-K random pick
  std::vector<std::pair<uint64_t, std::string>> waiting_instances;
  for (const auto& inst : awake) {
    auto lm = get_instance_load_metrics(inst);
    uint64_t waiting = lm.has_value() ? lm->waiting_requests_num : 0;
    waiting_instances.emplace_back(waiting, inst);
  }
  std::sort(waiting_instances.begin(), waiting_instances.end());

  // Pick randomly from top-K (same pattern as dispatch_load)
  int k = std::min(config_.topk_random_dispatch,
                   static_cast<int32_t>(waiting_instances.size()));
  int idx = 0;
  if (k > 1) {
    std::uniform_int_distribution<int> dist(0, k - 1);
    idx = dist(rng_);
  }

  const std::string& selected = waiting_instances[idx].second;
  request->routing.prefill_name = selected;
  request->routing.decode_name = selected;
  return true;
}

bool LlumnixInstanceMgr::dispatch_round_robin(
    std::shared_ptr<Request> request) {
  auto awake = get_awake_instances(request->model);
  if (awake.empty()) return false;

  std::sort(awake.begin(), awake.end());

  // P3-18: Protect rr_index_ with req_mutex_ for thread safety
  std::lock_guard<std::mutex> lock(req_mutex_);
  size_t& idx = rr_index_[request->model];
  idx = idx % awake.size();
  const std::string& selected = awake[idx];
  ++idx;

  request->routing.prefill_name = selected;
  request->routing.decode_name = selected;
  return true;
}

// ============================================================================
// Global Scheduler Lifecycle
// ============================================================================

void LlumnixInstanceMgr::start_global_scheduler() {
  if (running_.exchange(true)) return;
  sched_thread_ = std::make_unique<std::thread>([this]() { scheduling_loop(); });
  LOG(INFO) << "Llumnix global scheduler started";
}

void LlumnixInstanceMgr::stop_global_scheduler() {
  if (!running_.exchange(false)) return;
  if (sched_thread_ && sched_thread_->joinable()) {
    sched_thread_->join();
  }
  LOG(INFO) << "Llumnix global scheduler stopped";
}

void LlumnixInstanceMgr::scheduling_loop() {
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(config_.schedule_interval_s * 1000)));
    if (!running_.load()) break;

    // Sync placement state from model instance managers
    {
      std::lock_guard<std::mutex> lock(placement_mutex_);
      instance_to_models_.clear();
      model_to_instances_.clear();
      for (const auto& [model_id, _] : MODELS) {
        auto awake = get_awake_instances(model_id);
        for (const auto& inst : awake) {
          instance_to_models_[inst].insert(model_id);
          model_to_instances_[model_id].insert(inst);
        }
      }
    }

    auto actions = gen_actions();
    if (!actions.empty()) {
      LOG(INFO) << "Llumnix scheduler: executing " << actions.size() << " actions";
      execute_actions(actions);
    }
  }
}

// ============================================================================
// Core Scheduling Algorithm
// ============================================================================

std::vector<LlumnixInstanceMgr::LlumnixAction>
LlumnixInstanceMgr::gen_actions() {
  std::vector<LlumnixAction> actions;

  // Phase 1: Activate needed models
  auto activate_actions = activate_needed_models();
  actions.insert(actions.end(), activate_actions.begin(),
                 activate_actions.end());

  // Phase 2: Migration rebalance
  auto migration_actions = migration_rebalance();
  actions.insert(actions.end(), migration_actions.begin(),
                 migration_actions.end());

  // Phase 3: Evict idle models
  auto evict_actions = evict_idle_models();
  actions.insert(actions.end(), evict_actions.begin(), evict_actions.end());

  // Sort: DEACTIVATE before ACTIVATE to free resources first
  std::sort(actions.begin(), actions.end(),
            [](const LlumnixAction& a, const LlumnixAction& b) {
              return a.type > b.type;  // DEACTIVATE(1) before ACTIVATE(0)
            });

  return actions;
}

std::vector<LlumnixInstanceMgr::LlumnixAction>
LlumnixInstanceMgr::activate_needed_models() {
  std::vector<LlumnixAction> actions;

  for (const auto& [model_id, _] : MODELS) {
    int32_t waiting = get_waiting_count(model_id);
    if (waiting == 0) continue;

    auto awake = get_awake_instances(model_id);
    if (!awake.empty()) continue;

    // No awake instance for this model with waiting requests - need to activate
    // Find an instance with enough space
    auto all_instances = get_all_instance_names();
    std::string best_instance;

    for (const auto& inst : all_instances) {
      if (!has_valid_xtensor_info(inst)) continue;

      // Check model count limit
      {
        std::lock_guard<std::mutex> lock(placement_mutex_);
        auto it = instance_to_models_.find(inst);
        if (it != instance_to_models_.end() &&
            static_cast<int32_t>(it->second.size()) >=
                config_.max_models_per_instance) {
          continue;
        }
      }

      if (has_enough_space_for_model(inst, model_id)) {
        best_instance = inst;
        break;
      }
    }

    if (!best_instance.empty()) {
      actions.push_back(
          {LlumnixAction::ACTIVATE, model_id, best_instance});
      LOG(INFO) << "Llumnix: need to activate " << model_id << " on "
                << best_instance << " (waiting=" << waiting << ")";
    }
  }

  return actions;
}

std::vector<LlumnixInstanceMgr::LlumnixAction>
LlumnixInstanceMgr::migration_rebalance() {
  std::vector<LlumnixAction> actions;

  for (const auto& [model_id, _] : MODELS) {
    auto awake = get_awake_instances(model_id);
    if (awake.size() < 2) continue;  // Need at least 2 instances

    // Filter into sources and destinations
    std::vector<std::pair<std::string, double>> sources, destinations;
    filter_migration_candidates(model_id, sources, destinations);

    if (sources.empty() || destinations.empty()) continue;

    std::optional<MigrationPair> pair;
    if (migration_policy_ == LlumnixMigrationPolicy::BALANCED) {
      pair = pair_balanced_migration(model_id, sources, destinations);
    } else {
      pair = pair_defrag_migration(model_id, sources, destinations);
    }

    if (pair.has_value()) {
      // Pick a model from the source to migrate
      std::string migrate_model = pick_model_for_migration(pair->src_instance);
      if (migrate_model.empty()) continue;

      actions.push_back(
          {LlumnixAction::DEACTIVATE, migrate_model, pair->src_instance});
      actions.push_back(
          {LlumnixAction::ACTIVATE, migrate_model, pair->dst_instance});

      LOG(INFO) << "Llumnix migration: " << migrate_model << " from "
                << pair->src_instance << " to " << pair->dst_instance;

      // One migration per scheduling round
      break;
    }
  }

  return actions;
}

std::vector<LlumnixInstanceMgr::LlumnixAction>
LlumnixInstanceMgr::evict_idle_models() {
  std::vector<LlumnixAction> actions;
  double now = now_seconds();

  // P3-16: Avoid potential deadlock from nested placement_mutex_ + req_mutex_.
  // Snapshot placement state under placement_mutex_, then release it before
  // calling get_total_active_count() which acquires req_mutex_.
  std::unordered_map<std::string, std::unordered_set<std::string>>
      model_instances_snapshot;
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    model_instances_snapshot = model_to_instances_;
  }

  for (const auto& [model_id, instances] : model_instances_snapshot) {
    // Don't evict if there are active requests
    if (get_total_active_count(model_id) > 0) continue;

    // Check min_instances constraint
    if (static_cast<int32_t>(instances.size()) <=
        config_.min_instances_per_model) {
      continue;
    }

    int32_t evicted = 0;
    for (const auto& inst : instances) {
      // P3-17: Check D2D protection - don't evict instances being used as D2D source
      {
        auto mgr = get_model_instance_mgr(model_id);
        if (mgr && !mgr->can_sleep(inst)) {
          DLOG(INFO) << "Llumnix: skip eviction of " << model_id << " on "
                     << inst << " (D2D locked)";
          continue;
        }
      }

      double last_used = 0.0;
      {
        std::lock_guard<std::mutex> lu_lock(last_used_mutex_);
        auto model_it = model_last_used_.find(model_id);
        if (model_it != model_last_used_.end()) {
          auto inst_it = model_it->second.find(inst);
          if (inst_it != model_it->second.end()) {
            last_used = inst_it->second;
          }
        }
      }

      if (last_used > 0.0 &&
          (now - last_used) > config_.model_idle_threshold_s) {
        actions.push_back({LlumnixAction::DEACTIVATE, model_id, inst});
        ++evicted;
        LOG(INFO) << "Llumnix: evict idle " << model_id << " from " << inst
                  << " (idle for " << (now - last_used) << "s)";

        // Respect min_instances: only evict excess
        if (static_cast<int32_t>(instances.size()) - evicted <=
            config_.min_instances_per_model) {
          break;
        }
      }
    }
  }

  return actions;
}

// ============================================================================
// Migration Helpers
// ============================================================================

void LlumnixInstanceMgr::filter_migration_candidates(
    const std::string& model_id,
    std::vector<std::pair<std::string, double>>& sources,
    std::vector<std::pair<std::string, double>>& destinations) {
  auto awake = get_awake_instances(model_id);

  for (const auto& inst : awake) {
    // P1-7: Skip instances without valid xtensor info (new instance protection).
    // Source: migration_scheduler.py:74-80 uses DummyLoad for new instances,
    // and migration filter blocks DummyLoad instances from participating.
    if (!has_valid_xtensor_info(inst)) continue;

    double load = compute_instance_load(inst, migration_load_metric_);
    auto lm = get_instance_load_metrics(inst);

    // Source: LoadFilter (migration_filter.py:82-98)
    // src condition: num_killed_requests > 0 OR load > threshold
    // dst condition: num_killed_requests == 0 AND load < threshold
    bool has_killed = lm.has_value() && lm->num_killed_requests > 0;

    if (has_killed || load > config_.migrate_out_load_threshold) {
      sources.emplace_back(inst, load);
    } else if (!has_killed && load < config_.migrate_out_load_threshold) {
      destinations.emplace_back(inst, load);
    }
    // Note: instances with load == threshold and no killed requests are
    // neither source nor destination (consistent with source's strict < / >)
  }

  // Sort: sources descending (most loaded first), destinations ascending
  std::sort(sources.begin(), sources.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::sort(destinations.begin(), destinations.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
}

std::optional<LlumnixInstanceMgr::MigrationPair>
LlumnixInstanceMgr::pair_balanced_migration(
    const std::string& model_id,
    std::vector<std::pair<std::string, double>>& sources,
    std::vector<std::pair<std::string, double>>& destinations) {
  // Source: Balanced.pair_migration (migration_policy.py:54-73)
  // Uses migration_load_metric_after_migrate_out/in which simulate
  // ±1 request and ±num_blocks_last_running_request blocks.
  // R3-6: Python's Balanced.pair_migration uses BaseLoad.__sub__ and __gt__ for
  // load_diff computation and threshold comparison. However, BaseLoad doesn't define
  // __sub__ or __gt__, so this code path would TypeError at runtime in Python.
  // Since the default migration_policy is defrag, this is likely untested in the source.
  // Our C++ implementation uses double arithmetic which works correctly.
  size_t n = std::min(sources.size(), destinations.size());
  for (size_t i = 0; i < n; ++i) {
    double src_load = sources[i].second;
    double dst_load = destinations[i].second;
    double load_diff_before = src_load - dst_load;

    // Simulate migration using actual load recomputation
    double src_after = compute_load_after_migrate(
        sources[i].first, migration_load_metric_, /*is_migrate_in=*/false);
    double dst_after = compute_load_after_migrate(
        destinations[i].first, migration_load_metric_, /*is_migrate_in=*/true);

    // Check: dst must stay below threshold after migration
    if (dst_after > config_.migrate_out_load_threshold) continue;

    double load_diff_after = src_after - dst_after;

    // Verify improvement: 0 < load_diff_after < load_diff_before
    if (load_diff_after > 0 && load_diff_after < load_diff_before) {
      return MigrationPair{sources[i].first, destinations[i].first, model_id};
    }
  }
  return std::nullopt;
}

std::optional<LlumnixInstanceMgr::MigrationPair>
LlumnixInstanceMgr::pair_defrag_migration(
    const std::string& model_id,
    std::vector<std::pair<std::string, double>>& sources,
    std::vector<std::pair<std::string, double>>& destinations) {
  // Aggressive: just pair the most loaded source with least loaded destination
  // Note: Source (migration_policy.py:76-86) returns min(len(src), len(dst)) pairs.
  // We only return one because migration_rebalance() breaks after the first pair.
  // If multi-migration-per-round is needed, change return type to vector.
  if (!sources.empty() && !destinations.empty()) {
    return MigrationPair{sources[0].first, destinations[0].first, model_id};
  }
  return std::nullopt;
}

std::string LlumnixInstanceMgr::pick_model_for_migration(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(placement_mutex_);
  auto it = instance_to_models_.find(instance_name);
  if (it == instance_to_models_.end() || it->second.empty()) return "";

  // Pick model with fewest requests
  std::string best_model;
  int32_t min_reqs = std::numeric_limits<int32_t>::max();
  for (const auto& model : it->second) {
    int32_t reqs = get_total_active_count(model);
    if (reqs < min_reqs) {
      min_reqs = reqs;
      best_model = model;
    }
  }
  return best_model;
}

// ============================================================================
// Action Execution
// ============================================================================

void LlumnixInstanceMgr::execute_actions(
    const std::vector<LlumnixAction>& actions) {
  for (const auto& action : actions) {
    if (action.type == LlumnixAction::ACTIVATE) {
      execute_activate(action.model_id, action.instance_name);
    } else {
      execute_deactivate(action.model_id, action.instance_name);
    }
  }
}

void LlumnixInstanceMgr::execute_activate(const std::string& model_id,
                                           const std::string& instance_name) {
  LOG(INFO) << "Llumnix: activating " << model_id << " on " << instance_name;

  // Deduct free pages before wakeup
  uint64_t model_bytes = get_model_size_bytes(model_id);
  deduct_free_pages(instance_name, model_bytes);

  send_model_wakeup(instance_name, model_id,
                    /*memory_increased_in_advance=*/false);

  // Update placement
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_[instance_name].insert(model_id);
    model_to_instances_[model_id].insert(instance_name);
  }

  // Initialize last-used time
  {
    std::lock_guard<std::mutex> lock(last_used_mutex_);
    model_last_used_[model_id][instance_name] = now_seconds();
  }
}

void LlumnixInstanceMgr::execute_deactivate(const std::string& model_id,
                                             const std::string& instance_name) {
  LOG(INFO) << "Llumnix: deactivating " << model_id << " on " << instance_name;

  // Update placement first
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_[instance_name].erase(model_id);
    if (instance_to_models_[instance_name].empty()) {
      instance_to_models_.erase(instance_name);
    }
    model_to_instances_[model_id].erase(instance_name);
    if (model_to_instances_[model_id].empty()) {
      model_to_instances_.erase(model_id);
    }
  }

  // Clean up last-used
  {
    std::lock_guard<std::mutex> lock(last_used_mutex_);
    auto it = model_last_used_.find(model_id);
    if (it != model_last_used_.end()) {
      it->second.erase(instance_name);
      if (it->second.empty()) {
        model_last_used_.erase(it);
      }
    }
  }

  // Drain and sleep in a detached thread (same pattern as Prism/SLLM)
  // Check can_sleep() before sleeping to respect D2D protection
  std::string mid = model_id;
  std::string iname = instance_name;
  std::thread([this, mid, iname]() {
    wait_for_model_drain(iname, mid);
    auto mgr = get_model_instance_mgr(mid);
    if (mgr && !mgr->can_sleep(iname)) {
      LOG(WARNING) << "Llumnix: skipping sleep of " << mid << " on " << iname
                   << " (D2D locked)";
      return;
    }
    send_model_sleep(iname, mid);
  }).detach();
}

}  // namespace xllm_service
