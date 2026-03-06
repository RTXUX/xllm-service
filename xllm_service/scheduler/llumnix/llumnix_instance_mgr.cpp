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
  if (metric == LlumnixLoadMetric::KV_BLOCKS_RATIO) {
    auto lm = get_instance_load_metrics(instance_name);
    if (!lm.has_value()) return 0.0;
    return static_cast<double>(lm->gpu_cache_usage_perc);
  } else {
    // REMAINING_STEPS: approximate as 1.0 - (1.0 / (1 + num_requests))
    // More requests -> higher load
    int32_t reqs = get_reqs_on_instance(instance_name);
    if (reqs == 0) return 0.0;
    return 1.0 - (1.0 / (1.0 + static_cast<double>(reqs)));
  }
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

  // Compute load per instance
  std::vector<std::pair<double, std::string>> load_instances;
  for (const auto& inst : awake) {
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

  // Pick instance with fewest waiting requests (use heartbeat waiting_requests_num)
  std::string best;
  uint64_t best_waiting = std::numeric_limits<uint64_t>::max();
  for (const auto& inst : awake) {
    auto lm = get_instance_load_metrics(inst);
    uint64_t waiting = lm.has_value() ? lm->waiting_requests_num : 0;
    if (waiting < best_waiting) {
      best_waiting = waiting;
      best = inst;
    }
  }

  request->routing.prefill_name = best;
  request->routing.decode_name = best;
  return true;
}

bool LlumnixInstanceMgr::dispatch_round_robin(
    std::shared_ptr<Request> request) {
  auto awake = get_awake_instances(request->model);
  if (awake.empty()) return false;

  std::sort(awake.begin(), awake.end());

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

  std::lock_guard<std::mutex> lock(placement_mutex_);
  for (const auto& [model_id, instances] : model_to_instances_) {
    // Don't evict if there are active requests
    if (get_total_active_count(model_id) > 0) continue;

    // Check min_instances constraint
    if (static_cast<int32_t>(instances.size()) <=
        config_.min_instances_per_model) {
      continue;
    }

    int32_t evicted = 0;
    for (const auto& inst : instances) {
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
    double load = compute_instance_load(inst, migration_load_metric_);
    if (load > config_.migrate_out_load_threshold) {
      sources.emplace_back(inst, load);
    } else {
      destinations.emplace_back(inst, load);
    }
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
  size_t n = std::min(sources.size(), destinations.size());
  for (size_t i = 0; i < n; ++i) {
    double src_load = sources[i].second;
    double dst_load = destinations[i].second;
    double load_diff_before = src_load - dst_load;

    // Simulate migration: src load decreases, dst load increases
    // Approximate: transfer load proportionally
    double transfer_load = src_load * 0.2;  // ~20% load transfer per migration
    double src_after = src_load - transfer_load;
    double dst_after = dst_load + transfer_load;

    // Check: dst must stay below threshold
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
  std::string mid = model_id;
  std::string iname = instance_name;
  std::thread([this, mid, iname]() {
    wait_for_model_drain(iname, mid);
    send_model_sleep(iname, mid);
  }).detach();
}

}  // namespace xllm_service
