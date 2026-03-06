#include "scheduler/prism/prism_instance_mgr.h"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace xllm_service {

// ============================================================================
// Constructor / Destructor
// ============================================================================

PrismInstanceMgr::PrismInstanceMgr(const Options& options,
                                     const std::shared_ptr<EtcdClient>& etcd_client,
                                     bool is_master_service,
                                     const PrismConfig& config)
    : InstanceMgr(options, etcd_client, is_master_service), config_(config) {
  LOG(INFO) << "PrismInstanceMgr created with schedule_interval="
            << config_.schedule_interval_s
            << "s, memory_pool_budget=" << config_.memory_pool_budget_gb
            << "GB, idle_threshold=" << config_.model_idle_threshold_s
            << "s, migrate_policy=" << config_.migrate_policy
            << ", max_models_per_instance=" << config_.max_models_per_instance;
}

PrismInstanceMgr::~PrismInstanceMgr() { stop_global_scheduler(); }

// ============================================================================
// Request Tracking (delegated to PrismRequestTracker)
// ============================================================================

void PrismInstanceMgr::enqueue_prism_request(const std::string& model,
                                              std::shared_ptr<PrismReq> req) {
  req_tracker_.enqueue_req(model, req);
}

void PrismInstanceMgr::start_prism_running(const std::string& rid,
                                            const std::string& instance_name) {
  req_tracker_.start_running(rid, instance_name);
}

void PrismInstanceMgr::finish_prism_request(const std::string& rid) {
  req_tracker_.finish_req(rid);
}

// ============================================================================
// Request Dispatch
// ============================================================================

bool PrismInstanceMgr::dispatch_prism_request(
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

  for (const auto& inst : awake) {
    int32_t reqs = req_tracker_.get_total_reqs_on_instance(inst);
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
// Resize
// ============================================================================

bool PrismInstanceMgr::send_model_resize(const std::string& instance_name,
                                          const std::string& model_id,
                                          uint64_t new_kv_cache_pages) {
  nlohmann::json json_body;
  json_body["model_id"] = model_id;
  json_body["new_kv_cache_pages"] = new_kv_cache_pages;

  LOG(INFO) << "Prism: send_model_resize instance=" << instance_name
            << " model=" << model_id
            << " new_pages=" << new_kv_cache_pages;

  return send_http_request(instance_name, "/resize", json_body.dump());
}

// ============================================================================
// Global Scheduler Lifecycle
// ============================================================================

void PrismInstanceMgr::start_global_scheduler() {
  prism_running_ = true;
  prism_sched_thread_ = std::make_unique<std::thread>(
      &PrismInstanceMgr::scheduling_loop, this);
  LOG(INFO) << "Prism global scheduler started (interval="
            << config_.schedule_interval_s << "s)";
}

void PrismInstanceMgr::stop_global_scheduler() {
  prism_running_ = false;
  if (prism_sched_thread_ && prism_sched_thread_->joinable()) {
    prism_sched_thread_->join();
  }
}

// ============================================================================
// Background Scheduling Loop
// ============================================================================

void PrismInstanceMgr::scheduling_loop() {
  while (prism_running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(config_.schedule_interval_s * 1000)));

    if (!prism_running_) break;

    try {
      // 1. Update remaining time budgets
      req_tracker_.update_time_budgets();

      // 2. Snapshot request queues (thread-safe deep copy)
      auto queues = req_tracker_.snapshot_queues();

      // 3. Update trackers
      violation_tracker_.update(queues);
      model_req_tracker_.update(queues);

      // 4. Generate and execute actions
      auto actions = gen_actions(queues);
      if (!actions.empty()) {
        LOG(INFO) << "Prism scheduler: executing " << actions.size()
                  << " actions";
        execute_actions(actions);
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Prism scheduling_loop exception: " << e.what();
    }
  }
}

// ============================================================================
// Core Scheduling Algorithm — gen_actions
// ============================================================================

std::vector<PrismInstanceMgr::PrismAction> PrismInstanceMgr::gen_actions(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::vector<PrismAction> all_actions;

  // Phase 1: Evict idle instances
  auto evict_actions = evict_idle_instances(queues);
  all_actions.insert(all_actions.end(), evict_actions.begin(),
                     evict_actions.end());

  // Phase 2: Migration planning
  std::vector<PrismAction> migrate_actions;
  if (config_.migrate_policy == "memory_per_request") {
    migrate_actions = plan_migration_by_memory(queues);
  } else if (config_.migrate_policy == "violation") {
    migrate_actions = plan_migration_by_violation(queues);
  }
  all_actions.insert(all_actions.end(), migrate_actions.begin(),
                     migrate_actions.end());

  // Phase 3: Activate needed models
  auto activate_actions = activate_needed_models(queues);
  all_actions.insert(all_actions.end(), activate_actions.begin(),
                     activate_actions.end());

  // Phase 4: Sort — DEACTIVATE before ACTIVATE
  std::stable_sort(all_actions.begin(), all_actions.end(),
                   [](const PrismAction& a, const PrismAction& b) {
                     return (a.type == PrismAction::DEACTIVATE) &&
                            (b.type != PrismAction::DEACTIVATE);
                   });

  return all_actions;
}

// ============================================================================
// Phase 1: Evict Idle Instances
// ============================================================================

std::vector<PrismInstanceMgr::PrismAction>
PrismInstanceMgr::evict_idle_instances(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::vector<PrismAction> actions;
  double now = PrismRequestTracker::now_seconds();

  std::lock_guard<std::mutex> lock(placement_mutex_);
  for (const auto& [instance_name, models] : instance_to_models_) {
    for (const auto& model_id : models) {
      double last_active = req_tracker_.get_model_last_active_time(model_id);
      double idle_time = (last_active > 0) ? (now - last_active) : 0.0;

      // Check if model has zero active requests
      bool has_reqs = false;
      auto qit = queues.find(model_id);
      if (qit != queues.end()) {
        has_reqs = !qit->second.waiting_reqs.empty() ||
                   !qit->second.running_reqs.empty();
      }

      if (idle_time > config_.model_idle_threshold_s && !has_reqs) {
        LOG(INFO) << "Prism: evicting idle model " << model_id
                  << " from instance " << instance_name
                  << " (idle " << idle_time << "s)";
        PrismAction action;
        action.type = PrismAction::DEACTIVATE;
        action.model_id = model_id;
        action.instance_name = instance_name;
        actions.push_back(action);
      }
    }
  }
  return actions;
}

// ============================================================================
// Phase 2a: Migration by Memory-Per-Request
// ============================================================================

std::vector<PrismInstanceMgr::PrismAction>
PrismInstanceMgr::plan_migration_by_memory(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::vector<PrismAction> actions;

  // Build instance-to-models mapping
  std::unordered_map<std::string, std::vector<std::string>> inst_models;
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    for (const auto& [inst, models] : instance_to_models_) {
      inst_models[inst] = std::vector<std::string>(models.begin(), models.end());
    }
  }

  if (inst_models.size() < 2) {
    return actions;  // Need at least 2 instances for migration
  }

  // Compute memory_per_request for each instance
  // mem_per_req = (gpu_mem - sum_model_weights) / max(1, avg_reqs)
  double gpu_mem_gb = 80.0;  // TODO: get from GpuHardwareSpec
  struct InstMemInfo {
    std::string name;
    double mem_per_req;
    std::vector<std::string> models;
  };
  std::vector<InstMemInfo> inst_infos;

  for (const auto& [inst, models] : inst_models) {
    if (models.empty()) continue;

    double total_weight = 0.0;
    double total_avg_reqs = 0.0;
    for (const auto& model : models) {
      total_weight += get_model_weight_gb(model);
      total_avg_reqs += model_req_tracker_.get_avg_request_count(model);
    }

    double mem_per_req = (gpu_mem_gb - total_weight) /
                         std::max(1.0, total_avg_reqs);
    inst_infos.push_back({inst, mem_per_req, models});
  }

  // Check stability: any pair with ratio > threshold?
  int unstable_pairs = 0;
  for (size_t i = 0; i < inst_infos.size(); ++i) {
    for (size_t j = i + 1; j < inst_infos.size(); ++j) {
      double hi = std::max(inst_infos[i].mem_per_req, inst_infos[j].mem_per_req);
      double lo = std::min(inst_infos[i].mem_per_req, inst_infos[j].mem_per_req);
      if (lo > 0 && hi / lo > config_.memory_per_request_ratio_threshold) {
        ++unstable_pairs;
      }
    }
  }

  if (unstable_pairs == 0) {
    return actions;  // Stable placement
  }

  // Sort by mem_per_req ascending (most congested first)
  std::sort(inst_infos.begin(), inst_infos.end(),
            [](const InstMemInfo& a, const InstMemInfo& b) {
              return a.mem_per_req < b.mem_per_req;
            });

  // Try to find a migration that reduces unstable_pairs
  for (size_t src_idx = 0; src_idx < inst_infos.size(); ++src_idx) {
    auto& src = inst_infos[src_idx];
    if (src.models.size() <= 1) continue;  // Can't migrate from single-model instance

    for (size_t dst_idx = inst_infos.size(); dst_idx > 0; --dst_idx) {
      size_t di = dst_idx - 1;
      if (di == src_idx) continue;
      auto& dst = inst_infos[di];

      double ratio = (dst.mem_per_req > 0 && src.mem_per_req > 0)
                         ? dst.mem_per_req / src.mem_per_req
                         : 0.0;
      if (ratio <= config_.memory_per_request_ratio_threshold) continue;

      // Try migrating each model from src (prefer models with fewest requests)
      std::vector<std::pair<std::string, double>> src_model_reqs;
      for (const auto& model : src.models) {
        src_model_reqs.push_back(
            {model, model_req_tracker_.get_avg_request_count(model)});
      }
      std::sort(src_model_reqs.begin(), src_model_reqs.end(),
                [](const auto& a, const auto& b) {
                  return a.second < b.second;
                });

      for (const auto& [model, reqs] : src_model_reqs) {
        if (reqs <= 0) continue;  // Don't migrate models with 0 requests

        // Check if destination has space
        if (!has_enough_space_for_model(dst.name, model)) continue;

        // Check if dst already has max models
        if (static_cast<int32_t>(dst.models.size()) >=
            config_.max_models_per_instance) continue;

        // Simulate: would this reduce unstable pairs?
        // Simple heuristic: if we're moving from congested to uncongested, it helps
        LOG(INFO) << "Prism: planning migration of model " << model
                  << " from " << src.name << " (mem/req="
                  << src.mem_per_req << ") to " << dst.name
                  << " (mem/req=" << dst.mem_per_req << ")";

        PrismAction deact;
        deact.type = PrismAction::DEACTIVATE;
        deact.model_id = model;
        deact.instance_name = src.name;
        actions.push_back(deact);

        PrismAction act;
        act.type = PrismAction::ACTIVATE;
        act.model_id = model;
        act.instance_name = dst.name;
        act.memory_pool_budget_gb = config_.memory_pool_budget_gb;
        actions.push_back(act);

        return actions;  // One migration per scheduling round
      }
    }
  }

  return actions;
}

// ============================================================================
// Phase 2b: Migration by Violation Proportion
// ============================================================================

std::vector<PrismInstanceMgr::PrismAction>
PrismInstanceMgr::plan_migration_by_violation(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::vector<PrismAction> actions;

  std::unordered_map<std::string, std::vector<std::string>> inst_models;
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    for (const auto& [inst, models] : instance_to_models_) {
      inst_models[inst] = std::vector<std::string>(models.begin(), models.end());
    }
  }

  if (inst_models.size() < 2) return actions;

  // Compute per-instance violation proportion
  struct InstViolInfo {
    std::string name;
    double violation_proportion;
    int32_t total_violated;
    int32_t total_reqs;
    std::vector<std::string> models;
  };
  std::vector<InstViolInfo> inst_infos;

  for (const auto& [inst, models] : inst_models) {
    int32_t total_violated = 0, total_reqs = 0;
    for (const auto& model : models) {
      auto stats = violation_tracker_.get_model_stats(model);
      total_violated += stats.violated_count;
      total_reqs += stats.total_reqs;
    }
    double proportion = (total_reqs > 0)
                            ? static_cast<double>(total_violated) / total_reqs
                            : 0.0;
    inst_infos.push_back({inst, proportion, total_violated, total_reqs, models});
  }

  // Check stability
  bool unstable = false;
  for (size_t i = 0; i < inst_infos.size() && !unstable; ++i) {
    for (size_t j = i + 1; j < inst_infos.size(); ++j) {
      double diff = std::abs(inst_infos[i].violation_proportion -
                             inst_infos[j].violation_proportion);
      if (diff > config_.violation_proportion_threshold) {
        unstable = true;
        break;
      }
    }
  }

  if (!unstable) return actions;

  // Sort by violation proportion descending (most violated first)
  std::sort(inst_infos.begin(), inst_infos.end(),
            [](const InstViolInfo& a, const InstViolInfo& b) {
              return a.violation_proportion > b.violation_proportion;
            });

  // Try migrating from highest-violation to lowest-violation
  for (size_t src_idx = 0; src_idx < inst_infos.size(); ++src_idx) {
    auto& src = inst_infos[src_idx];
    if (src.models.size() <= 1) continue;

    for (size_t dst_idx = inst_infos.size(); dst_idx > 0; --dst_idx) {
      size_t di = dst_idx - 1;
      if (di == src_idx) continue;
      auto& dst = inst_infos[di];

      double diff = src.violation_proportion - dst.violation_proportion;
      if (diff <= config_.violation_proportion_threshold) continue;

      // Pick model with fewest requests from src
      std::string best_model;
      double min_reqs = std::numeric_limits<double>::max();
      for (const auto& model : src.models) {
        double reqs = model_req_tracker_.get_avg_request_count(model);
        if (reqs < min_reqs) {
          min_reqs = reqs;
          best_model = model;
        }
      }

      if (best_model.empty()) continue;
      if (!has_enough_space_for_model(dst.name, best_model)) continue;
      if (static_cast<int32_t>(dst.models.size()) >=
          config_.max_models_per_instance) continue;

      LOG(INFO) << "Prism: violation-based migration of model " << best_model
                << " from " << src.name << " (viol=" << src.violation_proportion
                << ") to " << dst.name
                << " (viol=" << dst.violation_proportion << ")";

      PrismAction deact;
      deact.type = PrismAction::DEACTIVATE;
      deact.model_id = best_model;
      deact.instance_name = src.name;
      actions.push_back(deact);

      PrismAction act;
      act.type = PrismAction::ACTIVATE;
      act.model_id = best_model;
      act.instance_name = dst.name;
      act.memory_pool_budget_gb = config_.memory_pool_budget_gb;
      actions.push_back(act);

      return actions;
    }
  }

  return actions;
}

// ============================================================================
// Phase 3: Activate Needed Models
// ============================================================================

std::vector<PrismInstanceMgr::PrismAction>
PrismInstanceMgr::activate_needed_models(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::vector<PrismAction> actions;

  // Find models with requests but no active instance
  std::vector<std::pair<std::string, double>> inactive_with_reqs;
  for (const auto& [model, queue] : queues) {
    bool has_reqs = !queue.waiting_reqs.empty() || !queue.running_reqs.empty();
    if (!has_reqs) continue;

    auto awake = get_awake_instances(model);
    if (!awake.empty()) continue;  // Already has active instance

    // Get violation proportion for priority
    auto stats = violation_tracker_.get_model_stats(model);
    inactive_with_reqs.push_back({model, stats.violation_proportion});
  }

  if (inactive_with_reqs.empty()) return actions;

  // Sort by violation proportion descending (most urgent first)
  std::sort(inactive_with_reqs.begin(), inactive_with_reqs.end(),
            [](const auto& a, const auto& b) {
              return a.second > b.second;
            });

  // Get all instances and their available memory for GPU clustering
  auto all_instances = get_all_instance_names();
  struct InstAvailInfo {
    std::string name;
    double available_gb;
  };
  std::vector<InstAvailInfo> inst_avail;
  for (const auto& inst : all_instances) {
    if (!has_valid_xtensor_info(inst)) continue;
    double free_bytes = static_cast<double>(get_instance_free_bytes(inst));
    double free_gb = free_bytes / (1024.0 * 1024.0 * 1024.0);
    inst_avail.push_back({inst, free_gb});
  }

  // Sort by available memory descending
  std::sort(inst_avail.begin(), inst_avail.end(),
            [](const InstAvailInfo& a, const InstAvailInfo& b) {
              return a.available_gb > b.available_gb;
            });

  // GPU clustering: group instances with similar available memory (5GB threshold)
  std::vector<std::vector<InstAvailInfo>> clusters;
  for (const auto& inst : inst_avail) {
    bool placed = false;
    for (auto& cluster : clusters) {
      if (!cluster.empty() &&
          std::abs(cluster[0].available_gb - inst.available_gb) <=
              config_.gpu_cluster_threshold_gb) {
        cluster.push_back(inst);
        placed = true;
        break;
      }
    }
    if (!placed) {
      clusters.push_back({inst});
    }
  }

  // For each model needing activation, find best placement
  for (const auto& [model, _viol_prop] : inactive_with_reqs) {
    bool placed = false;
    for (const auto& cluster : clusters) {
      for (const auto& inst : cluster) {
        // Check space and model count limit
        if (!has_enough_space_for_model(inst.name, model)) continue;
        {
          std::lock_guard<std::mutex> lock(placement_mutex_);
          if (static_cast<int32_t>(instance_to_models_[inst.name].size()) >=
              config_.max_models_per_instance) {
            continue;
          }
        }

        LOG(INFO) << "Prism: activating model " << model << " on instance "
                  << inst.name << " (available=" << inst.available_gb << "GB)";

        PrismAction action;
        action.type = PrismAction::ACTIVATE;
        action.model_id = model;
        action.instance_name = inst.name;
        action.memory_pool_budget_gb = config_.memory_pool_budget_gb;
        actions.push_back(action);
        placed = true;
        break;
      }
      if (placed) break;
    }
    if (!placed) {
      LOG(WARNING) << "Prism: no instance found for model " << model;
    }
  }

  return actions;
}

// ============================================================================
// Action Execution
// ============================================================================

void PrismInstanceMgr::execute_actions(
    const std::vector<PrismAction>& actions) {
  for (const auto& action : actions) {
    switch (action.type) {
      case PrismAction::ACTIVATE:
        execute_activate(action.model_id, action.instance_name,
                         action.memory_pool_budget_gb);
        break;
      case PrismAction::DEACTIVATE:
        execute_deactivate(action.model_id, action.instance_name);
        break;
      case PrismAction::RESIZE:
        // Resize handled separately via send_model_resize
        break;
    }
  }
}

void PrismInstanceMgr::execute_activate(const std::string& model_id,
                                         const std::string& instance_name,
                                         double memory_pool_gb) {
  LOG(INFO) << "Prism: executing ACTIVATE model=" << model_id
            << " instance=" << instance_name
            << " budget=" << memory_pool_gb << "GB";

  // Use inherited send_model_wakeup (D2D with H2D fallback)
  send_model_wakeup(instance_name, model_id, /*memory_increased_in_advance=*/false);

  // Update placement mapping
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_[instance_name].insert(model_id);
    model_to_instance_[model_id] = instance_name;
  }
}

void PrismInstanceMgr::execute_deactivate(const std::string& model_id,
                                           const std::string& instance_name) {
  LOG(INFO) << "Prism: executing DEACTIVATE model=" << model_id
            << " instance=" << instance_name;

  // Update placement mapping first
  {
    std::lock_guard<std::mutex> lock(placement_mutex_);
    instance_to_models_[instance_name].erase(model_id);
    if (model_to_instance_.count(model_id) &&
        model_to_instance_[model_id] == instance_name) {
      model_to_instance_.erase(model_id);
    }
  }

  // Use inherited send_model_sleep (async drain)
  // Launch in detached thread to avoid blocking scheduling loop
  std::thread([this, instance_name, model_id]() {
    // Wait for inflight requests to drain
    wait_for_model_drain(instance_name, model_id);
    send_model_sleep(instance_name, model_id);
  }).detach();
}

// ============================================================================
// Helpers
// ============================================================================

std::vector<std::string> PrismInstanceMgr::get_all_instance_names() {
  // Use MODELS list (same as InstanceMgr) to get model_instance_mgrs,
  // then collect all instance names from them
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

double PrismInstanceMgr::get_model_weight_gb(const std::string& model_id) {
  uint64_t bytes = get_model_size_bytes(model_id);
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

}  // namespace xllm_service
