#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "scheduler/managers/instance_mgr.h"
#include "scheduler/prism/prism_request_tracker.h"
#include "scheduler/prism/prism_trackers.h"

namespace xllm_service {

class PrismInstanceMgr : public InstanceMgr {
 public:
  PrismInstanceMgr(const Options& options,
                   const std::shared_ptr<EtcdClient>& etcd_client,
                   bool is_master_service,
                   const PrismConfig& config);
  ~PrismInstanceMgr();

  // ===== Request tracking (called by Scheduler) =====
  void enqueue_prism_request(const std::string& model,
                             std::shared_ptr<PrismReq> req);
  void start_prism_running(const std::string& rid,
                           const std::string& instance_name);
  void finish_prism_request(const std::string& rid);

  // ===== Request dispatch =====
  // Select lowest-load WAKEUP instance for the model.
  // Sets request->routing.prefill_name and decode_name.
  // Returns false if no instance is available.
  bool dispatch_prism_request(std::shared_ptr<Request> request);

  // ===== Resize =====
  bool send_model_resize(const std::string& instance_name,
                         const std::string& model_id,
                         uint64_t new_kv_cache_pages);

  // ===== Global scheduler lifecycle =====
  void start_global_scheduler();
  void stop_global_scheduler();

 private:
  // ===== Background scheduling thread =====
  void scheduling_loop();

  // ===== Core scheduling algorithm (Prism gen_actions) =====
  struct PrismAction {
    enum Type { ACTIVATE, DEACTIVATE, RESIZE };
    Type type;
    std::string model_id;
    std::string instance_name;
    double memory_pool_budget_gb = 0.0;
  };

  std::vector<PrismAction> gen_actions(
      const std::unordered_map<std::string, PrismModelQueue>& queues);

  // Phase 1: Evict idle models
  std::vector<PrismAction> evict_idle_instances(
      const std::unordered_map<std::string, PrismModelQueue>& queues);

  // Phase 2: Migration decisions
  std::vector<PrismAction> plan_migration_by_memory(
      const std::unordered_map<std::string, PrismModelQueue>& queues);
  std::vector<PrismAction> plan_migration_by_violation(
      const std::unordered_map<std::string, PrismModelQueue>& queues);

  // Phase 3: Activate models that have requests but no active instance
  std::vector<PrismAction> activate_needed_models(
      const std::unordered_map<std::string, PrismModelQueue>& queues);

  // Action execution
  void execute_actions(const std::vector<PrismAction>& actions);
  void execute_activate(const std::string& model_id,
                        const std::string& instance_name,
                        double memory_pool_gb);
  void execute_deactivate(const std::string& model_id,
                          const std::string& instance_name);

  // ===== Placement helpers =====
  // Get all registered instance names
  std::vector<std::string> get_all_instance_names();

  // Get model weight in GB (from xtensor info or model_memory_specs)
  double get_model_weight_gb(const std::string& model_id);

  // ===== Internal state =====
  PrismConfig config_;

  // Request trackers
  PrismRequestTracker req_tracker_;
  PrismViolationTracker violation_tracker_;
  PrismMemoryTracker memory_tracker_;
  PrismModelRequestTracker model_req_tracker_;

  // Model->instance placement mapping (Prism's logical view)
  // instance_name -> set of active model_ids on that instance
  std::unordered_map<std::string, std::unordered_set<std::string>>
      instance_to_models_;
  // model_id -> instance_name where the model is active
  std::unordered_map<std::string, std::string> model_to_instance_;
  std::mutex placement_mutex_;

  // Scheduling thread
  std::unique_ptr<std::thread> prism_sched_thread_;
  std::atomic<bool> prism_running_{false};
};

}  // namespace xllm_service
