#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "scheduler/llumnix/llumnix_load.h"
#include "scheduler/managers/instance_mgr.h"

namespace xllm_service {

class LlumnixInstanceMgr : public InstanceMgr {
 public:
  LlumnixInstanceMgr(const Options& options,
                      const std::shared_ptr<EtcdClient>& etcd_client,
                      bool is_master_service,
                      const LlumnixConfig& config);
  ~LlumnixInstanceMgr();

  // ===== Request tracking (called by Scheduler) =====

  struct ReqInfo {
    std::string rid;
    std::string model;
    std::string instance_name;
    enum State { WAITING, RUNNING };
    State state = WAITING;
  };

  void enqueue_request(const std::string& model,
                       const std::string& request_id);
  void start_running(const std::string& request_id,
                     const std::string& instance_name);
  void finish_request(const std::string& request_id);

  // ===== Request dispatch =====
  // Select instance for the model using the configured dispatch policy.
  // Sets request->routing.prefill_name and decode_name.
  // Returns false if no instance is available.
  bool dispatch_request(std::shared_ptr<Request> request);

  // ===== Global scheduler lifecycle =====
  void start_global_scheduler();
  void stop_global_scheduler();

 private:
  // ===== Background scheduling thread =====
  void scheduling_loop();

  // ===== Action types =====
  struct LlumnixAction {
    enum Type { ACTIVATE, DEACTIVATE };
    Type type;
    std::string model_id;
    std::string instance_name;
  };

  // ===== Core scheduling algorithm =====
  std::vector<LlumnixAction> gen_actions();

  // Phase 1: Activate models with waiting requests but no WAKEUP instance
  std::vector<LlumnixAction> activate_needed_models();

  // Phase 2: Migration rebalance (model-level sleep/wake)
  std::vector<LlumnixAction> migration_rebalance();

  // Phase 3: Evict idle models
  std::vector<LlumnixAction> evict_idle_models();

  // Action execution
  void execute_actions(const std::vector<LlumnixAction>& actions);
  void execute_activate(const std::string& model_id,
                        const std::string& instance_name);
  void execute_deactivate(const std::string& model_id,
                          const std::string& instance_name);

  // ===== Dispatch policies =====
  bool dispatch_load(std::shared_ptr<Request> request);
  bool dispatch_balanced(std::shared_ptr<Request> request);
  bool dispatch_queue(std::shared_ptr<Request> request);
  bool dispatch_round_robin(std::shared_ptr<Request> request);

  // ===== Load computation =====
  // Compute load for an instance using the specified metric.
  // For KV_BLOCKS_RATIO: higher = more loaded (busy when >= threshold).
  // For REMAINING_STEPS: higher = more loaded (internally inverted from source
  //   where higher remaining_steps = less loaded).
  double compute_instance_load(const std::string& instance_name,
                               LlumnixLoadMetric metric);

  // Compute load after simulating a migration (add/remove one request).
  // Used by balanced migration to evaluate post-migration load.
  double compute_load_after_migrate(const std::string& instance_name,
                                     LlumnixLoadMetric metric,
                                     bool is_migrate_in);

  // Check if an instance is "busy" per the source's is_busy() logic.
  bool is_instance_busy(const std::string& instance_name,
                        LlumnixLoadMetric metric,
                        double threshold);

  // ===== Migration helpers =====
  struct MigrationPair {
    std::string src_instance;
    std::string dst_instance;
    std::string model_id;
  };

  // Filter instances into migration source/destination candidates
  void filter_migration_candidates(
      const std::string& model_id,
      std::vector<std::pair<std::string, double>>& sources,
      std::vector<std::pair<std::string, double>>& destinations);

  // Balanced migration pairing
  std::optional<MigrationPair> pair_balanced_migration(
      const std::string& model_id,
      std::vector<std::pair<std::string, double>>& sources,
      std::vector<std::pair<std::string, double>>& destinations);

  // Defrag migration pairing (aggressive)
  std::optional<MigrationPair> pair_defrag_migration(
      const std::string& model_id,
      std::vector<std::pair<std::string, double>>& sources,
      std::vector<std::pair<std::string, double>>& destinations);

  // ===== Helpers =====
  std::vector<std::string> get_all_instance_names();
  double get_model_weight_gb(const std::string& model_id);

  // Request count helpers
  int32_t get_waiting_count(const std::string& model_id);
  int32_t get_running_count(const std::string& model_id);
  int32_t get_total_active_count(const std::string& model_id);
  int32_t get_reqs_on_instance(const std::string& instance_name);

  // Pick model from an instance that has the fewest requests (for migration)
  std::string pick_model_for_migration(const std::string& instance_name);

  static double now_seconds();

  // ===== Internal state =====
  LlumnixConfig config_;
  LlumnixDispatchPolicy dispatch_policy_;
  LlumnixMigrationPolicy migration_policy_;
  LlumnixLoadMetric dispatch_load_metric_;
  LlumnixLoadMetric migration_load_metric_;

  // Request tracking
  std::unordered_map<std::string, std::shared_ptr<ReqInfo>> requests_;
  std::unordered_map<std::string, std::vector<std::string>> model_requests_;
  std::mutex req_mutex_;

  // Model->instance placement mapping
  std::unordered_map<std::string, std::unordered_set<std::string>>
      instance_to_models_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      model_to_instances_;
  std::mutex placement_mutex_;

  // Per-model round-robin counter
  std::unordered_map<std::string, size_t> rr_index_;

  // Model last-used time (for idle eviction)
  std::unordered_map<std::string,
                     std::unordered_map<std::string, double>>
      model_last_used_;  // model_id -> instance_name -> timestamp
  std::mutex last_used_mutex_;

  // Random engine for top-K dispatch
  std::mt19937 rng_;

  // Scheduling thread
  std::unique_ptr<std::thread> sched_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace xllm_service
