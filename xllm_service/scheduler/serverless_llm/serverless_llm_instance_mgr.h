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
#include "scheduler/serverless_llm/serverless_llm_store_mgr.h"

namespace xllm_service {

class ServerlessLLMInstanceMgr : public InstanceMgr {
 public:
  ServerlessLLMInstanceMgr(const Options& options,
                           const std::shared_ptr<EtcdClient>& etcd_client,
                           bool is_master_service,
                           const ServerlessLLMConfig& config);
  ~ServerlessLLMInstanceMgr();

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
  // Select lowest-load WAKEUP instance for the model.
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
  struct SLLMAction {
    enum Type { ACTIVATE, DEACTIVATE };
    Type type;
    std::string model_id;
    std::string instance_name;
  };

  // ===== Core scheduling algorithm =====
  std::vector<SLLMAction> gen_actions();

  // Phase 1: Activate models with waiting requests but no WAKEUP instance
  std::vector<SLLMAction> activate_needed_models();

  // Phase 2: Evict idle models (LRU)
  std::vector<SLLMAction> evict_idle_models();

  // Phase 3: Auto-scale based on concurrency
  std::vector<SLLMAction> auto_scale_models();

  // Action execution
  void execute_actions(const std::vector<SLLMAction>& actions);
  void execute_activate(const std::string& model_id,
                        const std::string& instance_name);
  void execute_deactivate(const std::string& model_id,
                          const std::string& instance_name);

  // ===== Storage-aware allocation =====

  struct LoadingEstimate {
    double latency_s;
    int storage_tier;  // 0=GPU, 1=D2D, 2=H2D
  };

  struct AllocationPlan {
    std::string instance_name;
    double total_latency;
    int storage_tier;
    std::vector<SLLMAction> evictions;
  };

  // Find best instance to load a model (storage-aware)
  std::optional<AllocationPlan> find_best_allocation(
      const std::string& model_id);

  // Estimate loading latency for a model on an instance
  LoadingEstimate estimate_loading_latency(const std::string& model_id,
                                           const std::string& instance_name);

  // ===== 0/1 Knapsack eviction =====

  struct EvictionCandidate {
    std::string model_id;
    std::string instance_name;
    uint64_t freed_bytes;
    double cost;  // drain_time + reload_time
  };

  struct KnapsackResult {
    std::vector<EvictionCandidate> evicted;
    double total_cost;
  };

  // Solve 0/1 knapsack to find minimum-cost eviction set
  std::optional<KnapsackResult> knapsack_eviction(
      const std::string& instance_name, uint64_t needed_bytes);

  // Get eviction candidates for an instance (idle models only)
  std::vector<EvictionCandidate> get_eviction_candidates(
      const std::string& instance_name);

  // ===== Helpers =====
  std::vector<std::string> get_all_instance_names();
  double get_model_weight_gb(const std::string& model_id);
  void sync_store_state();

  // Request count helpers
  int32_t get_waiting_count(const std::string& model_id);
  int32_t get_running_count(const std::string& model_id);
  int32_t get_total_active_count(const std::string& model_id);
  int32_t get_reqs_on_instance(const std::string& instance_name);

  static double now_seconds();

  // ===== Internal state =====
  ServerlessLLMConfig config_;
  ServerlessLLMStoreMgr store_mgr_;

  // Request tracking
  std::unordered_map<std::string, std::shared_ptr<ReqInfo>> requests_;
  // model_id -> list of request_ids (waiting + running)
  std::unordered_map<std::string, std::vector<std::string>> model_requests_;
  std::mutex req_mutex_;

  // Model->instance placement mapping
  // instance_name -> set of active model_ids
  std::unordered_map<std::string, std::unordered_set<std::string>>
      instance_to_models_;
  // model_id -> set of instances where it's active
  std::unordered_map<std::string, std::unordered_set<std::string>>
      model_to_instances_;
  std::mutex placement_mutex_;

  // Scheduling thread
  std::unique_ptr<std::thread> sched_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace xllm_service
