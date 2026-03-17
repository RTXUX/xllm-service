#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "scheduler/managers/instance_mgr.h"

namespace xllm_service {

class BlitzScaleInstanceMgr : public InstanceMgr {
 public:
  BlitzScaleInstanceMgr(const Options& options,
                        const std::shared_ptr<EtcdClient>& etcd_client,
                        bool is_master_service,
                        const BlitzScaleConfig& config);
  ~BlitzScaleInstanceMgr();

  struct ReqInfo {
    std::string rid;
    std::string model;
    std::string prefill_instance;
    std::string decode_instance;
    int32_t prompt_tokens = 0;
    int32_t prompt_blocks = 1;
    double enqueue_time = 0.0;
    enum State { WAITING, RUNNING };
    State state = WAITING;
  };

  // Non-blocking: enqueue request into pending queue and wake dispatch thread
  void add_pending_request(std::shared_ptr<Request> request);

  // Called when prefill_instance finishes a prefill: marks it idle and wakes
  // the dispatch thread so it can pull the next pending request
  void notify_prefill_done(const std::string& prefill_instance);

  void start_running(const std::string& request_id,
                     const std::string& prefill_instance,
                     const std::string& decode_instance);
  void finish_request(const std::string& request_id);

  bool dispatch_request(std::shared_ptr<Request> request);

  void start_global_scheduler();
  void stop_global_scheduler();

 private:
  enum class Role {
    PREFILL,
    DECODE,
  };

  struct BlitzAction {
    enum Type {
      ACTIVATE_PREFILL,
      ACTIVATE_DECODE,
      DEACTIVATE_PREFILL,
      DEACTIVATE_DECODE,
      FLIP_TO_PREFILL,
      FLIP_TO_DECODE,
    };

    Type type;
    std::string model_id;
    std::string instance_name;
  };

  struct OverprovisionState {
    std::optional<double> prefill_since_s;
    std::optional<double> decode_since_s;
  };

  void enqueue_request(const std::string& model,
                       const std::string& request_id,
                       int32_t prompt_tokens,
                       double enqueue_time);

  // Enqueue prefill_instance as idle for model_id; wakes dispatch thread
  void enqueue_idle_prefill(const std::string& model_id,
                            const std::string& instance_name);

  void scheduling_loop();
  void dispatch_loop();
  void dispatch_pending_requests();
  // Pull-model dispatch: prefill instance is already known; only selects decode
  bool dispatch_with_prefill(std::shared_ptr<Request> request,
                             const std::string& prefill_instance);
  void sync_role_state();
  std::vector<BlitzAction> gen_actions();
  std::vector<BlitzAction> gen_model_actions(const std::string& model_id);
  void execute_actions(const std::vector<BlitzAction>& actions);
  void execute_activate(const std::string& model_id,
                        const std::string& instance_name,
                        Role role);
  void execute_deactivate(const std::string& model_id,
                          const std::string& instance_name,
                          Role role);
  void execute_flip(const std::string& model_id,
                    const std::string& instance_name,
                    Role role);

  std::vector<std::string> get_all_instance_names();
  int32_t get_waiting_count(const std::string& model_id);
  int32_t get_waiting_prefill_tokens(const std::string& model_id);
  int32_t get_active_count(const std::string& model_id);
  int32_t get_request_blocks(int32_t prompt_tokens) const;
  int32_t estimate_used_blocks(const std::string& instance_name);
  int32_t get_prefill_load(const std::string& instance_name);
  int32_t get_decode_load(const std::string& instance_name);
  std::vector<std::string> get_role_instances(const std::string& model_id,
                                              Role role);
  std::vector<std::string> get_dispatch_candidates(const std::string& model_id,
                                                   Role role);
  std::optional<std::string> select_prefill_instance(
      const std::string& model_id);
  std::optional<std::string> select_decode_instance(
      const std::string& model_id,
      int32_t request_blocks);
  std::optional<std::string> select_activation_instance(
      const std::string& model_id);
  std::optional<std::string> select_scale_down_instance(
      const std::string& model_id,
      Role role);
  std::optional<std::string> select_flip_instance(const std::string& model_id,
                                                  Role from_role);
  int32_t compute_waiting_decode_blocks(const std::string& model_id);
  int32_t compute_prefill_tokens(const std::string& model_id);
  std::pair<int32_t, int32_t> compute_scale_plan(
      const std::string& model_id,
      int32_t current_prefill,
      int32_t current_decode,
      int32_t waiting_prefill_tokens,
      int32_t waiting_decode_blocks,
      int32_t prefill_tokens);
  void update_overprovision_state(const std::string& model_id,
                                  int32_t current_prefill,
                                  int32_t current_decode,
                                  int32_t desired_prefill,
                                  int32_t desired_decode,
                                  int32_t* delta_prefill,
                                  int32_t* delta_decode);
  void assign_role_locked(const std::string& model_id,
                          const std::string& instance_name,
                          Role role);
  void remove_role_locked(const std::string& model_id,
                          const std::string& instance_name,
                          Role role);
  bool engaged_model(const std::string& model_id,
                     int32_t waiting_count,
                     int32_t prefill_count,
                     int32_t decode_count) const;
  static double now_seconds();

  BlitzScaleConfig config_;

  std::unordered_map<std::string, std::shared_ptr<ReqInfo>> requests_;
  std::unordered_map<std::string, std::vector<std::string>> model_requests_;
  mutable std::mutex req_mutex_;

  std::unordered_map<std::string, std::unordered_set<std::string>>
      prefill_instances_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      decode_instances_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      instance_to_models_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      model_to_instances_;
  mutable std::mutex role_mutex_;

  std::unordered_map<std::string,
                     std::unordered_map<std::string, double>>
      model_last_used_;
  std::mutex last_used_mutex_;

  std::unordered_map<std::string, OverprovisionState> overprovision_state_;
  std::mutex overprovision_mutex_;

  // Per-model pending queue (FCFS; requests are fully tokenized)
  std::unordered_map<std::string, std::deque<std::shared_ptr<Request>>>
      pending_queues_;
  // Per-model sum of prompt tokens for O(1) get_waiting_prefill_tokens()
  std::unordered_map<std::string, int32_t> pending_tokens_;
  // Per-model queue of idle prefill instances (pull model)
  std::unordered_map<std::string, std::deque<std::string>> idle_prefill_queues_;
  // Protects pending_queues_, pending_tokens_, and idle_prefill_queues_
  mutable std::mutex pending_mutex_;

  // Reactive dispatch: woken on new request or new instance capacity
  std::condition_variable dispatch_cv_;

  std::unique_ptr<std::thread> sched_thread_;
  std::unique_ptr<std::thread> dispatch_thread_;
  std::atomic<bool> running_{false};
};

}  // namespace xllm_service
