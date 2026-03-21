#pragma once

#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace xllm_service {

enum class PrismReqState { WAITING, RUNNING, FINISHED };

struct PrismReq {
  std::string rid;               // service_request_id
  std::string model;
  double arrival_time = 0.0;     // seconds since epoch
  double slo = 30.0;             // TTFT SLO in seconds
  int32_t prompt_len = 0;
  int32_t output_len = 512;      // expected output tokens (default 512)
  double profiled_prefill_time = 0.0;  // estimated prefill time in seconds
  double priority = 0.0;         // lower = more urgent (arrival + slo - prefill_time)
  double start_running_time = 0.0;
  double finish_time = 0.0;
  double remaining_time_budget = 0.0;  // = slo - (now - arrival_time)
  std::string instance_name;     // instance executing this request
  PrismReqState state = PrismReqState::WAITING;
};

struct PrismModelQueue {
  std::string model_name;
  std::deque<std::shared_ptr<PrismReq>> waiting_reqs;
  std::list<std::shared_ptr<PrismReq>> running_reqs;
  double last_active_time = 0.0;  // last time a request arrived or finished
};

class PrismRequestTracker {
 public:
  PrismRequestTracker() = default;

  // Request lifecycle
  void enqueue_req(const std::string& model, std::shared_ptr<PrismReq> req);
  void start_running(const std::string& rid, const std::string& instance_name);
  // Returns the model_id of the finished request, or empty string if not found.
  std::string finish_req(const std::string& rid);

  // Queries (thread-safe deep copy)
  std::unordered_map<std::string, PrismModelQueue> snapshot_queues();

  // Count total requests (waiting + running) on a given instance
  int32_t get_total_reqs_on_instance(const std::string& instance_name);

  // Get the last active time for a model
  double get_model_last_active_time(const std::string& model);

  // Update remaining_time_budget for all requests
  void update_time_budgets();

  // Evict all waiting requests for a model (used on deactivation)
  void evict_waiting_reqs(const std::string& model);

  // Utility: current time in seconds since epoch
  static double now_seconds();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, PrismModelQueue> model_queues_;
  std::unordered_map<std::string, std::shared_ptr<PrismReq>> all_reqs_;  // rid -> req
};

}  // namespace xllm_service
