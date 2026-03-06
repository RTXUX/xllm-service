#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

#include "scheduler/prism/prism_request_tracker.h"

namespace xllm_service {

// Tracks SLO violations in a sliding time window (default 3 seconds).
// Corresponds to Prism's RequestViolationTracker.
class PrismViolationTracker {
 public:
  static constexpr double kWindowSeconds = 3.0;

  PrismViolationTracker() = default;

  // Update with current request state snapshot.
  // Records whether each request's remaining_time_budget <= 0 (violated).
  void update(const std::unordered_map<std::string, PrismModelQueue>& queues);

  struct ViolationStats {
    int32_t violated_count = 0;
    double violation_proportion = 0.0;  // violated / total
    int32_t total_reqs = 0;
  };

  ViolationStats get_model_stats(const std::string& model) const;

 private:
  struct Record {
    double timestamp;
    std::string rid;
    bool violated;
    std::string model;
  };

  void prune_old_records(double now);

  mutable std::mutex mutex_;
  // model -> deque of records within the window
  std::unordered_map<std::string, std::deque<Record>> history_;
};

// Tracks per-instance memory-per-request in a sliding window (default 30s).
// Corresponds to Prism's RequestMemoryTracker.
class PrismMemoryTracker {
 public:
  static constexpr double kWindowSeconds = 30.0;

  PrismMemoryTracker() = default;

  // Update memory stats for each instance.
  // memory_per_request = (gpu_mem_gb - sum_model_weights_gb) / max(1, total_reqs)
  void update(
      const std::unordered_map<std::string, std::vector<std::string>>&
          instance_to_models,
      const std::unordered_map<std::string, int32_t>& instance_total_reqs,
      const std::unordered_map<std::string, double>& model_weights_gb,
      double gpu_mem_gb);

  struct MemoryStats {
    double avg_memory_per_request = 0.0;
    double avg_total_reqs = 0.0;
  };

  MemoryStats get_instance_stats(const std::string& instance_name) const;

 private:
  struct Record {
    double timestamp;
    double memory_per_request;
    int32_t total_reqs;
  };

  void prune_old_records(double now);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::deque<Record>> history_;
};

// Tracks per-model request count in a sliding window (default 30s).
// Used for smoothing request counts in migration decisions.
// Corresponds to Prism's ModelRequestTracker.
class PrismModelRequestTracker {
 public:
  static constexpr double kWindowSeconds = 30.0;

  PrismModelRequestTracker() = default;

  void update(const std::unordered_map<std::string, PrismModelQueue>& queues);

  // Returns smoothed average request count for a model.
  double get_avg_request_count(const std::string& model) const;

 private:
  struct Record {
    double timestamp;
    int32_t count;
  };

  void prune_old_records(double now);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::deque<Record>> history_;
};

}  // namespace xllm_service
