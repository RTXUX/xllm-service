#include "scheduler/prism/prism_trackers.h"

#include "scheduler/prism/prism_request_tracker.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xllm_service {

// ============================================================================
// PrismViolationTracker
// ============================================================================

void PrismViolationTracker::prune_old_records(double now) {
  double cutoff = now - kWindowSeconds;
  for (auto& [model, records] : history_) {
    while (!records.empty() && records.front().timestamp < cutoff) {
      records.pop_front();
    }
  }
}

void PrismViolationTracker::update(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::lock_guard<std::mutex> lock(mutex_);
  double now = PrismRequestTracker::now_seconds();
  prune_old_records(now);

  for (const auto& [model, queue] : queues) {
    int32_t violated = 0;
    int32_t total = 0;
    // Count violations across waiting + running requests
    for (const auto& req : queue.waiting_reqs) {
      ++total;
      if (req->remaining_time_budget <= 0) ++violated;
    }
    for (const auto& req : queue.running_reqs) {
      ++total;
      if (req->remaining_time_budget <= 0) ++violated;
    }
    if (total > 0) {
      history_[model].push_back({now, violated, total});
    }
  }
}

PrismViolationTracker::ViolationStats
PrismViolationTracker::get_model_stats(const std::string& model) const {
  std::lock_guard<std::mutex> lock(mutex_);
  ViolationStats stats;
  auto it = history_.find(model);
  if (it == history_.end() || it->second.empty()) {
    return stats;
  }

  double now = PrismRequestTracker::now_seconds();
  double cutoff = now - kWindowSeconds;

  for (const auto& record : it->second) {
    if (record.timestamp >= cutoff) {
      stats.total_reqs += record.total_count;
      stats.violated_count += record.violated_count;
    }
  }
  if (stats.total_reqs > 0) {
    stats.violation_proportion =
        static_cast<double>(stats.violated_count) / stats.total_reqs;
  }
  return stats;
}

// ============================================================================
// PrismMemoryTracker
// ============================================================================

void PrismMemoryTracker::prune_old_records(double now) {
  double cutoff = now - kWindowSeconds;
  for (auto& [inst, records] : history_) {
    while (!records.empty() && records.front().timestamp < cutoff) {
      records.pop_front();
    }
  }
}

void PrismMemoryTracker::update(
    const std::unordered_map<std::string, std::vector<std::string>>&
        instance_to_models,
    const std::unordered_map<std::string, int32_t>& instance_total_reqs,
    const std::unordered_map<std::string, double>& model_weights_gb,
    double gpu_mem_gb) {
  std::lock_guard<std::mutex> lock(mutex_);
  double now = PrismRequestTracker::now_seconds();
  prune_old_records(now);

  for (const auto& [instance_name, models] : instance_to_models) {
    double total_model_weight = 0.0;
    for (const auto& model : models) {
      auto wit = model_weights_gb.find(model);
      if (wit != model_weights_gb.end()) {
        total_model_weight += wit->second;
      }
    }

    int32_t total_reqs = 0;
    auto rit = instance_total_reqs.find(instance_name);
    if (rit != instance_total_reqs.end()) {
      total_reqs = rit->second;
    }

    double mem_per_req =
        (gpu_mem_gb - total_model_weight) / std::max(1, total_reqs);
    history_[instance_name].push_back({now, mem_per_req, total_reqs});
  }
}

PrismMemoryTracker::MemoryStats PrismMemoryTracker::get_instance_stats(
    const std::string& instance_name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  MemoryStats stats;
  auto it = history_.find(instance_name);
  if (it == history_.end() || it->second.empty()) {
    return stats;
  }

  double now = PrismRequestTracker::now_seconds();
  double cutoff = now - kWindowSeconds;

  double weighted_sum = 0.0;
  double total_reqs_sum = 0.0;
  int count = 0;
  for (const auto& record : it->second) {
    if (record.timestamp >= cutoff) {
      weighted_sum += record.memory_per_request * record.total_reqs;
      total_reqs_sum += record.total_reqs;
      ++count;
    }
  }
  if (count > 0) {
    stats.avg_memory_per_request =
        (total_reqs_sum > 0) ? weighted_sum / total_reqs_sum : 0.0;
    stats.avg_total_reqs = total_reqs_sum / count;
  }
  return stats;
}

// ============================================================================
// PrismModelRequestTracker
// ============================================================================

void PrismModelRequestTracker::prune_old_records(double now) {
  double cutoff = now - kWindowSeconds;
  for (auto& [model, records] : history_) {
    while (!records.empty() && records.front().timestamp < cutoff) {
      records.pop_front();
    }
  }
}

void PrismModelRequestTracker::update(
    const std::unordered_map<std::string, PrismModelQueue>& queues) {
  std::lock_guard<std::mutex> lock(mutex_);
  double now = PrismRequestTracker::now_seconds();
  prune_old_records(now);

  for (const auto& [model, queue] : queues) {
    int32_t count = static_cast<int32_t>(queue.waiting_reqs.size() +
                                          queue.running_reqs.size());
    history_[model].push_back({now, count});
  }
}

double PrismModelRequestTracker::get_avg_request_count(
    const std::string& model) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = history_.find(model);
  if (it == history_.end() || it->second.empty()) {
    return 0.0;
  }

  double now = PrismRequestTracker::now_seconds();
  double cutoff = now - kWindowSeconds;

  double sum = 0.0;
  int count = 0;
  for (const auto& record : it->second) {
    if (record.timestamp >= cutoff) {
      sum += record.count;
      ++count;
    }
  }
  return count > 0 ? sum / count : 0.0;
}

}  // namespace xllm_service
