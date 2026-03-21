#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xllm_service {

// Multi-tier storage state tracker for ServerlessLLM.
// Tracks which models are loaded on which instances and their storage tier:
//   Tier 0: Model is WAKEUP on the instance (GPU memory, latency=0)
//   Tier 1: Model has a D2D source available (another instance has it in GPU)
//   Tier 2: Model must be loaded from host memory/disk (H2D)
class ServerlessLLMStoreMgr {
 public:
  ServerlessLLMStoreMgr() = default;
  ~ServerlessLLMStoreMgr() = default;

  // Get storage tier for a model on a specific instance.
  // Returns 0 (GPU), 1 (D2D available), or 2 (H2D only).
  int get_storage_tier(const std::string& model_id,
                       const std::string& instance_name) const;

  // Update the set of awake (Tier 0) models per instance.
  // Called from scheduling_loop to sync with InstanceMgr state.
  void update_awake_models(
      const std::unordered_map<std::string, std::vector<std::string>>&
          instance_to_awake_models);

  // Update the set of models that have D2D sources available.
  // A model has D2D if at least one other instance has it awake.
  void update_d2d_sources(
      const std::unordered_set<std::string>& models_with_d2d);

  // --- I/O queue modeling ---
  // Track per-instance I/O operations to estimate queue wait time.
  void record_io_start(const std::string& instance_name);
  void record_io_complete(const std::string& instance_name);
  double get_io_queue_wait(const std::string& instance_name) const;

  // --- LRU tracking ---
  // Touch a model to update its last-used timestamp.
  void touch_model(const std::string& model_id,
                   const std::string& instance_name);

  // Get idle duration in seconds for a model on an instance.
  double get_idle_duration(const std::string& model_id,
                           const std::string& instance_name) const;

  // Get models sorted by idle time (longest idle first) for an instance.
  // Returns (model_id, idle_seconds) pairs.
  std::vector<std::pair<std::string, double>> get_lru_order(
      const std::string& instance_name) const;

 private:
  static double now_seconds();

  mutable std::mutex mutex_;

  // instance_name -> set of model_ids that are WAKEUP (Tier 0)
  std::unordered_map<std::string, std::unordered_set<std::string>>
      awake_models_;

  // Set of model_ids that have at least one D2D source (Tier 1)
  std::unordered_set<std::string> d2d_available_models_;

  // (instance_name, model_id) -> last touch timestamp
  std::unordered_map<std::string, std::unordered_map<std::string, double>>
      last_touch_time_;

  // instance_name -> number of active I/O operations
  std::unordered_map<std::string, int> active_io_count_;

  // Estimated time per I/O operation (seconds)
  static constexpr double kEstimatedIoTimeSec = 5.0;
};

}  // namespace xllm_service
