#include "scheduler/serverless_llm/serverless_llm_store_mgr.h"

#include <algorithm>
#include <chrono>

namespace xllm_service {

double ServerlessLLMStoreMgr::now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int ServerlessLLMStoreMgr::get_storage_tier(
    const std::string& model_id, const std::string& instance_name) const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Tier 0: model is WAKEUP on this instance
  auto it = awake_models_.find(instance_name);
  if (it != awake_models_.end() &&
      it->second.find(model_id) != it->second.end()) {
    return 0;
  }

  // Tier 1: D2D source available (another instance has it)
  if (d2d_available_models_.find(model_id) != d2d_available_models_.end()) {
    return 1;
  }

  // Tier 2: H2D only
  return 2;
}

void ServerlessLLMStoreMgr::update_awake_models(
    const std::unordered_map<std::string, std::vector<std::string>>&
        instance_to_awake_models) {
  std::lock_guard<std::mutex> lock(mutex_);
  awake_models_.clear();
  for (const auto& [instance_name, models] : instance_to_awake_models) {
    awake_models_[instance_name] =
        std::unordered_set<std::string>(models.begin(), models.end());
  }
}

void ServerlessLLMStoreMgr::update_d2d_sources(
    const std::unordered_set<std::string>& models_with_d2d) {
  std::lock_guard<std::mutex> lock(mutex_);
  d2d_available_models_ = models_with_d2d;
}

void ServerlessLLMStoreMgr::record_io_start(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  active_io_count_[instance_name]++;
}

void ServerlessLLMStoreMgr::record_io_complete(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = active_io_count_.find(instance_name);
  if (it != active_io_count_.end() && it->second > 0) {
    it->second--;
  }
}

double ServerlessLLMStoreMgr::get_io_queue_wait(
    const std::string& instance_name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = active_io_count_.find(instance_name);
  if (it == active_io_count_.end() || it->second == 0) {
    return 0.0;
  }
  return it->second * kEstimatedIoTimeSec;
}

void ServerlessLLMStoreMgr::touch_model(const std::string& model_id,
                                         const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  last_touch_time_[instance_name][model_id] = now_seconds();
}

double ServerlessLLMStoreMgr::get_idle_duration(
    const std::string& model_id, const std::string& instance_name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto inst_it = last_touch_time_.find(instance_name);
  if (inst_it == last_touch_time_.end()) {
    return 0.0;
  }
  auto model_it = inst_it->second.find(model_id);
  if (model_it == inst_it->second.end()) {
    return 0.0;
  }
  return now_seconds() - model_it->second;
}

std::vector<std::pair<std::string, double>>
ServerlessLLMStoreMgr::get_lru_order(const std::string& instance_name) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::pair<std::string, double>> result;
  double now = now_seconds();

  auto inst_it = last_touch_time_.find(instance_name);
  if (inst_it == last_touch_time_.end()) {
    return result;
  }

  // Only include models that are currently awake on this instance
  auto awake_it = awake_models_.find(instance_name);
  if (awake_it == awake_models_.end()) {
    return result;
  }

  for (const auto& model_id : awake_it->second) {
    auto model_it = inst_it->second.find(model_id);
    double idle = (model_it != inst_it->second.end())
                      ? (now - model_it->second)
                      : 0.0;
    result.push_back({model_id, idle});
  }

  // Sort by idle time descending (longest idle first)
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  return result;
}

}  // namespace xllm_service
