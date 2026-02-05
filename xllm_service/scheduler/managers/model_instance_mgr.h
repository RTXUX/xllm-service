/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <brpc/channel.h>

#include <deque>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class ModelInstanceMgr {
 public:
  ModelInstanceMgr(const std::string& model_id);
  ~ModelInstanceMgr();

  bool get_next_instance_pair(Routing* routing);
  
  // Instance management
  void add_instance(const std::string& instance_name, const InstanceMetaInfo& info);
  void remove_instance(const std::string& instance_name);
  void update_instance_info(const std::string& instance_name, const InstanceMetaInfo& info);
  
  // State management
  void set_instance_state(const std::string& instance_name, const std::string& model_id, int state);
  
  // Scheduling helpers
  void flip_prefill_to_decode(const std::string& instance_name);
  void flip_decode_to_prefill(const std::string& instance_name);
  
  std::vector<std::string> get_prefill_list() const;
  std::vector<std::string> get_decode_list() const;

  bool send_model_sleep(const std::string& instance_name, std::shared_ptr<brpc::Channel> channel);
  bool send_model_wakeup(const std::string& instance_name, std::shared_ptr<brpc::Channel> channel);
  // D2D wakeup: transfer weights from remote device instead of host
  bool send_model_wakeup_d2d(const std::string& instance_name,
                             std::shared_ptr<brpc::Channel> channel,
                             const D2DWakeupInfo& d2d_info);
  bool set_model_state(const std::string& instance_name, ModelState state);
  ModelState get_model_state(const std::string& instance_name);
  bool is_model_waking_up();
  int32_t get_wakeup_count();
  int32_t get_allocation_count();
  std::vector<std::string> get_awake_instances();
  // Get awake instances that are not locked (can be evicted)
  std::vector<std::string> get_unlocked_instances();
  // Atomically get awake instances and lock all of them, returns locked instance list
  std::vector<std::string> get_awake_instances_and_lock();

  // D2D reference counting - protect source instances from being slept during D2D transfer
  void acquire_d2d_lock(const std::string& instance_name);
  void release_d2d_lock(const std::string& instance_name);
  // Batch unlock multiple instances
  void release_d2d_locks(const std::vector<std::string>& instance_names);
  bool can_sleep(const std::string& instance_name);
  int32_t get_d2d_ref_count(const std::string& instance_name);

  void update_model_heat(int64_t token_count);
  int64_t get_model_heat();

  std::shared_mutex* get_instance_state_single_mutex(const std::string& instance_name);

  void auto_flipping(const std::unordered_map<std::string, LatencyMetrics>& latency_metrics);

 private:
  bool send_http_request(std::shared_ptr<brpc::Channel> channel,
                         const std::string& uri,
                         const std::string& request_body);

  // Must be called while holding model_heat_mutex_
  void prune_model_heat_locked();

  std::string model_id_;

  // Lock hierarchy within ModelInstanceMgr (acquire in this order):
  //   1. mutex_                       — protects instances_, prefill_index_, decode_index_
  //   2. instance_state_all_mutex_   — protects instance_states_, wakeup_count_, allocation_count_
  //   3. d2d_ref_mutex_               — protects d2d_ref_counts_
  //   4. model_heat_mutex_            — protects model_heat_records_, model_heat_
  //   5. wakeup_mutex_                — protects wakeup_cv_, wakeup_instance_name_
  //
  // These are all leaf-level locks relative to InstanceMgr's locks.
  // Do not hold any of these when acquiring InstanceMgr locks.
  
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;
  
  // Scheduling indices
  std::vector<std::string> prefill_index_;
  std::vector<std::string> decode_index_;
  uint32_t next_prefill_index_ = 0;
  uint32_t next_decode_index_ = 0;

  // Instance states
  // the all_mutex protects instance_states_, wakeup_count_, allocation_count_
  // the single_mutexes are designed specifically to reduce contention of send_http_request (towards different instances)
  std::shared_mutex instance_state_all_mutex_;
  std::unordered_map<std::string, std::unique_ptr<std::shared_mutex>> instance_state_single_mutexes_;
  std::unordered_map<std::string, ModelState> instance_states_;
  // How many instances are in WAKEUP state
  int32_t wakeup_count_ = 0;
  // How many instances are already allocated, i.e. WAKING_UP || SENDING_WAKEUP_REQUEST || WAKEUP
  int32_t allocation_count_ = 0;

  // Wait logic
  std::mutex wakeup_mutex_;
  std::condition_variable wakeup_cv_;
  std::string wakeup_instance_name_;

  // Model heat (token count) tracking
  struct HeatRecord {
    std::chrono::steady_clock::time_point timestamp;
    int64_t token_count;
  };
  static constexpr int64_t kModelHeatRetentionSeconds = 5;
  std::mutex model_heat_mutex_;
  std::deque<HeatRecord> model_heat_records_;
  int64_t model_heat_ = 0;

  // D2D reference counting - protects source instances during D2D transfer
  std::mutex d2d_ref_mutex_;
  std::unordered_map<std::string, int32_t> d2d_ref_counts_;

};

}  // namespace xllm_service