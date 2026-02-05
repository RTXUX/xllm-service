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

#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <chrono>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/time_predictor.h"
#include "common/types.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "scheduler/managers/model_instance_mgr.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class InstanceMgr final {
 public:

  const std::vector<std::pair<std::string, std::string>> MODELS = {
    {"Qwen3-8B", "/export/home/models/Qwen3-8B"},
    {"Qwen2-7B", "/export/home/models/Qwen2-7B"},
    {"Qwen2.5-14B", "/export/home/models/Qwen2.5-14B"},
    {"Qwen3-4B", "/export/home/models/Qwen3-4B"}
    // {"Qwen2.5-3b", "/export/home/models/Qwen2.5-3b"}
    // {"Qwen3-30B-A3B-Instruct-2507", "/export/home/models/Qwen3-30B-A3B-Instruct-2507"}
    // {"Qwen3-30B-A3B-W8A8", "/export/home/models/Qwen3-30B-A3B-W8A8"},
    // {"Qwen3-32B-W8A8", "/export/home/models/Qwen3-32B-W8A8"}
  };

  std::atomic<uint16_t> master_node_port = 40000;
  
  static constexpr int kMaxWakeupTimeoutms = 10000;

  static constexpr int kTensorParallelSize = 2;

 public:
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  bool get_next_instance_pair(const std::string& model_id, Routing* routing);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  void get_load_metrics(LoadBalanceInfos* infos);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  void record_load_metrics_update(const std::string& instance_name,
                                  const proto::LoadMetrics& load_metrics);
  bool upload_load_metrics();

  // update the recent token latency metrics for the corresponding instance
  void update_latency_metrics(const std::string& instance_name,
                              const proto::LatencyMetrics& latency_metrics);

  // update request metrics under different actions
  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  // select instances based on the SLO
  bool select_instance_pair_on_slo(std::shared_ptr<Request> request);

  void set_as_master();

  void fork_master_and_sleep(const std::string& instance_name,
                             std::shared_ptr<brpc::Channel> channel);

  void on_heartbeat(const std::string& instance_name);

  void send_model_sleep(const std::string& instance_name,
                        const std::string& model_id);

  void send_model_wakeup(const std::string& instance_name,
                         const std::string& model_id,
                         bool memory_increased_in_advance);

  void update_model_heat(const std::string& model_id,
                         int64_t token_count);
  
  int32_t get_wakeup_count(const std::string& model_id);

  std::vector<std::string> get_awake_instances(const std::string& model_id);
  bool is_model_waking_up(const std::string& model_id);

  std::vector<std::string> allocate_instance_for_model(const std::string& model_id,
                                                       int32_t target_model_count);
  void auto_scaling();

  ModelInstanceMgr* get_model_instance_mgr(const std::string& model_id);

  // Update XTensor info for an instance from heartbeat
  void update_xtensor_info(const std::string& instance_name,
                           const proto::XTensorHeartbeatInfo& xtensor_info);

  // Get XTensor info for an instance
  std::optional<InstanceXTensorInfo> get_instance_xtensor_info(
      const std::string& instance_name);

  // Find a suitable source instance for D2D weight transfer
  // Returns D2DWakeupInfo with source instance info, or empty if no suitable source found
  std::optional<D2DWakeupInfo> find_d2d_source(const std::string& model_id,
                                                const std::string& target_instance_name);

  // Check if an instance has valid xtensor info from heartbeat
  bool has_valid_xtensor_info(const std::string& instance_name);

  // Get model size in bytes (prefers xtensor info, fallback to specs)
  uint64_t get_model_size_bytes(const std::string& model_id);

  // Get instance free space in bytes from xtensor info
  uint64_t get_instance_free_bytes(const std::string& instance_name);

  // Check if instance has enough space to load a model
  bool has_enough_space_for_model(const std::string& instance_name,
                                   const std::string& model_id);

 private:
  void init_model_memory_specs();
  double get_model_memory_size(const std::string& model_id);
  // Select models to evict on a specific instance to free up required_space
  EvictionPlanInfo select_eviction_candidates(const std::string& instance_name, double required_space);

  // Compute max contiguous free space after evicting specified models
  uint64_t compute_max_contiguous_free_space(
      const InstanceXTensorInfo& xtensor_info,
      const std::vector<std::string>& models_to_evict,
      uint64_t total_memory_bytes);

  // Locally adjust free pages after allocation (before next heartbeat)
  void deduct_free_pages(const std::string& instance_name, uint64_t bytes);

 private:

  // send_http_request(instance_name, ...) uses inst_mutex to get_channel()
  // send_http_request(channel, ...) does not use inst_mutex
  bool send_http_request(const std::string& instance_name,
                         const std::string& uri,
                         const std::string& request_body);

  bool send_http_request(std::shared_ptr<brpc::Channel> channel,
                         const std::string& uri,
                         const std::string& request_body);

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  void init();

  bool create_channel(const std::string& target_uri);
  // use etcd as ServiceDiscovery
  void update_instance_metainfo(const etcd::Response& response,
                                const uint64_t& prefix_len);

  void update_load_metrics(const etcd::Response& response,
                           const uint64_t& prefix_len);

  TimePredictor& get_time_predictor(const std::string& instance_name);

  void flip_prefill_to_decode(std::string& instance_name);

  void flip_decode_to_prefill(std::string& instance_name);

  void register_instance(const std::string& instance_name,
                         InstanceMetaInfo metainfo);

 private:
  Options options_;

  bool exited_ = false;
  bool use_etcd_ = false;
  std::atomic_bool is_master_service_ = false;

  std::shared_ptr<EtcdClient> etcd_client_;

  static constexpr double kMaxInstanceMemoryGB = 60.0;
  // model_id -> memory_spec (GB)
  std::unordered_map<std::string, double> model_memory_specs_;

  // instances_, instance_model_states_, model_count_ use shared_mutex
  // because they only change when instance registration or model state changes
  // global_model_heat_ and instance_memory_usage_ use mutex
  // because they change frequently when requests come
  // pending_infos_ use mutex because there's no read-only operations
  
  std::shared_mutex inst_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;

  std::shared_mutex model_instance_mgr_mutex_;
  std::unordered_map<std::string, std::shared_ptr<ModelInstanceMgr>> model_instance_mgrs_;

  // instance_name -> current_memory_usage (GB)
  std::mutex instance_memory_mutex_;
  std::unordered_map<std::string, double> instance_memory_usage_;

  // stores InstanceMetaInfo before receiving heartbeat
  std::mutex pending_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> pending_infos_;

  std::shared_mutex load_metric_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;

  std::mutex update_mutex_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;

  // "instance name" -> "TimePredictor" map
  std::mutex time_predictor_mutex_;
  std::unordered_map<std::string, TimePredictor> time_predictors_;

  // Record the latest token latency metrics for each instance, including TTFT
  // and TBT.
  std::mutex latency_metrics_mutex_;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics_;

  // Record the request metrics for each instance, including prefill token
  // count, prefill request count, estimated prefill execution time, decode
  // token count, and decode request count.
  std::mutex request_metrics_mutex_;
  std::unordered_map<std::string, RequestMetrics> request_metrics_;  
  
  std::mutex allocation_mutex_;

  // XTensor memory info per instance
  std::mutex xtensor_info_mutex_;
  std::unordered_map<std::string, InstanceXTensorInfo> instance_xtensor_infos_;

  ThreadPool threadpool_;
};

}  // namespace xllm_service
