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

#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <tuple>
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
#include "scheduler/resource_model/resource_model.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class InstanceMgr final {
 public:

  const std::vector<std::pair<std::string, std::string>> MODELS = {
    {"Qwen3-8B", "/export/home/models/Qwen3-8B"},
    {"Qwen2-7B", "/export/home/models/Qwen2-7B"}
    // {"Qwen2.5-14B", "/export/home/models/Qwen2.5-14B"},
    // {"Qwen3-4B", "/export/home/models/Qwen3-4B"}
    // {"Qwen2.5-3b", "/export/home/models/Qwen2.5-3b"}
    // {"Qwen3-30B-A3B-Instruct-2507", "/export/home/models/Qwen3-30B-A3B-Instruct-2507"}
    // {"Qwen3-30B-A3B-W8A8", "/export/home/models/Qwen3-30B-A3B-W8A8"},
    // {"Qwen3-32B-W8A8", "/export/home/models/Qwen3-32B-W8A8"}
  };

  std::atomic<uint16_t> master_node_port = 40000;
  
  static constexpr int kMaxWakeupTimeoutms = 10000;

  static constexpr int kTensorParallelSize = 1;

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

  // Get the estimated prefill done time (absolute ms since epoch) for an instance
  int64_t get_estimated_prefill_done_time(const std::string& instance_name);

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

  // Trigger per-request auto-scaling. Computes GPU targets for elastic pool models
  // using resource_model, then directly wakes/sleeps instances.
  // Scale-up is blocking; scale-down spawns async drain threads.
  // Budget = total_gpus - steady_needed_gpus.
  void dynamic_part_auto_scaling();

  std::shared_ptr<ModelInstanceMgr> get_model_instance_mgr(const std::string& model_id);

  // --- Dual-pool scheduling ---

  // Assign a new model to a pool based on current gpu_target.
  // If gpu_target <= 1: steady pool (bin-packed). If > 1: elastic pool.
  // Blocks until wakeup completes for steady pool models.
  void assign_model_to_pool(const std::string& model_id);

  // Get current pool assignment for a model
  PoolType get_model_pool(const std::string& model_id);

  // Check if steady pool model needs upgrade to elastic. Returns true if upgraded.
  bool steady_part_check_upgrading(const std::string& model_id);

  // Periodic: sleep models with 0 heat, repack bins, release unused instances.
  void steady_part_auto_repacking();

  // Periodic: demote elastic models with sustained low demand (gpu_target<=1)
  // back to steady pool via find_or_create_steady_bin.
  void elastic_to_steady_demotion();

  // Route a request to the steady pool instance hosting the model.
  bool route_to_steady_instance(std::shared_ptr<Request> request);

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

  // Predict TTFT for a specific instance (heterogeneous instances, for SLO_AWARE)
  double predict_ttft(const std::string& instance_name,
                      const std::string& model_id, int32_t token_count);

  // Predict TTFT using any available instance (homogeneous instances, for LST_IMH)
  double predict_ttft_any_instance(const std::string& model_id, int32_t token_count);

  // Block until model on instance has 0 inflight requests. Returns false on error.
  bool wait_for_model_drain(const std::string& instance_name,
                             const std::string& model_id);

  // Locally adjust free pages after allocation (before next heartbeat)
  void deduct_free_pages(const std::string& instance_name, uint64_t bytes);

 private:
  void init_model_memory_specs();
  void init_model_resource_coefficients();
  double get_model_memory_size(const std::string& model_id);

  // --- Dual-pool private helpers ---

  // 2D First Fit: find a bin for a model, or create a new one.
  // If all instances occupied, reclaims from elastic pool (blocking).
  // Must be called with allocation_mutex_ held.
  // When allow_reclaim is false, skip Phase 3 (elastic pool reclamation).
  std::string find_or_create_steady_bin(const std::string& model_id,
                                         const ResourceNeeds& needs,
                                         bool allow_reclaim = true);

  // Full FFD repack of all steady bins.
  // Returns list of (model_id, old_instance, new_instance) moves.
  // Must be called with allocation_mutex_ held.
  std::vector<std::tuple<std::string, std::string, std::string>>
      repack_steady_bins();

  // Remove a model from its steady bin, reclaiming resources.
  // Must be called with allocation_mutex_ held.
  void remove_model_from_steady_bin(const std::string& model_id);

  // Get resource needs for a model using its current heat.
  ResourceNeeds get_model_resource_needs(const std::string& model_id);

  // Check if instance is in the steady pool
  bool is_steady_pool_instance(const std::string& instance_name);

  // Get number of GPUs reserved by the steady pool
  int32_t steady_needed_gpus();

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

  // Lock hierarchy (acquire in this order to prevent deadlocks):
  //   1. allocation_mutex_           — outermost, serializes allocation decisions
  //   2. inst_mutex_                 — protects instances_, cached_channels_
  //   3. model_instance_mgr_mutex_  — protects model_instance_mgrs_ map
  //   4. xtensor_info_mutex_         — protects instance_xtensor_infos_
  //   5. pending_mutex_              — protects pending_infos_
  //   6. time_predictor_mutex_       — protects time_predictors_
  //   7. request_metrics_mutex_      — protects request_metrics_
  //   8. latency_metrics_mutex_      — protects latency_metrics_
  //   9. update_mutex_               — protects updated_metrics_, removed_instance_
  //  10. load_metric_mutex_          — protects load_metrics_
  //
  // ModelInstanceMgr internal locks (mutex_, instance_state_all_mutex_,
  // d2d_ref_mutex_, model_heat_mutex_) are leaf-level locks and must not
  // be held when acquiring any InstanceMgr mutex.

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

  // instances_ use shared_mutex because they only change when
  // instance registration or model state changes.
  // pending_infos_ use mutex because there's no read-only operations
  
  std::shared_mutex inst_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;

  std::shared_mutex model_instance_mgr_mutex_;
  std::unordered_map<std::string, std::shared_ptr<ModelInstanceMgr>> model_instance_mgrs_;

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

  // Per-instance list of inflight prefill requests (dispatched but not yet
  // FINISH_PREFILL / CANCEL). Used to propagate EPDT corrections to all
  // remaining requests so that later FINISH_PREFILL does not double-count.
  // Protected by request_metrics_mutex_.
  std::unordered_map<std::string, std::vector<std::shared_ptr<Request>>>
      inflight_prefill_requests_;  
  
  std::mutex allocation_mutex_;

  // Two-pool auto-scaling state
  std::atomic<int32_t> total_available_gpus_{0};
  // model_id -> resource model (read-only after init)
  std::unordered_map<std::string, std::unique_ptr<ResourceModel>> model_resource_models_;
  // GPU hardware spec (read-only after init)
  GpuHardwareSpec gpu_hw_spec_;

  // --- Dual-pool state (protected by allocation_mutex_) ---
  // model_id -> which pool it belongs to
  std::unordered_map<std::string, PoolType> model_pool_assignments_;
  // Steady pool bins (each bin = one instance with co-located models)
  std::vector<SteadyBin> steady_bins_;
  // Instances occupied by elastic pool (one model each)
  std::unordered_set<std::string> elastic_occupied_instances_;
  // Pending steady pool GPU reservations (used during instance reclamation)
  int32_t pending_steady_gpus_ = 0;
  // Repack timer thread
  std::unique_ptr<std::thread> repack_thread_;

  // --- Elastic-to-steady demotion state (protected by allocation_mutex_) ---
  // model_id -> timestamp when gpu_target first dropped to <= 1
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      elastic_low_demand_since_;
  // Demotion check thread
  std::unique_ptr<std::thread> demotion_thread_;

  // XTensor memory info per instance
  std::mutex xtensor_info_mutex_;
  std::unordered_map<std::string, InstanceXTensorInfo> instance_xtensor_infos_;

  // Instances that have completed fork_master for all models (ready for D2D linking)
  std::mutex fork_done_mutex_;
  std::unordered_set<std::string> fork_done_instances_;

  ThreadPool threadpool_;
};

}  // namespace xllm_service
