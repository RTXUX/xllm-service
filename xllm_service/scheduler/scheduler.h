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

#include "chat_template/jinja_chat_template.h"
#include "common/call_data.h"
#include "common/concurrent_queue.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/xllm/output.h"
#include "etcd_client/etcd_client.h"
#include "loadbalance_policy/loadbalance_policy.h"
#include "managers/global_kvcache_mgr.h"
#include "managers/instance_mgr.h"
#include "request/request.h"
#include "response_handler.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

#include <condition_variable>
#include <queue>

namespace xllm_service {

// Job representation for the scheduling algorithm
struct SchedulingJob {
  std::shared_ptr<Request> request;
  int64_t processing_time_ms;  // p_j: estimated prefill time
  int64_t deadline_ms;         // d_j: arrival_time + ttft_slo
};

// Machine (instance) representation for the scheduling algorithm
struct MachineInfo {
  std::string instance_name;
  int64_t availability_time_ms;  // T_i: estimated time until free
};

// A scheduler for scheduling requests and instances
class Scheduler final {
 public:
  Scheduler(const Options& options);
  ~Scheduler();

  bool schedule(std::shared_ptr<Request> request);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& target_name);

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  InstanceMgr* get_instance_mgr() { return instance_mgr_.get(); }

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  void handle_instance_heartbeat(const proto::HeartbeatRequest* req);

  void exited() { exited_ = true; }

  // register new requests from http service
  // keep http callback util request finished.
  // `handle_generation` will handle response with these callbacks.
  bool record_new_request(std::shared_ptr<ChatCallData> call_data,
                          std::shared_ptr<Request> request);
  bool record_new_request(std::shared_ptr<CompletionCallData> call_data,
                          std::shared_ptr<Request> request);
  void finish_request(const std::string& service_request_id,
                      bool error = false);

  // handle generations from prefill/decode instance
  bool handle_generation(const llm::RequestOutput& request_output);

  // update request metrics for prefill finished request
  void update_request_metrics_for_prefill(
      const std::string& service_request_id);

 private:
  DISALLOW_COPY_AND_ASSIGN(Scheduler);

  void update_master_service_heartbeat();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);

  void process_request_queue(const std::string& model_name);

  // LST-IMH dispatch coordinator (runs in dedicated thread)
  void dispatch_coordinator();

  // Signal the dispatch coordinator to wake up
  void signal_dispatch();

  // Run LST-IMH algorithm: assign jobs to machines maximizing on-time completions
  // Returns: instance_name -> vector of assigned jobs (sorted by EDD)
  std::unordered_map<std::string, std::vector<SchedulingJob>>
      run_lst_imh(std::vector<SchedulingJob>& jobs,
                  std::vector<MachineInfo>& machines);

  // Moore-Hodgson algorithm for single machine: maximize on-time jobs
  // Returns jobs that can be completed on-time, sorted by EDD
  std::vector<SchedulingJob> moore_hodgson(
      const std::vector<SchedulingJob>& jobs, int64_t T);

  Tokenizer* get_tls_tokenizer();

 private:
  Options options_;

  bool exited_ = false;

  bool is_master_service_ = false;

  TokenizerArgs tokenizer_args_;

  // chat template instance
  std::unique_ptr<JinjaChatTemplate> chat_template_;

  std::shared_ptr<EtcdClient> etcd_client_;

  std::unique_ptr<Tokenizer> tokenizer_;

  std::shared_ptr<InstanceMgr> instance_mgr_;

  std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr_;

  std::unique_ptr<LoadBalancePolicy> lb_policy_;

  std::unique_ptr<std::thread> heartbeat_thread_;

  // `model name` -> `request queue` map
  std::unordered_map<std::string,
                     std::shared_ptr<ConcurrentQueue<std::shared_ptr<Request>>>>
      request_queues_;
  std::mutex queue_mutex_;

  // `model name` -> `processing thread` map
  std::unordered_map<std::string, std::vector<std::unique_ptr<std::thread>>>
      processing_threads_;


  // `service request id` -> `request` map
  std::unordered_map<std::string, std::shared_ptr<Request>> requests_;
  std::mutex request_mutex_;

  // use threadpool to handle all RequestOuputs queue
  static constexpr size_t kOutputTheadNum_ = 128;  // magic num
  ThreadPool output_threadpools_[kOutputTheadNum_];
  // A request will be handled in the same thread to guarantee the token's
  // order.
  std::unordered_map<std::string, size_t> remote_requests_output_thread_map_;
  size_t next_thread_idx = 0;
  std::mutex thread_map_mutex_;

  // used when receive token from decode instance.
  ResponseHandler response_handler_;

  // === LST-IMH mode members ===

  // Pending request queue (requests waiting to be dispatched)
  std::mutex pending_queue_mutex_;
  std::vector<std::shared_ptr<Request>> pending_queue_;

  // Dispatch coordinator synchronization
  std::mutex dispatch_mutex_;
  std::condition_variable dispatch_cv_;
  bool dispatch_signal_ = false;

  // Dispatch coordinator thread
  std::unique_ptr<std::thread> dispatch_coordinator_thread_;

  // Model name for LST-IMH mode (single model assumption)
  std::string lst_imh_model_name_;
};

}  // namespace xllm_service