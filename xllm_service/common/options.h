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

#include <string>

#include "common/macros.h"

namespace xllm_service {

class Options {
 public:
  Options() = default;
  ~Options() = default;

  // http server options
  PROPERTY(std::string, server_host);

  PROPERTY(int32_t, http_port) = 9998;

  PROPERTY(int32_t, http_idle_timeout_s) = -1;

  PROPERTY(int32_t, http_num_threads) = 32;

  PROPERTY(int32_t, http_max_concurrency) = 0;

  // rpc server options
  PROPERTY(int32_t, rpc_port) = 9999;

  PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  PROPERTY(int32_t, rpc_num_threads) = 32;

  PROPERTY(int32_t, rpc_max_concurrency) = 0;

  PROPERTY(int32_t, num_threads) = 32;

  PROPERTY(int32_t, max_concurrency) = 32;

  PROPERTY(int32_t, timeout_ms) = 32;

  // instance manager options
  PROPERTY(std::string, etcd_addr);

  PROPERTY(int32_t, detect_disconnected_instance_interval) = 15;

  // scheduler options
  PROPERTY(std::string, load_balance_policy);

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(uint32_t, murmur_hash3_seed) = 1024;

  PROPERTY(std::string, service_name);

  // tokenizer options
  PROPERTY(std::string, tokenizer_path);

  // trace options
  PROPERTY(bool, enable_request_trace) = false;

  // default TTFT SLO in milliseconds (used when request doesn't specify one)
  PROPERTY(int32_t, default_ttft_slo_ms) = 30000;

  // LST-IMH pre-pull threshold in milliseconds
  PROPERTY(int32_t, lst_imh_pre_pull_ms) = 0;

  // Prism baseline options
  PROPERTY(std::string, baseline_type);  // "" = default, "PRISM" = Prism mode
  PROPERTY(double, prism_schedule_interval_s) = 5.0;
  PROPERTY(double, prism_memory_pool_budget_gb) = 6.0;
  PROPERTY(double, prism_idle_threshold_s) = 50.0;
  PROPERTY(std::string, prism_migrate_policy) = "memory_per_request";
  PROPERTY(int32_t, prism_max_models_per_instance) = 4;
  PROPERTY(int32_t, prism_backend_queue_threshold) = 10;

  // ServerlessLLM baseline options
  PROPERTY(double, sllm_schedule_interval_s) = 1.0;
  PROPERTY(double, sllm_idle_threshold_s) = 60.0;
  PROPERTY(double, sllm_d2d_speed_gbps) = 25.0;
  PROPERTY(double, sllm_h2d_speed_gbps) = 6.0;
  PROPERTY(double, sllm_drain_alpha) = 0.001;
  PROPERTY(double, sllm_drain_beta) = 0.5;
  PROPERTY(int32_t, sllm_target_ongoing_requests) = 8;
  PROPERTY(int32_t, sllm_min_instances) = 0;
  PROPERTY(int32_t, sllm_max_instances) = 8;
  PROPERTY(bool, sllm_enable_knapsack) = true;
  PROPERTY(int32_t, sllm_max_models_per_instance) = 4;

  // Llumnix baseline options
  PROPERTY(double, llumnix_schedule_interval_s) = 1.0;
  PROPERTY(double, llumnix_idle_threshold_s) = 60.0;
  PROPERTY(double, llumnix_migrate_out_load_threshold) = 0.8;
  PROPERTY(int32_t, llumnix_topk_random_dispatch) = 1;
  PROPERTY(int32_t, llumnix_max_models_per_instance) = 4;
  PROPERTY(int32_t, llumnix_min_instances) = 0;
  PROPERTY(int32_t, llumnix_max_instances) = 8;
  PROPERTY(std::string, llumnix_dispatch_load_metric) = "remaining_steps";
  PROPERTY(std::string, llumnix_migration_load_metric) = "remaining_steps";
  PROPERTY(std::string, llumnix_dispatch_policy) = "load";
  PROPERTY(std::string, llumnix_migration_policy) = "defrag";
  PROPERTY(double, llumnix_dispatch_busy_threshold) = 1.0;
  PROPERTY(double, llumnix_dispatch_busy_threshold_remaining_steps) = 10.0;
};

}  // namespace xllm_service