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

#include <gflags/gflags.h>

DECLARE_string(server_host);

DECLARE_int32(http_server_port);

DECLARE_int32(http_server_idle_timeout_s);

DECLARE_int32(http_server_num_threads);

DECLARE_int32(http_server_max_concurrency);

DECLARE_int32(rpc_server_port);

DECLARE_int32(rpc_server_idle_timeout_s);

DECLARE_int32(rpc_server_num_threads);

DECLARE_int32(rpc_server_max_concurrency);

DECLARE_uint32(murmur_hash3_seed);

DECLARE_int32(timeout_ms);

DECLARE_string(listen_addr);

DECLARE_int32(port);

DECLARE_int32(idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_int32(max_concurrency);

DECLARE_string(etcd_addr);

DECLARE_string(load_balance_policy);

DECLARE_int32(detect_disconnected_instance_interval);

DECLARE_int32(block_size);

DECLARE_string(tokenizer_path);

DECLARE_bool(enable_request_trace);

DECLARE_int32(target_ttft);

DECLARE_int32(target_tpot);

DECLARE_int32(default_ttft_slo_ms);

DECLARE_int32(lst_imh_pre_pull_ms);

DECLARE_bool(enable_prefill_only_mode);

DECLARE_double(gpu_hbm_per_gpu_gb);

DECLARE_double(gpu_compute_sm_per_gpu);

// Prism baseline flags
DECLARE_string(baseline_type);
DECLARE_double(prism_schedule_interval_s);
DECLARE_double(prism_memory_pool_budget_gb);
DECLARE_double(prism_idle_threshold_s);
DECLARE_string(prism_migrate_policy);
DECLARE_int32(prism_max_models_per_instance);
DECLARE_int32(prism_backend_queue_threshold);

// ServerlessLLM baseline flags
DECLARE_double(sllm_schedule_interval_s);
DECLARE_double(sllm_idle_threshold_s);
DECLARE_double(sllm_d2d_speed_gbps);
DECLARE_double(sllm_h2d_speed_gbps);
DECLARE_double(sllm_drain_alpha);
DECLARE_double(sllm_drain_beta);
DECLARE_int32(sllm_target_ongoing_requests);
DECLARE_int32(sllm_min_instances);
DECLARE_int32(sllm_max_instances);
DECLARE_bool(sllm_enable_knapsack);
DECLARE_int32(sllm_max_models_per_instance);

// BlitzScale baseline flags
DECLARE_double(blitzscale_schedule_interval_s);
DECLARE_double(blitzscale_scale_down_threshold_ms);
DECLARE_uint32(blitzscale_tokens_prefilled_per_sec);
DECLARE_uint32(blitzscale_tokens_transferred_per_sec);
DECLARE_uint32(blitzscale_max_blocks_per_replica);
DECLARE_double(blitzscale_prefill_lower_bound);
DECLARE_double(blitzscale_prefill_upper_bound);
DECLARE_double(blitzscale_decode_lower_bound);
DECLARE_double(blitzscale_decode_upper_bound);
DECLARE_double(blitzscale_migration_lower_bound);
DECLARE_double(blitzscale_migration_upper_bound);
DECLARE_int32(blitzscale_min_prefill_instances);
DECLARE_int32(blitzscale_max_prefill_instances);
DECLARE_int32(blitzscale_min_decode_instances);
DECLARE_int32(blitzscale_max_decode_instances);
DECLARE_int32(blitzscale_max_models_per_instance);

// Llumnix baseline flags
DECLARE_double(llumnix_schedule_interval_s);
DECLARE_double(llumnix_idle_threshold_s);
DECLARE_double(llumnix_migrate_out_load_threshold);
DECLARE_int32(llumnix_topk_random_dispatch);
DECLARE_int32(llumnix_max_models_per_instance);
DECLARE_int32(llumnix_min_instances);
DECLARE_int32(llumnix_max_instances);
DECLARE_string(llumnix_dispatch_load_metric);
DECLARE_string(llumnix_migration_load_metric);
DECLARE_string(llumnix_dispatch_policy);
DECLARE_string(llumnix_migration_policy);
DECLARE_double(llumnix_dispatch_busy_threshold);
DECLARE_double(llumnix_dispatch_busy_threshold_remaining_steps);
