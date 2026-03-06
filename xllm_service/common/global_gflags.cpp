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

#include "common/global_gflags.h"

#include "brpc/reloadable_flags.h"

DEFINE_string(server_host,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(http_server_port, 8888, "Port for xllm http service to listen on");

DEFINE_int32(http_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(http_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(http_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(rpc_server_port, 8889, "Port for xllm rpc service to listen on");

DEFINE_int32(rpc_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(rpc_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(rpc_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_string(etcd_addr,
              "0.0.0.0:2379",
              "etcd adderss for save instance meta info");

DEFINE_uint32(murmur_hash3_seed, 1024, "default Murmur Hash seed");

DEFINE_int32(port, 8888, "Port for xllm service to listen on");

DEFINE_int32(num_threads, 32, "Number of threads to process requests");

DEFINE_int32(max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(timeout_ms,
             -1,
             "Max duration of bRPC Channel. -1 means wait indefinitely.");

DEFINE_string(listen_addr,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_string(load_balance_policy,
              "RR",
              "Disaggregated prefill-decode policy.");

DEFINE_int32(detect_disconnected_instance_interval,
             15,
             "The interval that server detect the disconnected instance.");

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_string(tokenizer_path, "", "tokenizer config path.");

DEFINE_bool(enable_request_trace, false, "Whether to enable request trace");

DEFINE_int32(default_ttft_slo_ms,
             30000,
             "Default TTFT SLO in milliseconds when request doesn't specify one.");

DEFINE_int32(lst_imh_pre_pull_ms,
             0,
             "LST-IMH pre-pull threshold in ms. Dispatch coordinator pulls "
             "requests when instance estimated remaining time <= this value. "
             "0 means wait until instance is completely idle (default).");

DEFINE_int32(target_ttft,
             1000,
             "Target Time to First Token (TTFT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_ttft, brpc::NonNegativeInteger);

DEFINE_int32(target_tpot,
             50,
             "Target Time Per Output Token (TPOT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_tpot, brpc::NonNegativeInteger);

DEFINE_bool(enable_prefill_only_mode,
            false,
            "When true, all forwarded requests have max_tokens overridden to 1 "
            "(prefill-only elastic pool mode).");

DEFINE_double(gpu_hbm_per_gpu_gb,
              80.0,
              "HBM capacity per GPU in GB, used for auto-scaling resource model.");

DEFINE_double(gpu_compute_sm_per_gpu,
              1.0,
              "Compute SM units per GPU, used for auto-scaling resource model.");

// Prism baseline flags
DEFINE_string(baseline_type,
              "",
              "Baseline scheduling type. Empty = default dual-pool, "
              "\"PRISM\" = Prism multi-model scheduling.");

DEFINE_double(prism_schedule_interval_s,
              5.0,
              "Prism global scheduling interval in seconds.");

DEFINE_double(prism_memory_pool_budget_gb,
              6.0,
              "Prism KV cache memory pool budget per model in GB.");

DEFINE_double(prism_idle_threshold_s,
              50.0,
              "Prism idle model eviction threshold in seconds.");

DEFINE_string(prism_migrate_policy,
              "memory_per_request",
              "Prism migration policy: \"memory_per_request\" or \"violation\".");

DEFINE_int32(prism_max_models_per_instance,
             4,
             "Prism maximum number of models colocated on one instance.");

// ServerlessLLM baseline flags
DEFINE_double(sllm_schedule_interval_s,
              1.0,
              "ServerlessLLM global scheduling interval in seconds.");

DEFINE_double(sllm_idle_threshold_s,
              60.0,
              "ServerlessLLM LRU idle model eviction threshold in seconds.");

DEFINE_double(sllm_d2d_speed_gbps,
              25.0,
              "ServerlessLLM D2D transfer speed in GB/s (Tier 1).");

DEFINE_double(sllm_h2d_speed_gbps,
              6.0,
              "ServerlessLLM H2D transfer speed in GB/s (Tier 2).");

DEFINE_double(sllm_drain_alpha,
              0.001,
              "ServerlessLLM drain time linear coefficient (alpha * tokens).");

DEFINE_double(sllm_drain_beta,
              0.5,
              "ServerlessLLM drain time constant (seconds).");

DEFINE_int32(sllm_target_ongoing_requests,
             8,
             "ServerlessLLM target concurrent requests per instance for auto-scaling.");

DEFINE_int32(sllm_min_instances,
             0,
             "ServerlessLLM minimum instances per model.");

DEFINE_int32(sllm_max_instances,
             8,
             "ServerlessLLM maximum instances per model.");

DEFINE_bool(sllm_enable_knapsack,
            true,
            "ServerlessLLM enable 0/1 knapsack DP eviction.");

DEFINE_int32(sllm_max_models_per_instance,
             4,
             "ServerlessLLM maximum number of models colocated on one instance.");

// Llumnix baseline flags
DEFINE_double(llumnix_schedule_interval_s,
              1.0,
              "Llumnix global scheduling interval in seconds.");

DEFINE_double(llumnix_idle_threshold_s,
              60.0,
              "Llumnix idle model eviction threshold in seconds.");

DEFINE_double(llumnix_migrate_out_load_threshold,
              0.8,
              "Llumnix load threshold above which an instance is a migration source.");

DEFINE_int32(llumnix_topk_random_dispatch,
             1,
             "Llumnix top-K for random selection in load-based dispatch.");

DEFINE_int32(llumnix_max_models_per_instance,
             4,
             "Llumnix maximum number of models colocated on one instance.");

DEFINE_int32(llumnix_min_instances,
             0,
             "Llumnix minimum instances per model.");

DEFINE_int32(llumnix_max_instances,
             8,
             "Llumnix maximum instances per model.");

DEFINE_string(llumnix_dispatch_load_metric,
              "kv_blocks_ratio",
              "Llumnix dispatch load metric: kv_blocks_ratio or remaining_steps.");

DEFINE_string(llumnix_migration_load_metric,
              "kv_blocks_ratio",
              "Llumnix migration load metric: kv_blocks_ratio or remaining_steps.");

DEFINE_string(llumnix_dispatch_policy,
              "load",
              "Llumnix dispatch policy: load, balanced, queue, or rr.");

DEFINE_string(llumnix_migration_policy,
              "balanced",
              "Llumnix migration policy: balanced or defrag.");