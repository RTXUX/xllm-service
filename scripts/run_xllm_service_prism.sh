#!/bin/bash

set -euxo pipefail

args=(
    "--baseline_type=PRISM"
    "--prism_schedule_interval_s=5.0"
    "--prism_memory_pool_budget_gb=6.0"
    "--prism_idle_threshold_s=50.0"
    "--prism_migrate_policy=memory_per_request"
    "--prism_max_models_per_instance=2"
    "--enable_prefill_only_mode=true"
    "--etcd_addr=127.0.0.1:32389"
    "--http_server_port=37888"
    "--rpc_server_port=37889"
    "--tokenizer_path=/export/home/models/Qwen2-7B"
)

ENABLE_DECODE_RESPONSE_TO_SERVICE=true \
    exec ./build/xllm_service/xllm_master_serving \
    "${args[@]}"