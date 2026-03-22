#!/bin/bash

set -euxo pipefail

args=(
    "--baseline_type=SERVERLESS_LLM"
    "--enable_prefill_only_mode=true"
    "--etcd_addr=127.0.0.1:32389"
    "--http_server_port=37888"
    "--rpc_server_port=37889"
    "--tokenizer_path=/export/home/models/Qwen2-7B"
    "--enable_d2d=false"
)

ENABLE_DECODE_RESPONSE_TO_SERVICE=true \
    exec ./build/xllm_service/xllm_master_serving \
    "${args[@]}"