#!/bin/bash

set -euxo pipefail

args=(
    "--baseline_type=BLITZSCALE"
    "--blitzscale_tokens_prefilled_per_sec=5000"
    "--blitzscale_tokens_transferred_per_sec=5000"
    "--blitzscale_min_prefill_instances=1"
    "--blitzscale_max_prefill_instances=2"
    "--blitzscale_min_decode_instances=0"
    "--blitzscale_max_decode_instances=0"
    "--enable_prefill_only_mode=true"
    "--etcd_addr=127.0.0.1:32389"
    "--http_server_port=37888"
    "--rpc_server_port=37889"
    "--tokenizer_path=/export/home/models/Qwen3-8B"
    "--enable_d2d=false"
)

ENABLE_DECODE_RESPONSE_TO_SERVICE=true \
    ./build/xllm_service/xllm_master_serving \
    "${args[@]}"
