#!/usr/bin/env bash
# ci-e2e-omni GPU smoke tests (2-GPU): end-to-end omni training paths.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "ci-e2e-omni" 2 "$@"

omni_trainer_args=()
if [[ -n "${RAY_MASTER_PORT_RANGE:-}" ]]; then
    omni_trainer_args+=("+trainer.ray_master_port_range=[${RAY_MASTER_PORT_RANGE}]")
fi

# Fixed at 2 GPUs: FSDP/FSPD2 needs >1 GPU to shard (NO_SHARD can't run the
# offload_to_cpu LoRA-sync summon).
run_test 0 "Qwen3-Omni Thinker GSPO LoRA e2e (V1)" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS=2 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh

run_test 1 "Qwen3-Omni multimodal offline MLLM DPO LoRA e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS=2 \
    bash tests/special_e2e/run_qwen3_omni_multimodal_offline_mllm_dpo_lora_smoke.sh "${omni_trainer_args[@]}"

run_test 2 "Qwen3-Omni Thinker VeOmni EP=2 image training and export" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    torchrun --standalone --nproc_per_node=2 tests/special_e2e/check_qwen3_omni_veomni_backend.py

run_test 3 "Qwen3-Omni Thinker VeOmni GSPO full-weight e2e (V1)" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS=2 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_veomni_smoke.sh

gpu_smoke_summary
