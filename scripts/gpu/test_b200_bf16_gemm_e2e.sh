#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  case "${gpu_name}" in
    *GB10*|*B200*|*B300*) ;;
    *)
      echo "[skip] b200_bf16_gemm: requires a Blackwell-family GPU (saw ${gpu_name})"
      exit 0
      ;;
  esac
fi

export TYR_GPU_FAMILY=BLACKWELL
exec ./scripts/gpu/run_e2e_kernel.sh Tyr.GPU.Kernels.Bf16Gemm RunB200Bf16Gemm b200_bf16_gemm "$@"
