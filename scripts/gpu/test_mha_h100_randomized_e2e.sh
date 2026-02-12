#!/usr/bin/env bash
set -euo pipefail

trials="${1:-10}"
export E2E_TRIALS="$trials"

exec ./scripts/gpu/run_e2e_kernel.sh \
  Tyr.GPU.Kernels.ThunderKittensFlashAttn \
  RunThunderKittensMhaH100 \
  mha_h100
