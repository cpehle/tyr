#!/usr/bin/env bash
set -euo pipefail

./scripts/gpu/run_e2e_kernel.sh \
  Tyr.GPU.Kernels.MhaH100 \
  RunFlashAttn \
  flashattn
