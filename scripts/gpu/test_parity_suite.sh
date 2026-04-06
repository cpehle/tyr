#!/usr/bin/env bash
set -euo pipefail

randomized_trials="${RANDOMIZED_MHA_TRIALS:-0}"

./scripts/gpu/test_leantest_e2e.sh
./scripts/gpu/test_b200_bf16_gemm_e2e.sh
./scripts/gpu/test_mha_h100_768_e2e.sh

if [[ "${randomized_trials}" =~ ^[0-9]+$ ]] && [[ "${randomized_trials}" -gt 0 ]]; then
  ./scripts/gpu/test_mha_h100_randomized_e2e.sh "${randomized_trials}"
fi
