#!/usr/bin/env bash
set -euo pipefail

./scripts/gpu/run_e2e_kernel.sh \
  Tyr.GPU.Kernels.LayerNormResidual \
  RunLayerNorm \
  layernorm \
  Tyr.GPU.Codegen.EmitNew
