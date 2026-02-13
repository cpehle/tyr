#!/usr/bin/env bash
set -euo pipefail
exec ./scripts/gpu/run_e2e_kernel.sh Tyr.GPU.Kernels.MhaH100 RunMhaH100 mha_h100
