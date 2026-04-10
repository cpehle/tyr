#!/bin/bash
# Load required modules for ThunderKittens development
# Uses GCCcore/12.3.0 to match locally compiled Lean 4

if ! type module >/dev/null 2>&1; then
  echo "Environment modules unavailable; assuming dependencies are already on PATH." >&2
  return 0 2>/dev/null || exit 0
fi

module purge
module load EB5
module load EB5Modules
module load EBModules
module load Arrow/14.0.1-gfbf-2023a
module load CUDA/12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

echo "Modules loaded:"
module list
