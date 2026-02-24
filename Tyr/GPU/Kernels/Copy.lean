/-
  Tyr/GPU/Kernels/Copy.lean

  Minimal ThunderKittens-style copy kernel:
  copy one 64x64 tile from input to output using global -> shared -> register flow.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Attribute

/-!
# `Tyr.GPU.Kernels.Copy`

GPU kernel module implementing Copy primitives for accelerated model workloads.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

@[gpu_kernel .SM90]
def copy64x64 (input : GPtr GpuFloat.Float32) (output : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "ThunderKittens-style minimal copy kernel"
  let coord ← blockCoord2D
  let reg ← allocRT .Float32 64 64
  let smem ← allocST .Float32 64 64
  loadGlobal smem input coord
  load reg smem
  store smem reg
  storeGlobal output smem coord
  sync

abbrev tkCopy := copy64x64

end Tyr.GPU.Kernels
