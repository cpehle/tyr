/- Rotary embedding forward kernel (ThunderKittens-style tile program). -/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Attribute

/-!
# `Tyr.GPU.Kernels.Rotary`

GPU kernel module implementing Rotary primitives for accelerated model workloads.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Kernels.Rotary

open Tyr.GPU
open Tyr.GPU.Codegen

/-- RoPE forward on one 64x64 tile.
    Input is split as [x1 | x2] over columns, rotated with sin/cos, and concatenated. -/
@[gpu_kernel .SM90]
def rotaryFwd (x_ptr : GPtr GpuFloat.Float32) (sin_ptr : GPtr GpuFloat.Float32)
    (cos_ptr : GPtr GpuFloat.Float32) (out_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64) (_head_dim : KVal UInt64) : KernelM Unit := do
  let coord ← blockCoord2D

  let x ← allocRT .Float32 64 64
  let x1 ← allocRT .Float32 64 32
  let x2 ← allocRT .Float32 64 32
  let sinT ← allocRT .Float32 64 32
  let cosT ← allocRT .Float32 64 32
  let y1 ← allocRT .Float32 64 32
  let y2 ← allocRT .Float32 64 32
  let tmp ← allocRT .Float32 64 32
  let negX2 ← allocRT .Float32 64 32
  let out ← allocRT .Float32 64 64

  let xShared ← allocST .Float32 64 64
  let sinShared ← allocST .Float32 64 32
  let cosShared ← allocST .Float32 64 32
  let outShared ← allocST .Float32 64 64

  loadGlobal xShared x_ptr coord
  loadGlobal sinShared sin_ptr coord
  loadGlobal cosShared cos_ptr coord
  sync

  load x xShared
  load sinT sinShared
  load cosT cosShared

  sliceCols x1 x 0 32
  sliceCols x2 x 32 32

  -- y1 = x1*cos - x2*sin
  mul y1 x1 cosT
  neg negX2 x2
  mul negX2 negX2 sinT
  add y1 y1 negX2

  -- y2 = x2*cos + x1*sin
  mul y2 x2 cosT
  mul tmp x1 sinT
  add y2 y2 tmp

  concatCols out y1 y2

  store outShared out
  sync
  storeGlobal out_ptr outShared coord

#check rotaryFwd.kernel
#check rotaryFwd.launch

end Tyr.GPU.Kernels.Rotary
