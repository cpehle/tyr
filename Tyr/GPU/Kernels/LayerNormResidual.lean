/-
  Tyr/GPU/Kernels/LayerNormResidual.lean

  LayerNorm + residual kernel aligned with ThunderKittens layernorm semantics:
  - out_resid = x + residual
  - out = layer_norm(out_resid, weight, bias)
  The kernel processes one [64 x 1024] tile per block row.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Attribute

/-!
# `Tyr.GPU.Kernels.LayerNormResidual`

GPU kernel module implementing Layer Norm Residual primitives for accelerated model workloads.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

@[gpu_kernel .SM90]
def tkLayerNorm
    (x_ptr : GPtr GpuFloat.Float32)
    (residual_ptr : GPtr GpuFloat.Float32)
    (weight_ptr : GPtr GpuFloat.Float32)
    (bias_ptr : GPtr GpuFloat.Float32)
    (out_ptr : GPtr GpuFloat.Float32)
    (out_resid_ptr : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "ThunderKittens-style layernorm + residual on 64x1024 tiles"

  let hiddenTiles : Nat := 16
  let invHidden : Float := 0.0009765625 -- 1 / 1024
  let eps : Float := 1.0e-5

  let coord ← blockCoord2D

  let x ← allocRT .Float32 64 64
  let residual ← allocRT .Float32 64 64
  let outResid ← allocRT .Float32 64 64
  let centered ← allocRT .Float32 64 64
  let sq ← allocRT .Float32 64 64

  -- Reuse one shared tile buffer to keep shared memory under SM90 limits.
  let tileShared ← allocST .Float32 64 64

  let sum ← zeroRV .Float32 64
  let sumSq ← zeroRV .Float32 64
  let tmpSum ← allocRV .Float32 64
  let tmpSqSum ← allocRV .Float32 64
  let mean ← allocRV .Float32 64
  let meanSq ← allocRV .Float32 64
  let var ← allocRV .Float32 64
  let invStd ← allocRV .Float32 64

  let weightShared ← allocSV .Float32 64
  let biasShared ← allocSV .Float32 64
  let weightRV ← allocRV .Float32 64
  let biasRV ← allocRV .Float32 64

  comment "Pass 1: out_resid and row statistics across full hidden dimension"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared x_ptr tileCoord
    sync
    load x tileShared

    loadGlobal tileShared residual_ptr tileCoord
    sync
    load residual tileShared

    add outResid x residual
    store tileShared outResid
    sync
    storeGlobal out_resid_ptr tileShared tileCoord

    rowSum tmpSum outResid
    addVec sum sum tmpSum
    mul sq outResid outResid
    rowSum tmpSqSum sq
    addVec sumSq sumSq tmpSqSum

  scalarMulVec mean sum invHidden
  scalarMulVec var sumSq invHidden
  mulVec meanSq mean mean
  subVec var var meanSq
  scalarAddVec var var eps
  rsqrtVec invStd var

  comment "Pass 2: normalize + affine (weight, bias)"
  for hiddenIdx in krange 0 hiddenTiles do
    let tileCoord := coord.withCol hiddenIdx.id

    loadGlobal tileShared out_resid_ptr tileCoord
    sync
    load outResid tileShared

    -- `weight_ptr`/`bias_ptr` are 1D tensors (len=1024); load one 64-element slice per tile.
    loadVecGlobalCol weightShared weight_ptr tileCoord
    loadVec weightRV weightShared

    loadVecGlobalCol biasShared bias_ptr tileCoord
    loadVec biasRV biasShared

    subCol centered outResid mean
    mulCol centered centered invStd
    mulRow centered centered weightRV
    addRow centered centered biasRV

    store tileShared centered
    sync
    storeGlobal out_ptr tileShared tileCoord

end Tyr.GPU.Kernels
