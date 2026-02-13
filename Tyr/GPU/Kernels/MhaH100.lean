/- ThunderKittens-style FlashAttention forward kernels for 128x64 (2 KV blocks). -/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- FlashAttention forward for two KV blocks (seq=128, head_dim=64).
    This kernel is currently non-causal because dynamic block-offset masking is
    not yet represented in the IR. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd2Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125 -- 1 / sqrt(64)

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- FlashAttention forward (2 KV blocks) with LSE output.
    `lse_ptr` stores one 64-element vector per query tile row. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd2BlockLse
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125 -- 1 / sqrt(64)

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lseShared ← allocSV .Float32 tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let lse ← computeLSE softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lseShared lse
  storeVecGlobalRow lse_ptr lseShared coord

/-- `mha_h100`-style forward:
    - output `o_ptr` (bf16)
    - output `l_ptr` where `l = -8 * lse` for `head_dim=64` (ThunderKittens convention) -/
@[gpu_kernel .SM90]
def tkMhaH100Fwd2Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let lScale : Float := -8.0

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lShared ← allocSV .Float32 tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let l ← computeLSE softmaxState
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lShared l
  storeVecGlobalRow l_ptr lShared coord

/-- `mha_h100` backward prep:
    `d_ptr[row] = sum_j(dO[row,j] * O[row,j])`. -/
@[gpu_kernel .SM90]
def tkMhaH100BwdPrep2Block
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (d_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let coord ← blockCoord2D

  let dO ← allocRT .BFloat16 tileSize tileSize
  let o ← allocRT .BFloat16 tileSize tileSize
  let dOf ← allocRT .Float32 tileSize tileSize
  let of ← allocRT .Float32 tileSize tileSize
  let prod ← allocRT .Float32 tileSize tileSize
  let dVec ← allocRV .Float32 tileSize

  let dOShared ← allocST .BFloat16 tileSize tileSize
  let oShared ← allocST .BFloat16 tileSize tileSize
  let dShared ← allocSV .Float32 tileSize

  loadGlobal dOShared dO_ptr coord
  loadGlobal oShared o_ptr coord
  sync
  load dO dOShared
  load o oShared

  convert dOf dO
  convert of o
  mul prod dOf of
  rowSum dVec prod

  storeVec dShared dVec
  storeVecGlobalRow d_ptr dShared coord

/-- `mha_h100` backward (2 blocks, non-causal) with partial `dK`/`dV` outputs.
    `dQ` is final.
    `dK_part_ptr`/`dV_part_ptr` store one 64x64 tile per `(kv_tile, q_tile)` pair.
    The host test reduces partials across `q_tile` to recover full `dK`/`dV`. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockPartials
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 2
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let dO ← allocRT .BFloat16 tileSize tileSize
  let dQ ← zeroRT .Float32 tileSize tileSize
  let lTk ← allocRV .Float32 tileSize
  let lse ← allocRV .Float32 tileSize
  let dVec ← allocRV .Float32 tileSize

  -- Reuse shared buffers aggressively to stay within SM90 shared-memory limits.
  let rowShared ← allocST .BFloat16 tileSize tileSize
  let colShared ← allocST .BFloat16 tileSize tileSize .Col
  let outShared ← allocST .Float32 tileSize tileSize
  let vecShared ← allocSV .Float32 tileSize

  loadGlobal rowShared q_ptr coord
  sync
  load q rowShared
  loadGlobal rowShared dO_ptr coord
  sync
  load dO rowShared

  loadVecGlobalRow vecShared l_ptr coord
  sync
  loadVec lTk vecShared
  loadVecGlobalRow vecShared d_ptr coord
  sync
  loadVec dVec vecShared
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize
    let dKPart ← zeroRT .Float32 tileSize tileSize
    let dVPart ← zeroRT .Float32 tileSize tileSize

    loadGlobal rowShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal colShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k rowShared
    load v colShared

    -- Reference-style orientation: keep score/probability blocks in KxQ order.
    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let pTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert pTBf16 pT
    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dOCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dVPart pTBf16 dOCol dVPart

    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := (coord.withRow kvIdx.id).withCol coord.r
    store outShared dKPart
    sync
    storeGlobal dK_part_ptr outShared dkvCoord
    store outShared dVPart
    sync
    storeGlobal dV_part_ptr outShared dkvCoord
    sync

  store outShared dQ
  sync
  storeGlobal dQ_ptr outShared coord

/-- FlashAttention forward for 12 KV blocks (seq=768, head_dim=64). -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd12Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125 -- 1 / sqrt(64)

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

/-- FlashAttention forward (12 KV blocks) with LSE output. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd12BlockLse
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125 -- 1 / sqrt(64)

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lseShared ← allocSV .Float32 tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let lse ← computeLSE softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lseShared lse
  storeVecGlobalRow lse_ptr lseShared coord

/-- `mha_h100`-style forward for 12 KV blocks (`seq=768`, `d=64`).
    Output convention matches ThunderKittens: `l = -8 * lse`. -/
@[gpu_kernel .SM90]
def tkMhaH100Fwd12Block
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125
  let lScale : Float := -8.0

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize
  let lShared ← allocSV .Float32 tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState
  let l ← computeLSE softmaxState
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lShared l
  storeVecGlobalRow l_ptr lShared coord

/-- `mha_h100` backward for 12 KV blocks (`seq=768`, non-causal) with partial
    `dK`/`dV` outputs accumulated across query tiles on host. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockPartials
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 12
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let dO ← allocRT .BFloat16 tileSize tileSize
  let dQ ← zeroRT .Float32 tileSize tileSize
  let lTk ← allocRV .Float32 tileSize
  let lse ← allocRV .Float32 tileSize
  let dVec ← allocRV .Float32 tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let colShared ← allocST .BFloat16 tileSize tileSize .Col
  let outShared ← allocST .Float32 tileSize tileSize
  let vecShared ← allocSV .Float32 tileSize

  loadGlobal rowShared q_ptr coord
  sync
  load q rowShared
  loadGlobal rowShared dO_ptr coord
  sync
  load dO rowShared

  loadVecGlobalRow vecShared l_ptr coord
  sync
  loadVec lTk vecShared
  loadVecGlobalRow vecShared d_ptr coord
  sync
  loadVec dVec vecShared
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize
    let dKPart ← zeroRT .Float32 tileSize tileSize
    let dVPart ← zeroRT .Float32 tileSize tileSize

    loadGlobal rowShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal colShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k rowShared
    load v colShared

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let pTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert pTBf16 pT
    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dOCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dVPart pTBf16 dOCol dVPart

    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := (coord.withRow kvIdx.id).withCol coord.r
    store outShared dKPart
    sync
    storeGlobal dK_part_ptr outShared dkvCoord
    store outShared dVPart
    sync
    storeGlobal dV_part_ptr outShared dkvCoord
    sync

  store outShared dQ
  sync
  storeGlobal dQ_ptr outShared coord

end Tyr.GPU.Kernels
