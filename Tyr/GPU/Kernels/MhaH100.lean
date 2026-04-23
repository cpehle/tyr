/- ThunderKittens-style FlashAttention forward kernels for 128x64 (2 KV blocks). -/
import Tyr.GPU.Codegen.Macros

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def asyncLoadGlobalTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : RTileCoord)
    (sem : Semaphore) : KernelM Unit := do
  initSemaphore sem 0 1
  blockSync
  expectBytes sem (rows * cols * dtype.bytes)
  loadGlobalAsync dst src coord sem.id
  waitSemaphore sem

private def asyncLoadGlobalPair {dtype : GpuFloat} {rows cols : Nat}
    {layoutA layoutB : TileLayout}
    (dstA : ST dtype rows cols layoutA)
    (srcA : GPtr dtype)
    (coordA : RTileCoord)
    (dstB : ST dtype rows cols layoutB)
    (srcB : GPtr dtype)
    (coordB : RTileCoord)
    (sem : Semaphore) : KernelM Unit := do
  initSemaphore sem 0 1
  blockSync
  expectBytes sem (2 * rows * cols * dtype.bytes)
  loadGlobalAsync dstA srcA coordA sem.id
  loadGlobalAsync dstB srcB coordB sem.id
  waitSemaphore sem

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
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
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
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
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
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
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
  let doSem ← allocSemaphore

  asyncLoadGlobalPair dOShared dO_ptr coord oShared o_ptr coord doSem
  load dO dOShared
  load o oShared

  convert dOf dO
  convert of o
  mul prod dOf of
  rowSum dVec prod

  storeVec dShared dVec
  storeVecGlobalRow d_ptr dShared coord

/-- `mha_h100` backward (2 blocks, non-causal) with in-kernel `dK`/`dV`
    accumulation. The launcher name keeps the older `...Partials` suffix for
    ABI stability, but `dK_part_ptr` and `dV_part_ptr` now point at final
    zero-initialized `[1, 1, seq, 64]` buffers. Each query tile contributes to
    the KV tile with TMA store-add, matching ThunderKittens' `kv_store`
    accumulation contract more closely than the earlier external partial stack. -/
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

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  -- Reuse one FP32 staging tile; wait before overwriting it with the next TMA store.
  let dKVShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
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

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    -- Reference-style orientation: keep score/probability blocks in KxQ order.
    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

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

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := coord.withRow kvIdx.id
    store dKVShared dKPart
    groupSync 4 4
    storeGlobalAdd dK_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4
    store dKVShared dVPart
    groupSync 4 4
    storeGlobalAdd dV_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4

  store dKVShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dKVShared coord

/-- Q-centric `dQ` pass for 2-block H100 MHA backward.
    This keeps the stable direct-store `dQ` accumulation while the K/V gradients
    are produced by the KV-centric sweep below. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockDq
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
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

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let dQShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

  store dQShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dQShared coord

/-- KV-centric `mha_h100` backward sweep for 2 blocks. This mirrors
    ThunderKittens' backward ownership more closely than the q-centric
    `...Partials` compatibility kernel: one CTA owns a KV tile, sweeps all query
    tiles, accumulates `dK`/`dV` in registers, then store-adds each once. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockKvSweep
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (_dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numQBlocks : Nat := 2
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let kvCoord ← blockCoord2D

  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let dKAccum ← zeroRT .Float32 tileSize tileSize
  let dVAccum ← zeroRT .Float32 tileSize tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let dKVShared ← allocST .Float32 tileSize tileSize
  let kvSem ← allocSemaphore
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore

  asyncLoadGlobalPair rowShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
  load k rowShared
  load v vShared

  let vRow ← allocRT .BFloat16 tileSize tileSize
  swapLayout vRow v

  for qIdx in krange 0 numQBlocks do
    let qCoord := kvCoord.withRow qIdx.id
    let q ← allocRT .BFloat16 tileSize tileSize
    let dO ← allocRT .BFloat16 tileSize tileSize
    let lTk ← allocRV .Float32 tileSize
    let lse ← allocRV .Float32 tileSize
    let dVec ← allocRV .Float32 tileSize

    asyncLoadGlobalTile rowShared q_ptr qCoord qSem
    load q rowShared
    asyncLoadGlobalTile rowShared dO_ptr qCoord doSem
    load dO rowShared

    loadVecGlobalRowRV lTk l_ptr qCoord
    loadVecGlobalRowRV dVec d_ptr qCoord
    scalarMulVec lse lTk invLScale

    -- Keep score/probability blocks in KxQ order, matching TK's bwd loop.
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

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
    mma dVAccum pTBf16 dOCol dVAccum

    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKAccum dSTBf16 qCol dKAccum

  store dKVShared dKAccum
  groupSync 4 4
  storeGlobalAdd dK_ptr dKVShared kvCoord
  tmaStoreCommitGroup
  tmaStoreAsyncWait
  groupSync 4 4
  store dKVShared dVAccum
  groupSync 4 4
  storeGlobalAdd dV_ptr dKVShared kvCoord
  tmaStoreCommitGroup
  tmaStoreAsyncWait

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
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
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
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
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
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile qShared q_ptr coord qSem
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    asyncLoadGlobalPair kShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    warpgroupMmT s q kShared
    mmaAsyncWait
    scalarMul s s scale
    onlineSoftmax s o softmaxState
    convert p s
    warpgroupMma o p vShared
    mmaAsyncWait
  finalizeSoftmax o softmaxState
  let l ← computeLSE softmaxState
  scalarMulVec l l lScale

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  storeVec lShared l
  storeVecGlobalRow l_ptr lShared coord

/-- `mha_h100` backward for 12 KV blocks (`seq=768`, non-causal) with in-kernel
    `dK`/`dV` accumulation. As above, the function name keeps the older
    `...Partials` suffix for ABI stability while callers pass final
    zero-initialized `[1, 1, seq, 64]` gradient buffers. -/
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
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  -- Reuse one FP32 staging tile; wait before overwriting it with the next TMA store.
  let dKVShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
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

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

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

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKPart dSTBf16 qCol dKPart
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := coord.withRow kvIdx.id
    store dKVShared dKPart
    groupSync 4 4
    storeGlobalAdd dK_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4
    store dKVShared dVPart
    groupSync 4 4
    storeGlobalAdd dV_part_ptr dKVShared dkvCoord
    tmaStoreCommitGroup
    tmaStoreAsyncWait
    groupSync 4 4

  store dKVShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dKVShared coord

/-- Q-centric `dQ` pass for 12-block H100 MHA backward (`seq=768`, `d=64`). -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockDq
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
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
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let dQShared ← allocST .Float32 tileSize tileSize
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadGlobalTile rowShared q_ptr coord qSem
  load q rowShared
  asyncLoadGlobalTile rowShared dO_ptr coord doSem
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT .BFloat16 tileSize tileSize
    let v ← allocRT .BFloat16 tileSize tileSize .Col
    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize

    asyncLoadGlobalPair rowShared k_ptr (coord.withRow kvIdx.id) vShared v_ptr (coord.withRow kvIdx.id) kvSem
    load k rowShared
    load v vShared
    let vRow ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT .Float32 tileSize tileSize
    scalarMul dSTScaled dST scale

    let dSTBf16 ← allocRT .BFloat16 tileSize tileSize
    convert dSTBf16 dSTScaled

    let dSRow ← allocRT .BFloat16 tileSize tileSize
    transpose dSRow dSTBf16
    let kCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

  store dQShared dQ
  groupSync 4 4
  storeGlobal dQ_ptr dQShared coord

/-- KV-centric `mha_h100` backward sweep for 12 blocks (`seq=768`, `d=64`).
    Each CTA owns one KV tile and sweeps all query tiles, matching the
    ThunderKittens K/V reduction direction. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockKvSweep
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (_dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numQBlocks : Nat := 12
  let scale : Float := 0.125
  let invLScale : Float := -0.125

  let kvCoord ← blockCoord2D

  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let dKAccum ← zeroRT .Float32 tileSize tileSize
  let dVAccum ← zeroRT .Float32 tileSize tileSize

  let rowShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let dKVShared ← allocST .Float32 tileSize tileSize
  let kvSem ← allocSemaphore
  let qSem ← allocSemaphore
  let doSem ← allocSemaphore

  asyncLoadGlobalPair rowShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
  load k rowShared
  load v vShared

  let vRow ← allocRT .BFloat16 tileSize tileSize
  swapLayout vRow v

  for qIdx in krange 0 numQBlocks do
    let qCoord := kvCoord.withRow qIdx.id
    let q ← allocRT .BFloat16 tileSize tileSize
    let dO ← allocRT .BFloat16 tileSize tileSize
    let lTk ← allocRV .Float32 tileSize
    let lse ← allocRV .Float32 tileSize
    let dVec ← allocRV .Float32 tileSize

    asyncLoadGlobalTile rowShared q_ptr qCoord qSem
    load q rowShared
    asyncLoadGlobalTile rowShared dO_ptr qCoord doSem
    load dO rowShared

    loadVecGlobalRowRV lTk l_ptr qCoord
    loadVecGlobalRowRV dVec d_ptr qCoord
    scalarMulVec lse lTk invLScale

    let sT ← zeroRT .Float32 tileSize tileSize
    let pT ← allocRT .Float32 tileSize tileSize
    let dPT ← zeroRT .Float32 tileSize tileSize
    let dST ← allocRT .Float32 tileSize tileSize

    mmaT sT k q sT
    scalarMul sT sT scale
    subRow sT sT lse
    exp pT sT

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
    mma dVAccum pTBf16 dOCol dVAccum

    let qCol ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dKAccum dSTBf16 qCol dKAccum

  store dKVShared dKAccum
  groupSync 4 4
  storeGlobalAdd dK_ptr dKVShared kvCoord
  tmaStoreCommitGroup
  tmaStoreAsyncWait
  groupSync 4 4
  store dKVShared dVAccum
  groupSync 4 4
  storeGlobalAdd dV_ptr dKVShared kvCoord
  tmaStoreCommitGroup
  tmaStoreAsyncWait

end Tyr.GPU.Kernels
