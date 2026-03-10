/-
  Tyr/GPU/Kernels/RingAttn.lean

  Ring attention surfaces split into the same coarse phases the vendored
  ThunderKittens kernel exposes:

  1. partial attention against the current KV ring buffer,
  2. communication to advance K/V into the next ping-pong buffer,
  3. numerically stable reduction of running O/L statistics.

  The Lean DSL still cannot encode the exact multi-launch schedule or peer index
  arithmetic from `ring_attn_h100.cu`, so the kernels below treat the caller's
  pointers as already-partitioned current/next views.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

namespace Tyr.GPU.Kernels.RingAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Ring phase 1: compute attention for the current K/V shard and emit a partial
output tile plus its per-row log-sum-exp summary. -/
@[gpu_kernel .SM90]
def ringAttnPartial (Q_ptr : GPtr GpuFloat.BFloat16) (K_curr_ptr : GPtr GpuFloat.BFloat16)
    (V_curr_ptr : GPtr GpuFloat.BFloat16) (O_partial_ptr : GPtr GpuFloat.Float32)
    (L_partial_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (seq_len, head_dim, ring_stage, dev_idx, world_size)
  let qoBlock : Nat := 64
  let kvBlock : Nat := 128
  let d : Nat := 128
  let coord ← blockCoord2D

  let q : RT GpuFloat.BFloat16 qoBlock d ← allocRT .BFloat16 qoBlock d
  let k : RT GpuFloat.BFloat16 kvBlock d ← allocRT .BFloat16 kvBlock d
  let v : RT GpuFloat.BFloat16 kvBlock d .Col ← allocRT .BFloat16 kvBlock d .Col
  let scores : RT GpuFloat.Float32 qoBlock kvBlock ← allocRT .Float32 qoBlock kvBlock
  let probBf : RT GpuFloat.BFloat16 qoBlock kvBlock ← allocRT .BFloat16 qoBlock kvBlock
  let oPartial : RT GpuFloat.Float32 qoBlock d ← zeroRT .Float32 qoBlock d

  let rowMaxVec : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let rowSumVec : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lseVec : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock

  let qShared : ST GpuFloat.BFloat16 qoBlock d ← allocST .BFloat16 qoBlock d
  let kShared : ST GpuFloat.BFloat16 kvBlock d ← allocST .BFloat16 kvBlock d
  let vShared : ST GpuFloat.BFloat16 kvBlock d .Col ← allocST .BFloat16 kvBlock d .Col
  let outShared : ST GpuFloat.Float32 qoBlock d ← allocST .Float32 qoBlock d
  let lseShared : SV GpuFloat.Float32 qoBlock ← allocSV .Float32 qoBlock

  comment "Ring partial phase: stationary Q, current-ring K/V"
  Support.asyncTileLoad qShared Q_ptr coord (qoBlock * d * 2)
  Support.asyncTileLoad kShared K_curr_ptr coord (kvBlock * d * 2)
  Support.asyncTileLoad vShared V_curr_ptr coord (kvBlock * d * 2)
  sync
  load q qShared
  load k kShared
  load v vShared

  let zeros : RT GpuFloat.Float32 qoBlock kvBlock ← zeroRT .Float32 qoBlock kvBlock
  mmaT scores q k zeros
  scalarMul scores scores 0.08838834764
  makeCausal scores scores (some (-1.0e10))

  rowMax rowMaxVec scores
  subCol scores scores rowMaxVec
  exp scores scores
  rowSum rowSumVec scores
  divCol scores scores rowSumVec

  convert probBf scores
  mma oPartial probBf v oPartial

  copyVec lseVec rowMaxVec
  logVec rowSumVec rowSumVec
  addVec lseVec lseVec rowSumVec

  store outShared oPartial
  storeGlobal O_partial_ptr outShared coord
  storeVec lseShared lseVec
  storeVecGlobalRow L_partial_ptr lseShared coord

/-- Ring phase 2: move the current K/V shard into the caller-provided "next"
buffers. This is the explicit communication stage the old monolithic kernel hid. -/
@[gpu_kernel .SM90]
def ringAttnComm (K_next_ptr : GPtr GpuFloat.BFloat16) (V_next_ptr : GPtr GpuFloat.BFloat16)
    (K_curr_ptr : GPtr GpuFloat.BFloat16) (V_curr_ptr : GPtr GpuFloat.BFloat16)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (ring_stage, dev_idx, world_size)
  let kvBlock : Nat := 128
  let d : Nat := 128
  let coord ← blockCoord2D

  let kCurr : RT GpuFloat.BFloat16 kvBlock d ← allocRT .BFloat16 kvBlock d
  let vCurr : RT GpuFloat.BFloat16 kvBlock d ← allocRT .BFloat16 kvBlock d
  let kShared : ST GpuFloat.BFloat16 kvBlock d ← allocST .BFloat16 kvBlock d
  let vShared : ST GpuFloat.BFloat16 kvBlock d ← allocST .BFloat16 kvBlock d
  let kExchange : ST GpuFloat.BFloat16 kvBlock d ← allocST .BFloat16 kvBlock d
  let vExchange : ST GpuFloat.BFloat16 kvBlock d ← allocST .BFloat16 kvBlock d

  comment "Ring communication phase: current K/V -> next ping-pong buffer"
  Support.asyncTileLoad kShared K_curr_ptr coord (kvBlock * d * 2)
  Support.asyncTileLoad vShared V_curr_ptr coord (kvBlock * d * 2)
  load kCurr kShared
  load vCurr vShared
  multimemStore kExchange kCurr
  multimemStore vExchange vCurr
  Support.barrierAllDevices "ring K/V transfer complete" 0
  storeGlobalAsync K_next_ptr kExchange coord
  storeGlobalAsync V_next_ptr vExchange coord
  sync
  Support.barrierAllDevices "ring K/V epilogue" 1

/-- Ring phase 3: merge the running `(O, L)` state with the current partial
contribution using the standard log-sum-exp reduction. -/
@[gpu_kernel .SM90]
def ringAttnReduce (O_running_ptr : GPtr GpuFloat.Float32) (O_block_ptr : GPtr GpuFloat.Float32)
    (L_running_ptr : GPtr GpuFloat.Float32) (L_block_ptr : GPtr GpuFloat.Float32)
    (O_out_ptr : GPtr GpuFloat.BFloat16) (L_out_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  let _ := (seq_len, head_dim)
  let qoBlock : Nat := 64
  let d : Nat := 128
  let coord ← blockCoord2D

  let oRunning : RT GpuFloat.Float32 qoBlock d ← allocRT .Float32 qoBlock d
  let oBlock : RT GpuFloat.Float32 qoBlock d ← allocRT .Float32 qoBlock d
  let oOut : RT GpuFloat.Float32 qoBlock d ← allocRT .Float32 qoBlock d
  let outBf : RT GpuFloat.BFloat16 qoBlock d ← allocRT .BFloat16 qoBlock d

  let lRunning : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lBlock : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lMax : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lScaleRunning : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lScaleBlock : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lDenom : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock
  let lOut : RV GpuFloat.Float32 qoBlock ← allocRV .Float32 qoBlock

  let oRunningShared : ST GpuFloat.Float32 qoBlock d ← allocST .Float32 qoBlock d
  let oBlockShared : ST GpuFloat.Float32 qoBlock d ← allocST .Float32 qoBlock d
  let oOutShared : ST GpuFloat.BFloat16 qoBlock d ← allocST .BFloat16 qoBlock d
  let lRunningShared : SV GpuFloat.Float32 qoBlock ← allocSV .Float32 qoBlock
  let lBlockShared : SV GpuFloat.Float32 qoBlock ← allocSV .Float32 qoBlock
  let lOutShared : SV GpuFloat.Float32 qoBlock ← allocSV .Float32 qoBlock

  comment "Ring reduction phase: stable merge of running and current partial tiles"
  loadGlobal oRunningShared O_running_ptr coord
  loadGlobal oBlockShared O_block_ptr coord
  loadVecGlobalRow lRunningShared L_running_ptr coord
  loadVecGlobalRow lBlockShared L_block_ptr coord
  sync
  load oRunning oRunningShared
  load oBlock oBlockShared
  loadVec lRunning lRunningShared
  loadVec lBlock lBlockShared

  Support.maxVec lMax lRunning lBlock
  subVec lScaleRunning lRunning lMax
  expVec lScaleRunning lScaleRunning
  subVec lScaleBlock lBlock lMax
  expVec lScaleBlock lScaleBlock

  mulCol oRunning oRunning lScaleRunning
  mulCol oBlock oBlock lScaleBlock
  add oOut oRunning oBlock

  addVec lDenom lScaleRunning lScaleBlock
  divCol oOut oOut lDenom
  logVec lDenom lDenom
  addVec lOut lMax lDenom

  convert outBf oOut
  store oOutShared outBf
  storeGlobal O_out_ptr oOutShared coord
  storeVec lOutShared lOut
  storeVecGlobalRow L_out_ptr lOutShared coord

/-- Compatibility shell for the previous monolithic forward entrypoint.
Callers should prefer the explicit `ringAttnPartial`, `ringAttnComm`, and
`ringAttnReduce` kernels above. -/
@[gpu_kernel .SM90]
def ringAttnFull (Q_ptr : GPtr GpuFloat.BFloat16) (K_curr_ptr : GPtr GpuFloat.BFloat16)
    (V_curr_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32)
    (batch_size : KVal UInt64) (num_heads : KVal UInt64)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    (ring_stage : KVal UInt32) (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (Q_ptr, K_curr_ptr, V_curr_ptr, O_ptr, L_ptr, batch_size, num_heads,
    seq_len, head_dim, ring_stage, dev_idx, world_size)
  comment "ringAttnFull is now an orchestration shell."
  comment "Launch order per stage:"
  comment "  1. ringAttnPartial on the current K/V shard"
  comment "  2. ringAttnComm to advance K/V around the ring"
  comment "  3. ringAttnReduce once a running O/L state already exists"
  Support.barrierAllDevices "ring orchestration handoff" 0

end Tyr.GPU.Kernels.RingAttn
