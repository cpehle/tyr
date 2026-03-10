/-
  Tyr/GPU/Kernels/UlyssesAttnBwd.lean

  Ulysses backward is kept as a transport/orchestration shell. The real local
  attention backward math belongs to `FlashAttnBwd`; this module owns the
  all-to-all reshapes around that launch boundary. That means the backward
  surface is still partially speculative even though its communication phases
  are now concrete.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.UlyssesAttn

open Tyr.GPU
open Tyr.GPU.Codegen

private def asyncTileLoad {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout) (src : GPtr dtype) (coord : RTileCoord)
    (bytes : Nat) : KernelM Unit := do
  let sem ← allocSemaphore
  initSemaphore sem 1
  expectBytes sem bytes
  loadGlobalAsync dst src coord sem.id
  waitSemaphore sem

private def barrierAllDevices (label : String) (barrierId : Nat) : KernelM Unit := do
  comment s!"Cross-device barrier: {label}"
  arriveAndWait barrierId

private def allToAllTile (label : String) (output_ptr : GPtr GpuFloat.BFloat16)
    (input_ptr : GPtr GpuFloat.BFloat16) (coord : RTileCoord)
    (storeAdd : Bool := false) : KernelM Unit := do
  let tileRows : Nat := 16
  let tileCols : Nat := 128
  let shard : RT GpuFloat.BFloat16 tileRows tileCols ← allocRT .BFloat16 tileRows tileCols
  let inputShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols
  let exchangeShared : ST GpuFloat.BFloat16 tileRows tileCols ← allocST .BFloat16 tileRows tileCols

  comment s!"All-to-all transport for {label}"
  asyncTileLoad inputShared input_ptr coord (tileRows * tileCols * 2)
  load shard inputShared
  multimemStore exchangeShared shard
  barrierAllDevices s!"{label} exchange complete" 0
  if storeAdd then
    storeGlobalAdd output_ptr exchangeShared coord
  else
    storeGlobalAsync output_ptr exchangeShared coord
  sync

/-- Phase 1 of Ulysses backward: redistribute `dO` from head-parallel shards to
sequence-parallel shards before the local backward launch. -/
@[gpu_kernel .SM90]
def ulyssesDoAllToAllBwd (dO_seq_ptr : GPtr GpuFloat.BFloat16)
    (dO_heads_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  comment "Ulysses backward phase 1: head-parallel dO -> sequence-parallel dO"
  allToAllTile "dO heads->seq" dO_seq_ptr dO_heads_ptr coord
  barrierAllDevices "backward dO transport complete" 1

/-- Phase 3 of Ulysses backward: return `dQ/dK/dV` from sequence-parallel views
to head-parallel views. `storeGlobalAdd` is used to make the reduction boundary
explicit in the current DSL. -/
@[gpu_kernel .SM90]
def ulyssesGradReturnAllToAll (dQ_heads_ptr : GPtr GpuFloat.BFloat16)
    (dK_heads_ptr : GPtr GpuFloat.BFloat16) (dV_heads_ptr : GPtr GpuFloat.BFloat16)
    (dQ_seq_ptr : GPtr GpuFloat.BFloat16) (dK_seq_ptr : GPtr GpuFloat.BFloat16)
    (dV_seq_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  comment "Ulysses backward phase 3: sequence-parallel grads -> head-parallel grads"
  allToAllTile "dQ seq->heads" dQ_heads_ptr dQ_seq_ptr coord true
  allToAllTile "dK seq->heads" dK_heads_ptr dK_seq_ptr coord true
  allToAllTile "dV seq->heads" dV_heads_ptr dV_seq_ptr coord true
  barrierAllDevices "backward gradient return complete" 1

/-- Speculative orchestration shell for Ulysses backward.
This kernel documents the intended launch boundary:
1. `ulyssesDoAllToAllBwd`
2. local `FlashAttnBwd` launch on sequence-parallel Q/K/V/dO
3. `ulyssesGradReturnAllToAll`

The communication choreography is concrete; the single-kernel representation of
the local backward launch is intentionally not faked here. -/
@[gpu_kernel .SM90]
def ulyssesAttnBwd (q_seq_ptr : GPtr GpuFloat.BFloat16) (k_seq_ptr : GPtr GpuFloat.BFloat16)
    (v_seq_ptr : GPtr GpuFloat.BFloat16) (o_seq_ptr : GPtr GpuFloat.BFloat16)
    (dO_seq_ptr : GPtr GpuFloat.BFloat16)
    (l_seq_ptr : GPtr GpuFloat.Float32) (d_seq_ptr : GPtr GpuFloat.Float32)
    (dQ_heads_ptr : GPtr GpuFloat.BFloat16) (dK_heads_ptr : GPtr GpuFloat.BFloat16)
    (dV_heads_ptr : GPtr GpuFloat.BFloat16)
    (dQ_seq_ptr : GPtr GpuFloat.BFloat16) (dK_seq_ptr : GPtr GpuFloat.BFloat16)
    (dV_seq_ptr : GPtr GpuFloat.BFloat16)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  let _ := (q_seq_ptr, k_seq_ptr, v_seq_ptr, o_seq_ptr, dO_seq_ptr, l_seq_ptr, d_seq_ptr,
    dQ_seq_ptr, dK_seq_ptr, dV_seq_ptr, seq_len, head_dim, dev_idx, world_size)
  let coord ← blockCoord2D

  comment "=== Ulysses Attention Backward (transport shell) ==="
  comment "This is intentionally not a bespoke fused backward kernel."
  comment "Local sequence-parallel FlashAttention backward is delegated to a separate launch."

  comment "Mark the sequence-parallel backward window as ready"
  let marker : RT GpuFloat.BFloat16 16 128 ← allocRT .BFloat16 16 128
  let markerShared : ST GpuFloat.BFloat16 16 128 ← allocST .BFloat16 16 128
  asyncTileLoad markerShared dQ_seq_ptr coord (16 * 128 * 2)
  load marker markerShared
  multimemStore markerShared marker
  barrierAllDevices "local backward window opened" 0

  comment "Return dQ/dK/dV after the local backward launch completes"
  let _markerOut : RT GpuFloat.BFloat16 16 128 ← allocRT .BFloat16 16 128
  let gradShared : ST GpuFloat.BFloat16 16 128 ← allocST .BFloat16 16 128
  asyncTileLoad gradShared dQ_seq_ptr coord (16 * 128 * 2)
  storeGlobalAdd dQ_heads_ptr gradShared coord
  asyncTileLoad gradShared dK_seq_ptr coord (16 * 128 * 2)
  storeGlobalAdd dK_heads_ptr gradShared coord
  asyncTileLoad gradShared dV_seq_ptr coord (16 * 128 * 2)
  storeGlobalAdd dV_heads_ptr gradShared coord
  barrierAllDevices "local backward window closed" 1

end Tyr.GPU.Kernels.UlyssesAttn
