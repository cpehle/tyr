/-
  Tyr/GPU/Kernels/UlyssesAttn.lean

  Ulysses is modeled here as transport/orchestration around all-to-all, matching
  the vendored ThunderKittens `ulysses_attn.cu`. The actual local attention math
  belongs to a separate FlashAttention launch; this module owns the movement
  between head-parallel and sequence-parallel layouts.
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
    (input_ptr : GPtr GpuFloat.BFloat16) (coord : RTileCoord) : KernelM Unit := do
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
  storeGlobalAsync output_ptr exchangeShared coord
  sync

/-- Standalone ThunderKittens-shaped all-to-all transport kernel.
The caller supplies already-partitioned views and chooses the axis mapping by
which input/output pointers it passes in. -/
@[gpu_kernel .SM90]
def allToAllFwd (output_ptr : GPtr GpuFloat.BFloat16) (input_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32)
    (scatter_axis : KVal UInt32) (gather_axis : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size, scatter_axis, gather_axis)
  let coord ← blockCoord2D
  comment "Ulysses all_to_all transport surface; axis selection is an orchestration concern"
  allToAllTile "generic shard" output_ptr input_ptr coord
  barrierAllDevices "all_to_all epilogue" 1

/-- Phase 1/2 of Ulysses forward: redistribute Q/K/V from head-parallel shards
into the sequence-parallel views consumed by a local FlashAttention launch. -/
@[gpu_kernel .SM90]
def ulyssesQkvAllToAll (q_seq_ptr : GPtr GpuFloat.BFloat16) (k_seq_ptr : GPtr GpuFloat.BFloat16)
    (v_seq_ptr : GPtr GpuFloat.BFloat16)
    (q_heads_ptr : GPtr GpuFloat.BFloat16) (k_heads_ptr : GPtr GpuFloat.BFloat16)
    (v_heads_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  comment "Ulysses QKV transport: head-parallel -> sequence-parallel"
  allToAllTile "Q heads->seq" q_seq_ptr q_heads_ptr coord
  allToAllTile "K heads->seq" k_seq_ptr k_heads_ptr coord
  allToAllTile "V heads->seq" v_seq_ptr v_heads_ptr coord
  barrierAllDevices "QKV transport complete" 1

/-- Phase 4 of Ulysses forward: return the local attention output from the
sequence-parallel layout back to the head-parallel view. -/
@[gpu_kernel .SM90]
def ulyssesAttnFwd (o_heads_ptr : GPtr GpuFloat.BFloat16) (o_seq_ptr : GPtr GpuFloat.BFloat16)
    (q_seq_ptr : GPtr GpuFloat.BFloat16) (k_seq_ptr : GPtr GpuFloat.BFloat16)
    (v_seq_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (q_seq_ptr, k_seq_ptr, v_seq_ptr, dev_idx, world_size)
  let coord ← blockCoord2D

  comment "Ulysses orchestration shell:"
  comment "1. `ulyssesQkvAllToAll` runs before local attention."
  comment "2. A separate FlashAttention launch consumes q_seq/k_seq/v_seq."
  comment "3. This kernel returns the local attention output back to head shards."
  allToAllTile "O seq->heads" o_heads_ptr o_seq_ptr coord
  barrierAllDevices "output return complete" 1

/-- Legacy compatibility surface: no fused attention body remains here.
This now aliases the output-return transport stage so the module stays focused
on all-to-all orchestration. -/
@[gpu_kernel .SM90]
def ulyssesAttnFusedFwd (o_heads_ptr : GPtr GpuFloat.BFloat16) (o_seq_ptr : GPtr GpuFloat.BFloat16)
    (dev_idx : KVal UInt32) (world_size : KVal UInt32) : KernelM Unit := do
  let _ := (dev_idx, world_size)
  let coord ← blockCoord2D

  comment "Legacy fused entrypoint retained as transport-only return path"
  allToAllTile "legacy O seq->heads" o_heads_ptr o_seq_ptr coord
  barrierAllDevices "legacy fused epilogue" 1

end Tyr.GPU.Kernels.UlyssesAttn
