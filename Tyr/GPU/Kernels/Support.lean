import Tyr.GPU.Kernels.Prelude

/-!
# Tyr.GPU.Kernels.Support

Shared helpers for the concrete kernel catalog.

This module exists to keep the per-family kernel files focused on the kernel
phase structure rather than repeating the same TMA load, cross-device barrier,
and small vector helper definitions.
-/

namespace Tyr.GPU.Kernels.Support

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Async global-to-shared tile load with an explicit byte-count contract. -/
def asyncTileLoad {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout) (src : GPtr dtype) (coord : RTileCoord)
    (bytes : Nat) : KernelM Unit := do
  let sem ← allocSemaphore
  initSemaphore sem 1
  expectBytes sem bytes
  loadGlobalAsync dst src coord sem.id
  waitSemaphore sem

/-- Cross-device barrier helper used by the multi-device kernel families. -/
def barrierAllDevices (label : String) (barrierId : Nat) : KernelM Unit := do
  comment s!"Cross-device barrier: {label}"
  arriveAndWait barrierId

/-- Small vector max helper until the DSL grows a first-class `maxVec` op. -/
def maxVec {dtype : GpuFloat} {len : Nat}
    (dst a b : RV dtype len) : KernelM Unit := do
  emit (.binary .Max dst.id a.id b.id)

/-- Generic 16x128 all-to-all exchange tile used by the Ulysses and distributed
transport surfaces. -/
def allToAllTile (label : String) (output_ptr : GPtr GpuFloat.BFloat16)
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

end Tyr.GPU.Kernels.Support
