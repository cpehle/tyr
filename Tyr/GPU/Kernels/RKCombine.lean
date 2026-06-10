/-
  Tyr/GPU/Kernels/RKCombine.lean

  Fused explicit-RK stage combination.

  A Dopri5 step on tensor states currently issues ~60 small libtorch calls
  (stage sums, solution/error combinations) at ~1.7µs of FFI/dispatch glue
  each — ~100µs/step for a 64-element state where the arithmetic itself is
  nanoseconds (see the `glue-timing` probe in `Tests/TestDiffEqTyped.lean`).
  This kernel fuses the end-of-step combination

      y1  = y0 + Σᵢ bᵢ·kᵢ
      err = Σᵢ (bᵢ − b̂ᵢ)·kᵢ

  into one launch per 64×64 tile. The body is generated from the solution
  and embedded weights, so any explicit tableau gets its fused kernel by
  applying `rkCombineBody` — the miniature tableau-driven generator.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Emit the fused stage combination for solution weights `b` and embedded
    weights `bHat` (error weights are `bᵢ − b̂ᵢ`; zero-weight stages are
    skipped at codegen time). One 64×64 Float32 tile per block; stages are
    streamed through a single shared tile to bound register pressure. -/
def rkCombineBody (b bHat : Array Float)
    (y0Ptr : GPtr GpuFloat.Float32)
    (kPtrs : Array (GPtr GpuFloat.Float32))
    (y1Ptr errPtr : GPtr GpuFloat.Float32) : KernelM Unit := do
  comment "fused explicit-RK combination: y1 = y0 + Σ bᵢkᵢ, err = Σ (bᵢ−b̂ᵢ)kᵢ"
  let coord ← blockCoord2D
  let smem ← allocST .Float32 64 64
  let work ← allocRT .Float32 64 64
  let scratch ← allocRT .Float32 64 64
  let acc ← allocRT .Float32 64 64
  let errAcc ← allocRT .Float32 64 64
  loadGlobal smem y0Ptr coord
  load acc smem
  zero errAcc
  sync
  let mut i := 0
  for kPtr in kPtrs do
    let bi := b.getD i 0.0
    let ei := bi - bHat.getD i 0.0
    i := i + 1
    if bi != 0.0 || ei != 0.0 then
      loadGlobal smem kPtr coord
      load work smem
      sync
      if bi != 0.0 then
        scalarMul scratch work bi
        add acc acc scratch
      if ei != 0.0 then
        scalarMul scratch work ei
        add errAcc errAcc scratch
  store smem acc
  sync
  storeGlobal y1Ptr smem coord
  sync
  store smem errAcc
  sync
  storeGlobal errPtr smem coord
  sync

private def dopri5B : Array Float :=
  #[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
    -2187.0 / 6784.0, 11.0 / 84.0, 0.0]

private def dopri5BHat : Array Float :=
  #[1951.0 / 21600.0, 0.0, 22642.0 / 50085.0, 451.0 / 720.0,
    -12231.0 / 42400.0, 649.0 / 6300.0, 1.0 / 60.0]

/-- Fused Dopri5 solution + error combination for one 64×64 tile:
    seven stage tensors in, `y1` and the embedded error estimate out, in a
    single kernel launch. -/
@[gpu_kernel .SM90]
def dopri5Combine64
    (y0 k1 k2 k3 k4 k5 k6 k7 y1 err : GPtr GpuFloat.Float32) : KernelM Unit :=
  rkCombineBody dopri5B dopri5BHat y0 #[k1, k2, k3, k4, k5, k6, k7] y1 err

/-- Blackwell-family variant of the fused Dopri5 combination. -/
@[gpu_kernel .SM90]
def dopri5Combine64Blackwell
    (y0 k1 k2 k3 k4 k5 k6 k7 y1 err : GPtr GpuFloat.Float32) : KernelM Unit := do
  setFamily .Blackwell
  rkCombineBody dopri5B dopri5BHat y0 #[k1, k2, k3, k4, k5, k6, k7] y1 err

end Tyr.GPU.Kernels
