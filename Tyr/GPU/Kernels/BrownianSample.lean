/-
  Tyr/GPU/Kernels/BrownianSample.lean

  Device-side keyed Gaussian sampling — the leaf operation of the virtual
  Brownian tree. Each element draws from its own deterministic key, exactly
  replicating the CPU sampler (`PRNGKey`, an LCG mix chain plus Box-Muller)
  so device-sampled Brownian increments agree with the CPU path per element
  (up to Float32 evaluation of the transcendentals).

  CPU reference for element i, with `root` the parent key state:

      state = mix(root + i + G)            -- PRNGKey.foldIn
      u1 = ((mix(state + G))  >>> 11) / 2⁵³ -- PRNGKey.normal01
      u2 = ((mix(state + G2)) >>> 11) / 2⁵³
      out[i] = sqrt(-2·log(max u1 1e-12)) · cos(2π·u2)

  where mix(x) = x·A + C on UInt64 and G is the 64-bit golden ratio.
-/

import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Codegen.Macros

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def lcgA : Int := 6364136223846793005
private def lcgC : Int := 1442695040888963407
private def golden : Int := 0x9e3779b97f4a7c15
private def golden2 : Int := 0xbf58476d1ce4e5b9

/-- `mix(x) = x·A + C` on u64 (wrapping). -/
private def mixU64 (x : KVal UInt64) : KernelM (KVal UInt64) := do
  let a ← constUInt64Val lcgA "lcg_a"
  let c ← constUInt64Val lcgC "lcg_c"
  let xa ← scalarMulVal x a "mix_mul"
  scalarAddVal xa c "mix"

/-- `(x >>> 11) / 2⁵³` as Float32 (the shift via unsigned division). -/
private def uniform01 (x : KVal UInt64) : KernelM (KVal Float32) := do
  let shift ← constUInt64Val 2048 "shift11"
  let mant ← scalarDivVal x shift "mant"
  let mantF ← castFloat32 mant "mant_f32"
  let invDenom ← constFloatVal (1.0 / 9007199254740992.0) "inv_2_53"
  scalarMulVal mantF invDenom "uniform"

/-- Per-element keyed standard normal:
    `out[i] = normal01(foldIn(root, i), 0)` for `i < n`. -/
def keyedNormalBody
    (out : GPtr GpuFloat.Float32)
    (root : KVal UInt64)
    (n : KVal UInt64) : KernelM Unit := do
  comment "keyed Gaussian draws: out[i] = normal01(foldIn(root, i), 0)"
  let g ← constUInt64Val golden "golden"
  let g2 ← constUInt64Val golden2 "golden2"
  for i in (← parallelThreadRange n) do
    -- foldIn root i
    let ri ← scalarAddVal root i "root_plus_i"
    let rig ← scalarAddVal ri g "fold_arg"
    let state ← mixU64 rig
    -- u1
    let s1 ← scalarAddVal state g "u1_arg"
    let m1 ← mixU64 s1
    let u1 ← uniform01 m1
    let eps ← constFloatVal 1.0e-12 "u1_eps"
    let u1 ← scalarBinary .Max u1 eps "u1_clamped"
    -- u2
    let s2 ← scalarAddVal state g2 "u2_arg"
    let m2 ← mixU64 s2
    let u2 ← uniform01 m2
    -- Box-Muller
    let logU1 ← scalarUnary .Log u1 "log_u1"
    let negTwo ← constFloatVal (-2.0) "neg_two"
    let r2 ← scalarMulVal logU1 negTwo "r_sq"
    let r ← scalarUnary .Sqrt r2 "r"
    let twoPi ← constFloatVal 6.28318530717958647692 "two_pi"
    let theta ← scalarMulVal u2 twoPi "theta"
    let cosT ← scalarUnary .Cos theta "cos_theta"
    let val ← scalarMulVal r cosT "normal"
    let off ← castScalar .UInt32 i "off_u32"
    storeFloat32Scalar out off val

@[gpu_kernel .SM90]
def keyedNormal (out : GPtr GpuFloat.Float32) (root : KVal UInt64) (n : KVal UInt64) :
    KernelM Unit :=
  keyedNormalBody out root n

/-- Blackwell-family variant. -/
@[gpu_kernel .SM90]
def keyedNormalBlackwell (out : GPtr GpuFloat.Float32) (root : KVal UInt64) (n : KVal UInt64) :
    KernelM Unit := do
  setFamily .Blackwell
  keyedNormalBody out root n

end Tyr.GPU.Kernels
