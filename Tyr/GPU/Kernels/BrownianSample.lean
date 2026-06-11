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

/-- `foldIn(key, tag)` on a raw key state: `mix(state + tag + G)`. -/
def foldState (state tag : KVal UInt64) : KernelM (KVal UInt64) := do
  let g ← constUInt64Val golden "fold_g"
  let st ← scalarAddVal state tag "fold_st"
  let stg ← scalarAddVal st g "fold_stg"
  mixU64 stg

/-- `PRNGKey.normal01 {state} 0` on device: two LCG draws + Box-Muller. -/
def gaussianFromState (state : KVal UInt64) : KernelM (KVal Float32) := do
  let g ← constUInt64Val golden "golden"
  let g2 ← constUInt64Val golden2 "golden2"
  let s1 ← scalarAddVal state g "u1_arg"
  let m1 ← mixU64 s1
  let u1 ← uniform01 m1
  let eps ← constFloatVal 1.0e-12 "u1_eps"
  let u1 ← scalarBinary .Max u1 eps "u1_clamped"
  let s2 ← scalarAddVal state g2 "u2_arg"
  let m2 ← mixU64 s2
  let u2 ← uniform01 m2
  let logU1 ← scalarUnary .Log u1 "log_u1"
  let negTwo ← constFloatVal (-2.0) "neg_two"
  let r2 ← scalarMulVal logU1 negTwo "r_sq"
  let r ← scalarUnary .Sqrt r2 "r"
  let twoPi ← constFloatVal 6.28318530717958647692 "two_pi"
  let theta ← scalarMulVal u2 twoPi "theta"
  let cosT ← scalarUnary .Cos theta "cos_theta"
  scalarMulVal r cosT "normal"

/-- Per-element keyed standard normal:
    `out[i] = normal01(foldIn(root, i), 0)` for `i < n`. -/
def keyedNormalBody
    (out : GPtr GpuFloat.Float32)
    (root : KVal UInt64)
    (n : KVal UInt64) : KernelM Unit := do
  comment "keyed Gaussian draws: out[i] = normal01(foldIn(root, i), 0)"
  for i in (← parallelThreadRange n) do
    let state ← foldState root i
    let val ← gaussianFromState state
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

/-! ## Virtual-Brownian-tree descent

The Brownian-bridge descent for a value query `w(t)` is time-driven: the
visited nodes depend only on the query and tree times, never on sampled
values. The HOST therefore precomputes the node schedule (see
`Tyr.GPU.Ops.BrownianFused.buildValueSchedule`) and the kernel walks it
branchlessly per element, drawing each node's Gaussian from the same keyed
chains as the CPU sampler.

Schedule encoding, six Float32 per record (all time tags are
`timeToUInt32` values ≤ 2²⁴, hence exact in f32):

  [tagA, tagB, tagExtra, kind, c0, c1]

  record 0:  root — w1 = c0 · N(intervalKey(tagA, tagB)), w0 = 0
  interior:  kind 0 (descend left:  w1 ← wMid) or 1 (right: w0 ← wMid),
             wMid = ½(w0+w1) + c0 · N(midpointKey(tagA, tagB))
  terminal:  kind 2 — value = wMid (midpoint hit)
             kind 3 — value = w0 + c1·(w1−w0) + c0 · N(pointKey(tagA, tagB, tagExtra))
-/

private def tagInterval : Int := 0x9e3779b9
private def tagMidpoint : Int := 0x243f6a88
private def tagPoint : Int := 0x13198a2e

/-- Keyed Gaussian for one schedule node: `N(chain(childState, opTag, tags…))`.
    The extra tag is always folded for the point chain (its draw is unused
    on non-bridge records). -/
private def nodeGaussian (childState : KVal UInt64) (opTag : Int)
    (tagA tagB : KVal UInt64) (tagExtra : Option (KVal UInt64)) :
    KernelM (KVal Float32) := do
  let base ← mixU64 childState   -- baseKey of the child state
  let op ← constUInt64Val opTag "op_tag"
  let k1 ← foldState base op
  let k2 ← foldState k1 tagA
  let k3 ← foldState k2 tagB
  let key ← match tagExtra with
    | some e => foldState k3 e
    | none => pure k3
  gaussianFromState key

/-- Load schedule field `slot` of record `node` as Float32. -/
private def schedField (sched : GPtr GpuFloat.Float32)
    (node : KVal UInt64) (slot : Nat) : KernelM (KVal Float32) := do
  let six ← constUInt64Val 6 "six"
  let s ← constUInt64Val (Int.ofNat slot) "slot"
  let base ← scalarMulVal node six "rec_base"
  let off64 ← scalarAddVal base s "rec_off"
  let off ← castScalar .UInt32 off64 "rec_off_u32"
  loadFloat32Scalar sched off "sched_f"

/-- Walk one value-query schedule for the element with key state
    `childState`; returns `w(t)`. -/
private def walkSchedule (childState : KVal UInt64)
    (sched : GPtr GpuFloat.Float32) (len : KVal UInt64) :
    KernelM (KVal Float32) := do
  let zeroU ← constUInt64Val 0 "zero_u"
  let oneU ← constUInt64Val 1 "one_u"
  -- root record
  let tagAF ← schedField sched zeroU 0
  let tagBF ← schedField sched zeroU 1
  let c0 ← schedField sched zeroU 4
  let tagA32 : KVal UInt32 ← castScalar .UInt32 tagAF "ta32"
  let tagB32 : KVal UInt32 ← castScalar .UInt32 tagBF "tb32"
  let tagA : KVal UInt64 ← castScalar .UInt64 tagA32 "ta"
  let tagB : KVal UInt64 ← castScalar .UInt64 tagB32 "tb"
  let rootDraw ← nodeGaussian childState tagInterval tagA tagB none
  let zeroF ← constFloatVal 0.0 "w0_init"
  let w0 ← allocScalar .Float32 "w0" (some zeroF)
  let w1v ← scalarMulVal c0 rootDraw "w1_init"
  let w1 ← allocScalar .Float32 "w1" (some w1v)
  let result ← allocScalar .Float32 "walk_result" (some w1v)
  let half ← constFloatVal 0.5 "half"
  let k0 ← constFloatVal 0.0 "kind0"
  let k1c ← constFloatVal 1.0 "kind1"
  let k2c ← constFloatVal 2.0 "kind2"
  let k3c ← constFloatVal 3.0 "kind3"
  for node in kstride oneU len oneU do
    let tA ← schedField sched node 0
    let tB ← schedField sched node 1
    let tE ← schedField sched node 2
    let kind ← schedField sched node 3
    let c0 ← schedField sched node 4
    let c1 ← schedField sched node 5
    let tA32 : KVal UInt32 ← castScalar .UInt32 tA "na32"
    let tB32 : KVal UInt32 ← castScalar .UInt32 tB "nb32"
    let tE32 : KVal UInt32 ← castScalar .UInt32 tE "ne32"
    let tagA : KVal UInt64 ← castScalar .UInt64 tA32 "na"
    let tagB : KVal UInt64 ← castScalar .UInt64 tB32 "nb"
    let tagE : KVal UInt64 ← castScalar .UInt64 tE32 "ne"
    -- midpoint draw and candidate
    let uMid ← nodeGaussian childState tagMidpoint tagA tagB none
    let mean ← scalarMulVal (← scalarAddVal w0 w1 "wsum") half "wmean"
    let wMid ← scalarAddVal mean (← scalarMulVal c0 uMid "mid_noise") "w_mid"
    -- bridge draw and candidate (uses pre-update w0/w1)
    let uPt ← nodeGaussian childState tagPoint tagA tagB (some tagE)
    let span ← scalarBinary .Sub w1 w0 "span"
    let drift ← scalarMulVal c1 span "drift"
    let bridge0 ← scalarAddVal w0 drift "bridge_mean"
    let wBridge ← scalarAddVal bridge0 (← scalarMulVal c0 uPt "pt_noise") "w_bridge"
    -- branchless updates
    let isLeft ← scalarEq kind k0 "is_left"
    let isRight ← scalarEq kind k1c "is_right"
    let isMid ← scalarEq kind k2c "is_mid"
    let isBridge ← scalarEq kind k3c "is_bridge"
    let w1' ← scalarSelect isLeft wMid w1 "w1_next"
    let w0' ← scalarSelect isRight wMid w0 "w0_next"
    assignScalar w1 w1'
    assignScalar w0 w0'
    let r1 ← scalarSelect isMid wMid result "res_mid"
    let r2 ← scalarSelect isBridge wBridge r1 "res_bridge"
    assignScalar result r2
  pure result

/-- Per-element Brownian increment `w(tB) − w(tA)` over two precomputed
    value-query schedules. -/
def vbtIncrementBody
    (out : GPtr GpuFloat.Float32)
    (schedA : GPtr GpuFloat.Float32)
    (schedB : GPtr GpuFloat.Float32)
    (root : KVal UInt64)
    (n lenA lenB : KVal UInt64) : KernelM Unit := do
  comment "virtual-Brownian-tree increment: out[i] = w_i(tB) − w_i(tA)"
  for i in (← parallelThreadRange n) do
    let childState ← foldState root i
    let wA ← walkSchedule childState schedA lenA
    let wB ← walkSchedule childState schedB lenB
    let inc ← scalarBinary .Sub wB wA "increment"
    let off ← castScalar .UInt32 i "out_off"
    storeFloat32Scalar out off inc

@[gpu_kernel .SM90]
def vbtIncrement
    (out schedA schedB : GPtr GpuFloat.Float32)
    (root : KVal UInt64) (n lenA lenB : KVal UInt64) : KernelM Unit :=
  vbtIncrementBody out schedA schedB root n lenA lenB

/-- Blackwell-family variant. -/
@[gpu_kernel .SM90]
def vbtIncrementBlackwell
    (out schedA schedB : GPtr GpuFloat.Float32)
    (root : KVal UInt64) (n lenA lenB : KVal UInt64) : KernelM Unit := do
  setFamily .Blackwell
  vbtIncrementBody out schedA schedB root n lenA lenB

end Tyr.GPU.Kernels
