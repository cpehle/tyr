/- Runtime attention problem description and current specialization selection.

   This is the thin operator-facing layer that should sit above concrete kernel
   families. Public wrappers should construct an `AttentionProblem`, let this
   module choose the best available specialization, and only then dispatch to a
   generated kernel or a portable fallback.

   The current specialization matrix is intentionally small:

   - `tkMhaH1002Block`   : one-H100, BF16, dense prefill, `seq=128`, `headDim=64`
   - `tkMhaH10012Block`  : one-H100, BF16, dense prefill, `seq=768`, `headDim=64`
   - `portable`          : everything else

   The key design point is that the public API is now centered on a runtime
   problem descriptor rather than a handful of ad hoc fixed-shape predicates.
-/
import Tyr.Torch
import Tyr.GPU.Types

namespace Tyr.GPU.Ops

open torch
open Tyr.GPU

/-- High-level attention execution mode. -/
inductive AttentionMode where
  | densePrefill
  | decode
  | slidingWindow
  deriving Repr, Inhabited, BEq

/-- Mask representation class exposed at the runtime surface. -/
inductive AttentionMaskKind where
  | none
  | padding
  | explicitQK
  deriving Repr, Inhabited, BEq

/-- Coarse head-dimension bucket used for specialization planning. -/
inductive HeadDimClass where
  | d64
  | d128
  | d256
  | d512
  | other (dim : UInt64)
  deriving Repr, Inhabited, BEq

/-- Grouped-query structure class. `grouped r` means `numQHeads / numKVHeads = r`. -/
inductive GqaClass where
  | equal
  | grouped (ratio : UInt64)
  | invalid
  deriving Repr, Inhabited, BEq

/-- Current native specialization set known to the runtime operator layer. -/
inductive AttentionSpecialization where
  | portable
  | tkMhaH1002Block
  | tkMhaH10012Block
  deriving Repr, Inhabited, BEq

namespace AttentionSpecialization

/-- Whether the selection corresponds to a native generated-kernel path. -/
def isNative : AttentionSpecialization → Bool
  | .portable => false
  | _ => true

/-- Tile-block count for the current H100 MHA specializations, when applicable. -/
def kvBlocks? : AttentionSpecialization → Option UInt64
  | .portable => none
  | .tkMhaH1002Block => some 2
  | .tkMhaH10012Block => some 12

end AttentionSpecialization

/-- Runtime description of an attention call.

    Fields are intentionally broader than the current native kernel set so this
    type can survive future backend growth without changing the public surface. -/
structure AttentionProblem where
  batch : UInt64
  numQHeads : UInt64
  numKVHeads : UInt64
  qSeq : UInt64
  kvSeq : UInt64
  headDim : UInt64
  dtype : DType
  device : Device
  arch : GpuArch := .SM90
  mode : AttentionMode := .densePrefill
  maskKind : AttentionMaskKind := .none
  dropoutP : Float := 0.0
  isCausal : Bool := false
  scale : Option Float := none
  enableGqa : Bool := false
  windowSize : Option UInt64 := none
  deriving Repr, Inhabited

namespace AttentionProblem

/-- Infer the broad attention mode from sequence lengths and optional windowing. -/
def inferMode (qSeq kvSeq : UInt64) (windowSize : Option UInt64 := none) : AttentionMode :=
  match windowSize with
  | some _ => .slidingWindow
  | none =>
      if qSeq == 1 && kvSeq >= qSeq then
        .decode
      else
        .densePrefill

/-- Construct a runtime problem descriptor from typed Q/K/V tensors. -/
def ofQKV
    {batch nHead nKvHead qSeq kvSeq headDim : UInt64}
    (q : T #[batch, nHead, qSeq, headDim])
    (k : T #[batch, nKvHead, kvSeq, headDim])
    (v : T #[batch, nKvHead, kvSeq, headDim])
    (attnMask : Option (T #[batch, kvSeq]) := none)
    (dropoutP : Float := 0.0)
    (isCausal : Bool := false)
    (scale : Option Float := none)
    (enableGqa : Bool := false)
    (windowSize : Option UInt64 := none)
    (arch : GpuArch := .SM90)
    : AttentionProblem :=
    let _sameDType := q.dtype == k.dtype && k.dtype == v.dtype
    let _sameDevice := q.device == k.device && k.device == v.device
    {
      batch := batch
      numQHeads := nHead
      numKVHeads := nKvHead
      qSeq := qSeq
      kvSeq := kvSeq
      headDim := headDim
      dtype := q.dtype
      device := q.device
      arch := arch
      mode := inferMode qSeq kvSeq windowSize
      maskKind := if attnMask.isSome then .padding else .none
      dropoutP := dropoutP
      isCausal := isCausal
      scale := scale
      enableGqa := enableGqa
      windowSize := windowSize
    }

/-- Shape-only helper for fixed-layout self-attention surfaces. -/
def selfAttention
    (seq : UInt64)
    (headDim : UInt64 := 64)
    (device : Device := .CUDA 0)
    (dtype : DType := .BFloat16)
    (isCausal : Bool := false)
    (arch : GpuArch := .SM90)
    : AttentionProblem := {
      batch := 1
      numQHeads := 1
      numKVHeads := 1
      qSeq := seq
      kvSeq := seq
      headDim := headDim
      dtype := dtype
      device := device
      arch := arch
      mode := inferMode seq seq none
      isCausal := isCausal
    }

/-- Whether the problem is placed on a CUDA device. -/
def isCuda (problem : AttentionProblem) : Bool :=
  match problem.device with
  | .CUDA _ => true
  | _ => false

/-- Whether query and KV sequence lengths match. -/
def isSelfAttention (problem : AttentionProblem) : Bool :=
  problem.qSeq == problem.kvSeq

/-- Whether `qSeq` and `kvSeq` are aligned to the current tile size. -/
def tileAligned (problem : AttentionProblem) (tile : UInt64 := 64) : Bool :=
  problem.qSeq % tile == 0 && problem.kvSeq % tile == 0

/-- Whether the provided scale matches the standard `1 / sqrt(headDim)` default. -/
def scaleMatchesDefault (problem : AttentionProblem) : Bool :=
  match problem.scale with
  | none => true
  | some s =>
      let expected := 1.0 / Float.sqrt problem.headDim.toFloat
      Float.abs (s - expected) <= 1.0e-6

/-- Coarse head-dimension bucket used by specialization planning. -/
def headDimClass (problem : AttentionProblem) : HeadDimClass :=
  match problem.headDim with
  | 64 => .d64
  | 128 => .d128
  | 256 => .d256
  | 512 => .d512
  | d => .other d

/-- Grouped-query ratio class derived from the runtime head counts. -/
def gqaClass (problem : AttentionProblem) : GqaClass :=
  if problem.numKVHeads == 0 then
    .invalid
  else if problem.numQHeads == problem.numKVHeads then
    .equal
  else if problem.numQHeads % problem.numKVHeads == 0 then
    .grouped (problem.numQHeads / problem.numKVHeads)
  else
    .invalid

/-- Whether the current one-H100 TK-backed selector may consider this problem. -/
def currentTkBaseEligible (problem : AttentionProblem) : Bool :=
  let deviceOk := problem.isCuda
  let dtypeOk := problem.dtype == .BFloat16
  let modeOk := problem.mode == .densePrefill
  let semanticsOk :=
    problem.maskKind == .none &&
    problem.dropoutP == 0.0 &&
    !problem.enableGqa &&
    problem.scaleMatchesDefault
  let shapeOk :=
    problem.batch == 1 &&
    problem.numQHeads == 1 &&
    problem.numKVHeads == 1 &&
    problem.isSelfAttention &&
    problem.headDim == 64 &&
    !problem.isCausal
  deviceOk && dtypeOk && modeOk && semanticsOk && shapeOk

/-- Current specialization selector.

    This is intentionally conservative. It centralizes the existing native
    coverage and portable fallback decision in one place so the wrappers do not
    each re-encode the same fixed-shape logic. -/
def currentSpecialization (problem : AttentionProblem) : AttentionSpecialization :=
  if !currentTkBaseEligible problem then
    .portable
  else if problem.qSeq == 128 then
    .tkMhaH1002Block
  else if problem.qSeq == 768 then
    .tkMhaH10012Block
  else
    .portable

/-- Whether the current runtime layer has a native specialization for `problem`. -/
def hasNativeKernel (problem : AttentionProblem) : Bool :=
  (currentSpecialization problem).isNative

end AttentionProblem

end Tyr.GPU.Ops
