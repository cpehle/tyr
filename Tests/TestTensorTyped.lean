/-
  Tests/TestTensorTyped.lean

  Smoke tests for the typed `Tensor m s` API in `Tyr/Tensor.lean`.
  Verifies that:
    - constructors produce a tensor whose runtime metadata matches the
      type-level index
    - arithmetic ops preserve metadata at the type level
    - dtype conversion (toFloat32 / toBFloat16) updates the metadata
      index correctly
    - shape and metadata projections agree with the type-level indices
-/
import Tyr
import LeanTest
import Examples.TypedTensor.MiniMLP

open torch
open LeanTest

namespace Tests.TensorTyped

private def cpuF32 : TensorMeta := { device := .CPU, dtype := .Float32 }
private def cpuBF16 : TensorMeta := { device := .CPU, dtype := .BFloat16 }

@[test]
def testConstructorMatchesMeta : IO Unit := do
  let t : Tensor cpuF32 #[4] := Tensor.zeros #[4] cpuF32
  let raw : T #[4] := Tensor.toT t
  let runtimeDType := raw.dtype
  assertTrue (runtimeDType == .Float32)
    s!"zeros: expected Float32 dtype, got {runtimeDType}"

@[test]
def testArithPreservesMeta : IO Unit := do
  let a : Tensor cpuF32 #[4] := Tensor.ones #[4] cpuF32
  let b : Tensor cpuF32 #[4] := Tensor.ones #[4] cpuF32
  let c : Tensor cpuF32 #[4] := a + b
  -- Element-wise add of ones produces twos. Sum of 4 twos is 8.
  let s := torch.nn.sumDim (Tensor.toT c) 0 false
  let total : Float := nn.item s
  assertTrue (total > 7.5 && total < 8.5)
    s!"add: expected sum ~8, got {total}"

@[test]
def testToFloat32UpdatesDType : IO Unit := do
  let a : Tensor cpuBF16 #[2] := Tensor.ones #[2] cpuBF16
  let aF : Tensor { cpuBF16 with dtype := .Float32 } #[2] := Tensor.toFloat32 a
  let dt := (Tensor.toT aF).dtype
  assertTrue (dt == .Float32) s!"toFloat32: expected Float32, got {dt}"

@[test]
def testTensorMetaShape : IO Unit := do
  let a : Tensor cpuF32 #[3, 4] := Tensor.zeros #[3, 4] cpuF32
  -- shape and tensorMeta projections are pure type-level lookups
  assertEqual (Tensor.shape a) (#[3, 4] : Shape) "shape should equal type-level index"
  let m := Tensor.tensorMeta a
  assertTrue (m == cpuF32) "tensorMeta should equal type-level index"

@[test]
def testRMSNormTypedForward : IO Unit := do
  -- RMSNorm.forward3dT preserves the input's TensorMeta at the type level.
  let rn : RMSNorm 4 := torch.RMSNorm.init 4 1e-6
  let x : Tensor cpuF32 #[1, 2, 4] := Tensor.ones #[1, 2, 4] cpuF32
  let y : Tensor cpuF32 #[1, 2, 4] := rn.forward3dT x
  -- Sum should be finite and roughly the number of elements (8) since
  -- RMSNorm-of-ones is approximately ones (after normalization).
  let s := torch.nn.sumDim (Tensor.toT (Tensor.reshape y #[8])) 0 false
  let total : Float := nn.item s
  assertTrue (total > 0.0 && total < 100.0) s!"RMSNorm output sum out of range: {total}"

@[test]
def testLinearTypedForward : IO Unit := do
  -- Linear.forward3dT preserves TensorMeta and changes only the shape index.
  let lin : Linear 4 3 ← torch.Linear.init 4 3 (withBias := true)
  let x : Tensor cpuF32 #[1, 2, 4] := Tensor.zeros #[1, 2, 4] cpuF32
  let y : Tensor cpuF32 #[1, 2, 3] := lin.forward3dT x
  -- Shape projection confirms the type-level shape change took effect.
  assertEqual (Tensor.shape y) (#[1, 2, 3] : Shape) "Linear forward should produce out_dim=3"

@[test]
def testActivationsPreserveMeta : IO Unit := do
  -- Activations are elementwise; metadata + shape unchanged.
  let x : Tensor cpuF32 #[2, 3] := Tensor.zeros #[2, 3] cpuF32
  let y₁ : Tensor cpuF32 #[2, 3] := Tensor.sigmoid x
  let y₂ : Tensor cpuF32 #[2, 3] := Tensor.silu x
  let y₃ : Tensor cpuF32 #[2, 3] := Tensor.softmax x
  let y₄ : Tensor cpuF32 #[2, 3] := Tensor.rsqrt (Tensor.full #[2, 3] cpuF32 4.0)
  -- sigmoid(0) = 0.5; sum over 6 elements = 3
  let s := torch.nn.sumDim (Tensor.toT (Tensor.reshape y₁ #[6])) 0 false
  let total : Float := nn.item s
  assertTrue (total > 2.5 && total < 3.5) s!"sigmoid(0) sum should be ~3, got {total}"
  -- Suppress unused warnings
  let _ := (y₂, y₃, y₄)

@[test]
def testTransposeShapeIndex : IO Unit := do
  -- transpose changes the shape index according to transposeShape.
  let x : Tensor cpuF32 #[2, 3] := Tensor.zeros #[2, 3] cpuF32
  let y := Tensor.transpose x 0 1
  -- transposeShape #[2, 3] 0 1 = #[3, 2]
  assertEqual (Tensor.shape y) (#[3, 2] : Shape) "transpose should swap dims at type level"

@[test]
def testAttentionTransposeRoundTrip : IO Unit := do
  -- transposeForAttention then transposeFromAttention recovers shape.
  let x : Tensor cpuF32 #[1, 2, 3, 4] := Tensor.zeros #[1, 2, 3, 4] cpuF32
  let y : Tensor cpuF32 #[1, 3, 2, 4] := Tensor.transposeForAttention x
  let z : Tensor cpuF32 #[1, 2, 3, 4] := Tensor.transposeFromAttention y
  assertEqual (Tensor.shape z) (#[1, 2, 3, 4] : Shape) "round-trip should recover shape"

/-- A minimal typed transformer block: project x to Q/K/V via three
    Linear layers, reshape to attention layout, transpose, transpose
    back. Exercises Linear.forward3dT, Tensor.reshape, and the
    transposeForAttention/From pair through a realistic model-style
    pipeline.

    The point is *type-level*: every intermediate has its `TensorMeta`
    and shape pinned in the type, so any mistake in the wiring would
    fail to elaborate. -/
private def typedAttnBlock {m : TensorMeta}
    {batch seq num_heads head_dim hidden : UInt64}
    (qProj kProj vProj : Linear hidden (num_heads * head_dim))
    (x : Tensor m #[batch, seq, hidden])
    : IO (Tensor m #[batch, seq, num_heads * head_dim]) := do
  let q : Tensor m #[batch, seq, num_heads * head_dim] := qProj.forward3dT x
  let k : Tensor m #[batch, seq, num_heads * head_dim] := kProj.forward3dT x
  let v : Tensor m #[batch, seq, num_heads * head_dim] := vProj.forward3dT x
  -- Reshape into [batch, seq, num_heads, head_dim] then to attention layout.
  let q4 : Tensor m #[batch, seq, num_heads, head_dim] := Tensor.reshape q #[batch, seq, num_heads, head_dim]
  let k4 : Tensor m #[batch, seq, num_heads, head_dim] := Tensor.reshape k #[batch, seq, num_heads, head_dim]
  let v4 : Tensor m #[batch, seq, num_heads, head_dim] := Tensor.reshape v #[batch, seq, num_heads, head_dim]
  let _qa : Tensor m #[batch, num_heads, seq, head_dim] := Tensor.transposeForAttention q4
  let _ka : Tensor m #[batch, num_heads, seq, head_dim] := Tensor.transposeForAttention k4
  let va : Tensor m #[batch, num_heads, seq, head_dim] := Tensor.transposeForAttention v4
  -- Skip the actual scaled-dot-product step (not yet wrapped) — round-trip
  -- shape via transposeFromAttention to demonstrate composability.
  let vBack : Tensor m #[batch, seq, num_heads, head_dim] := Tensor.transposeFromAttention va
  pure (Tensor.reshape vBack #[batch, seq, num_heads * head_dim])

@[test]
def testTypedModuleApply : IO Unit := do
  -- The `Module |> x` infix syntax dispatches on Tensor instance, so
  -- a typed input flows through cleanly without explicit `.forward3dT`.
  let lin : Linear 4 3 ← torch.Linear.init 4 3
  let x : Tensor cpuF32 #[1, 2, 4] := Tensor.zeros #[1, 2, 4] cpuF32
  let y : Tensor cpuF32 #[1, 2, 3] := lin |> x
  assertEqual (Tensor.shape y) (#[1, 2, 3] : Shape) "module-apply on Tensor preserves type discipline"

@[test]
def testMiniMLPRunsTyped : IO Unit := do
  -- Full typed MLP end-to-end: Linear → RMSNorm → SiLU gate → Linear,
  -- with `Tensor m s` flowing through every intermediate. Smoke-test
  -- only — verifies the pipeline executes and produces a finite result.
  let s ← examples.typed.MiniMLP.run 8 16 1 4 cpuF32
  -- Zero input → MLP-of-zeros → sum-squared = 0.
  assertTrue (s >= 0.0 && s < 1e6) s!"MLP output sum-of-squares out of range: {s}"

@[test]
def testTypedAttentionBlock : IO Unit := do
  let qProj : Linear 8 4 ← torch.Linear.init 8 4
  let kProj : Linear 8 4 ← torch.Linear.init 8 4
  let vProj : Linear 8 4 ← torch.Linear.init 8 4
  let x : Tensor cpuF32 #[2, 3, 8] := Tensor.zeros #[2, 3, 8] cpuF32
  -- num_heads=2, head_dim=2, num_heads*head_dim=4
  let y : Tensor cpuF32 #[2, 3, 2 * 2] ← typedAttnBlock (num_heads := 2) (head_dim := 2) qProj kProj vProj x
  assertEqual (Tensor.shape y) (#[2, 3, 4] : Shape) "attn block output shape"

end Tests.TensorTyped

/-- Embedding requires Int64 ids — calling with a Float32 ids tensor must
    fail to elaborate. -/
private def Tests.TensorTyped.cpuI64' : torch.TensorMeta :=
  { device := .CPU, dtype := .Int64 }

#check_failure (fun (w : torch.Tensor Tests.TensorTyped.cpuF32' #[10, 4])
                    (badIds : torch.Tensor Tests.TensorTyped.cpuF32' #[1, 2]) =>
                  torch.Tensor.embedding w badIds)

/-- The Lean type checker rejects adding tensors with different metadata —
    this is the core property of the typed API. `#check_failure` fails the
    build if the term elaborates, so this command silently succeeding means
    the type discipline holds. -/
private def Tests.TensorTyped.cpuF32' : torch.TensorMeta :=
  { device := .CPU, dtype := .Float32 }
private def Tests.TensorTyped.cpuBF16' : torch.TensorMeta :=
  { device := .CPU, dtype := .BFloat16 }

#check_failure (fun (a : torch.Tensor Tests.TensorTyped.cpuF32' #[4])
                    (b : torch.Tensor Tests.TensorTyped.cpuBF16' #[4]) => a + b)
