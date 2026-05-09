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

end Tests.TensorTyped

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
