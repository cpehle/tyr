import LeanTest
import Tyr.Typed

open torch

namespace Tests.TestTyped

@[test]
def testDTypePromotionPolicy : IO Unit := do
  LeanTest.assertEqual (DType.promote .Float16 .BFloat16) .Float32
    "mixed f16/bf16 should promote to f32"
  LeanTest.assertEqual (DType.promote .Int16 .UInt8) .Int16
    "small mixed integer promotion should keep enough signed range"
  LeanTest.assertEqual (DType.meanResult .Int64) .Float32
    "integer means should promote to a floating dtype"
  LeanTest.assertEqual (DType.sumResult .Float8E4M3FN) .Float32
    "float8 reductions should accumulate as f32 in the typed policy"

@[test]
def testTensorSpecAlgebra : IO Unit := do
  LeanTest.assertEqual (TensorSpec.numelOfShape #[2, 3, 4]) 24
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[3] #[2, 3]) (some #[2, 3])
  LeanTest.assertEqual (TensorSpec.broadcastShapes? #[2, 3] #[4, 3]) none
  LeanTest.assertEqual (TensorSpec.matmulShape? #[2, 3] #[3, 4]) (some #[2, 4])
  LeanTest.assertEqual (TensorSpec.matmulShape? #[2, 3] #[5, 4]) none
  let lhs : TensorSpec := { shape := #[2, 3], dtype := .Float16 }
  let rhs : TensorSpec := { shape := #[3], dtype := .BFloat16 }
  LeanTest.assertTrue (TensorSpec.pointwise? lhs rhs == some { shape := #[2, 3], dtype := .Float32 })
    "pointwise specs should combine broadcast shape and promoted dtype"

@[test]
def testStrictSpecChecks : IO Unit := do
  let expected : TensorSpec := { shape := #[2, 3], dtype := .Float32 }
  let same : TensorSpec := { shape := #[2, 3], dtype := .Float32 }
  let wrongShape : TensorSpec := { shape := #[3, 2], dtype := .Float32 }
  let wrongDType : TensorSpec := { shape := #[2, 3], dtype := .BFloat16 }
  match TensorSpec.checkCompatible expected same with
  | .ok () => pure ()
  | .error err => LeanTest.fail err
  match TensorSpec.checkCompatible expected wrongShape with
  | .ok () => LeanTest.fail "shape mismatch should fail"
  | .error err => LeanTest.assertTrue (err.containsSubstr "Expected shape")
  match TensorSpec.checkCompatible expected wrongDType with
  | .ok () => LeanTest.fail "dtype mismatch should fail"
  | .error err => LeanTest.assertTrue (err.containsSubstr "Expected dtype")

@[test]
def testDevicePolicyChecks : IO Unit := do
  match DevicePolicy.check (.exact .CPU) .CPU with
  | .ok () => pure ()
  | .error err => LeanTest.fail err
  match DevicePolicy.check (.exact .MPS) .CPU with
  | .ok () => LeanTest.fail "device mismatch should fail"
  | .error err => LeanTest.assertTrue (err.containsSubstr "Expected device")

end Tests.TestTyped
