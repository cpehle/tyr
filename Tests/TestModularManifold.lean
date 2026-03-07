import LeanTest
import Tyr.Modular
import Tyr.Manifolds

namespace Tests.ModularManifold

open LeanTest
open torch
open Tyr.Modular

@[test]
def testStiefelLinearConstraintAfterUpdate : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 true
  let gW ← randn #[16, 8] false
  let gB ← randn #[16] false
  let layer' := ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02

  let w := MatrixManifoldCarrier.toMatrix layer'.weight
  let wtW := nn.mm (nn.transpose2d w) w
  let I := eye 8
  LeanTest.assertTrue (allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "StiefelLinear weight should remain on Stiefel after dual-map update"

@[test]
def testGrassmannLinearConstraintAfterUpdate : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Grassmann) 6 12 true
  let gW ← randn #[12, 6] false
  let gB ← randn #[12] false
  let layer' := ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02

  let w := MatrixManifoldCarrier.toMatrix layer'.weight
  let wtW := nn.mm (nn.transpose2d w) w
  let I := eye 6
  LeanTest.assertTrue (allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "GrassmannLinear representative should preserve X^T X ≈ I"

@[test]
def testModularBudgetBridgeProducesFiniteMultiplier : IO Unit := do
  let l1 ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 false
  let l2 ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 16 16 false
  let modules : Array (StiefelLinear 8 16 × StiefelLinear 16 16) := #[(l1, l2)]

  let merged : Array (Sequential (StiefelLinear 8 16) (StiefelLinear 16 16)) :=
    modules.map (fun (a, b) => Sequential.mk' a b)
  let scales := sequentialDownstreamScales {} merged
  LeanTest.assertTrue (!scales.isEmpty) "Expected non-empty modular scale array"
  for s in scales do
    LeanTest.assertTrue (Float.isFinite s && s > 0.0)
      s!"Expected positive finite modular scale, got {s}"

  let baseCfg : torch.Optim.DualOptimizer.Config := {}
  let cfg' := applyMatrixBudgetFromModules baseCfg {} merged
  LeanTest.assertTrue cfg'.useModularBudget
    "applyMatrixBudgetFromModules should enable modular budgeting"
  LeanTest.assertTrue (Float.isFinite cfg'.budget.matrix && cfg'.budget.matrix > 0.0)
    s!"Expected positive finite matrix budget multiplier, got {cfg'.budget.matrix}"

@[test]
def testManifoldLinearBenchmarkPositive : IO Unit := do
  let mut layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 32 64 false
  let t0 ← IO.monoNanosNow
  for _ in [:3] do
    let gW ← randn #[64, 32] false
    layer := ManifoldLinear.applyDualMapUpdate layer gW none 0.01
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  LeanTest.assertTrue (elapsedMs > 0.0)
    s!"Expected positive benchmark elapsed time, got {elapsedMs}"

@[test]
def testHyperbolicVectorConstraintAfterUpdate : IO Unit := do
  let p0 ← ManifoldVectorParam.init (V := Tyr.AD.Hyperbolic) 6
  let g ← randn #[7] false
  let p1 := ManifoldVectorParam.applyDualMapUpdate p0 g 0.02
  let coords := VectorManifoldCarrier.toVector p1.value
  let inner := Tyr.AD.Hyperbolic.minkowskiInner coords coords
  LeanTest.assertTrue (Float.abs (inner + 1.0) < 1e-4)
    s!"HyperbolicVector should preserve <x,x>_L = -1 after update, got {inner}"

@[test]
def testHyperbolicVectorBenchmarkPositive : IO Unit := do
  let mut p ← ManifoldVectorParam.init (V := Tyr.AD.Hyperbolic) 8
  let t0 ← IO.monoNanosNow
  for _ in [:3] do
    let g ← randn #[9] false
    p := ManifoldVectorParam.applyDualMapUpdate p g 0.01
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  LeanTest.assertTrue (elapsedMs > 0.0)
    s!"Expected positive hyperbolic-vector benchmark elapsed time, got {elapsedMs}"

end Tests.ModularManifold
