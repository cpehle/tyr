/-
  Tests/TestDualOptimizerDispatch.lean

  Focused tests for DualOptimizer matrix backend dispatch and composable
  matrix backend operation bundles.
-/
import LeanTest
import Tyr.Optim.DualOptimizer

namespace Tests.DualOptimizerDispatch

open LeanTest
open torch
open torch.Optim.DualOptimizer

@[test]
def testMatrixStateInitializationDispatch : IO Unit := do
  let p ← randn #[8, 4] false

  let cfgNor : Config := { matrixOptimizer := .norMuon }
  let stNor := initMatrixParamState cfgNor p
  match stNor with
  | .norMuon _ => pure ()
  | _ => LeanTest.fail "Expected NorMuon state for norMuon backend"

  let cfgStiefel : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .stiefel
    preferGenericManifoldPath := false
  }
  LeanTest.assertTrue (usesSpecializedStiefelPath cfgStiefel)
    "Expected specialized Stiefel path to be enabled"
  let stStiefel := initMatrixParamState cfgStiefel p
  match stStiefel with
  | .manifoldMuon _ => pure ()
  | _ => LeanTest.fail "Expected specialized manifold-Muon state for Stiefel path"

  let cfgGrassmann : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
  }
  let stGrass := initMatrixParamState cfgGrassmann p
  match stGrass with
  | .genericManifold _ => pure ()
  | _ => LeanTest.fail "Expected generic manifold state for non-Stiefel manifold path"

@[test]
def testStepMatrixSingleStiefelConstraint : IO Unit := do
  let pRaw ← randn #[8, 4] false
  let p := autograd.set_requires_grad
    (autograd.detach (torch.Optim.ManifoldMuon.retractStiefel pRaw)) true
  let g ← randn #[8, 4] false

  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .stiefel
    preferGenericManifoldPath := false
    matrixLr := 0.02
  }
  let st0 := initMatrixParamState cfg p
  let (p1, st1) ← stepMatrixSingle p g st0 cfg 0 1.0 1.0

  let wtW := nn.mm (nn.transpose2d p1) p1
  let I := eye 4
  LeanTest.assertTrue (allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "Specialized Stiefel path should preserve W^T W ≈ I"
  match st1 with
  | .manifoldMuon _ => pure ()
  | _ => LeanTest.fail "Expected specialized manifold-Muon state after Stiefel step"

@[test]
def testStepMatrixSingleGrassmannConstraint : IO Unit := do
  let pRaw ← randn #[9, 3] false
  let p := autograd.set_requires_grad
    (autograd.detach (Tyr.AD.Grassmann.project 9 3 pRaw).matrix) true
  let g ← randn #[9, 3] false

  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
    matrixLr := 0.02
  }
  let st0 := initMatrixParamState cfg p
  let (p1, st1) ← stepMatrixSingle p g st0 cfg 0 1.0 1.0

  let xtx := nn.mm (nn.transpose2d p1) p1
  let I := eye 3
  LeanTest.assertTrue (allclose xtx I (rtol := 1e-4) (atol := 1e-5))
    "Generic Grassmann path should preserve X^T X ≈ I"
  match st1 with
  | .genericManifold _ => pure ()
  | _ => LeanTest.fail "Expected generic manifold state after Grassmann step"

@[test]
def testMatrixBackendOpsComposable : IO Unit := do
  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
    matrixLr := 0.01
  }
  let ops := matrixBackendOps (m := 12) (n := 4) cfg 0 1.0 1.0

  let p0Raw ← randn #[12, 4] false
  let p1Raw ← randn #[12, 4] false
  let p0 := autograd.set_requires_grad
    (autograd.detach (Tyr.AD.Grassmann.project 12 4 p0Raw).matrix) true
  let p1 := autograd.set_requires_grad
    (autograd.detach (Tyr.AD.Grassmann.project 12 4 p1Raw).matrix) true
  let g0 ← randn #[12, 4] false
  let g1 ← randn #[12, 4] false

  let s0 := ops.initState p0
  let s1 := ops.initState p1
  let (ps', ss') ← ops.stepGroupLocal #[p0, p1] #[g0, g1] #[s0, s1]

  LeanTest.assertEqual ps'.size 2
  LeanTest.assertEqual ss'.size 2

@[test]
def testMatrixBackendOpsBenchmarkPositive : IO Unit := do
  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
    matrixLr := 0.01
  }
  let mut p := (Tyr.AD.Grassmann.project 20 6 (← randn #[20, 6] false)).matrix
  p := autograd.set_requires_grad (autograd.detach p) true
  let mut st := initMatrixParamState cfg p

  let t0 ← IO.monoNanosNow
  for _ in [:3] do
    let g ← randn #[20, 6] false
    let (p', st') ← stepMatrixSingle p g st cfg 0 1.0 1.0
    p := p'
    st := st'
  let t1 ← IO.monoNanosNow

  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  LeanTest.assertTrue (elapsedMs > 0.0)
    s!"Expected positive benchmark elapsed time, got {elapsedMs}ms"

end Tests.DualOptimizerDispatch
