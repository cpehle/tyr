/-
  Tests/TestManifoldMuon.lean

  Tests for the experimental manifold optimizer stack:
  - Dual-map geometry behavior
  - Stiefel manifold-Muon step invariants
  - Small benchmark smoke check
-/
import LeanTest
import Tyr.Manifolds
import Tyr.Optim.ManifoldMuon

namespace Tests.ManifoldMuon

open LeanTest
open Tyr.AD
open Tyr.AD.DifferentiableManifold
open torch
open torch.Optim.ManifoldMuon

@[test]
def testFloatDualMapDiffersFromRiemannian : IO Unit := do
  let x : Float := 0.0
  let g : Float := 3.0
  let lr : Float := 0.1

  let xRiemann := gradientStep x g lr
  let xDual := DualMapGeometry.dualMapStep x g lr

  -- For Float sign-dual-map: xDual should move by lr, not lr*|g|.
  LeanTest.assertTrue (Float.abs (xRiemann - (-0.3)) < 1e-6)
    s!"Expected Euclidean step -0.3, got {xRiemann}"
  LeanTest.assertTrue (Float.abs (xDual - (-0.1)) < 1e-6)
    s!"Expected sign-dual step -0.1, got {xDual}"

@[test]
def testTangentProjectionConstraint : IO Unit := do
  let W := (Stiefel.identity 4).matrix
  let G ← randn #[4, 4] false
  let Z := tangentProject W G

  let lhs := nn.mm (nn.transpose2d W) Z + nn.mm (nn.transpose2d Z) W
  let zero := zeros_like lhs

  LeanTest.assertTrue (allclose lhs zero (rtol := 1e-4) (atol := 1e-5))
    "Projected update should satisfy Stiefel tangent constraint"

@[test]
def testStiefelConstraintAfterStep : IO Unit := do
  let W0raw ← randn #[8, 4] false
  let W0 := autograd.set_requires_grad (autograd.detach (retractStiefel W0raw)) true
  let G ← randn #[8, 4] false

  let st := initParamState W0
  let cfg : Config := {
    lr := 0.02
    momentum := 0.95
    numIters := 3
    dualAscentSteps := 1
    dualAscentLr := 0.1
    distributed := false
  }
  let (W1, _st1) ← stepSingle W0 G st cfg

  let wtW := nn.mm (nn.transpose2d W1) W1
  let I := eye 4

  LeanTest.assertTrue (allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "ManifoldMuon step should retract back to Stiefel (W^T W ≈ I)"

@[test]
def testManifoldMuonBenchmarkPositive : IO Unit := do
  let avgMs ← benchmarkLocalStep (m := 32) (n := 16) 3
  LeanTest.assertTrue (avgMs > 0.0) s!"Expected positive average step time, got {avgMs}"

end Tests.ManifoldMuon
