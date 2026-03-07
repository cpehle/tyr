import LeanTest
import Tyr.Modular.MetricFactor
import Tyr.Manifolds.Embedded

namespace Tests.MetricFactor

open LeanTest
open torch
open Tyr.Modular
open Tyr.AD

private def vectorToColumn {n : UInt64} (v : T #[n]) : T #[n, 1] :=
  reshape v #[n, 1]

private def columnToVector {n : UInt64} (v : T #[n, 1]) : T #[n] :=
  reshape v #[n]

@[test]
def testMetricFactorPullbackMatchesExplicitMatmul : IO Unit := do
  let lMat ← randn #[3, 4] false
  let a ← randn #[4, 2] false
  let factor : MetricFactor 3 4 := { matrix := lMat }
  let pulled := MetricFactor.pullback factor a
  let expected := nn.mm lMat a
  LeanTest.assertTrue (allclose pulled.matrix expected (rtol := 1e-4) (atol := 1e-5))
    "MetricFactor.pullback should match explicit matrix multiplication"

@[test]
def testWoodburySolveMatchesDenseInverse : IO Unit := do
  let kMat ← randn #[3, 5] false
  let rawDiag ← randn #[5] false
  let diag := add_scalar (nn.abs rawDiag) 1.0
  let g ← randn #[5] false
  let factor : MetricFactor 3 5 := { matrix := kMat }
  let mass : DiagonalMass 5 := { diag := diag }

  let woodbury := MetricFactor.solveWoodbury mass factor g
  let denseInv := linalg.inv (MetricFactor.denseMetric mass factor)
  let dense := columnToVector (nn.mm denseInv (vectorToColumn g))

  LeanTest.assertTrue (allclose woodbury dense (rtol := 1e-4) (atol := 1e-5))
    "Woodbury solve should match dense inverse solve on small problems"

@[test]
def testEmbeddedEuclideanRetractAmbientStepMatchesEuclideanUpdate : IO Unit := do
  let x ← randn #[4] false
  let g ← randn #[4] false
  let lr : Float := 0.1
  let x' := EmbeddedManifold.retractAmbientStep x g lr
  let expected := x - mul_scalar g lr
  LeanTest.assertTrue (allclose x' expected (rtol := 1e-5) (atol := 1e-6))
    "EmbeddedManifold.retractAmbientStep should reduce to Euclidean GD in flat space"

@[test]
def testEmbeddedStiefelRetractAmbientStepPreservesConstraint : IO Unit := do
  let x ← Stiefel.random 6 3
  let g ← randn #[6, 3] false
  let x' := EmbeddedManifold.retractAmbientStep x g 0.05
  let wtW := nn.mm (nn.transpose2d x'.matrix) x'.matrix
  let I := eye 3
  LeanTest.assertTrue (allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "Embedded ambient-step retraction should preserve the Stiefel constraint"

end Tests.MetricFactor
