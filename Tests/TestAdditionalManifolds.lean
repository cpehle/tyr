import LeanTest
import Tyr.Manifolds.Product
import Tyr.Manifolds.Sphere
import Tyr.Manifolds.Oblique
import Tyr.Manifolds.Positive
import Tyr.Manifolds.PoincareBall
import Tyr.Manifolds.PositiveDefinite
import Tyr.Manifolds.PSD
import Tyr.Manifolds.FixedRank

namespace Tests.AdditionalManifolds

open LeanTest
open torch
open Tyr.AD

@[test]
def testSphereProjectionKeepsUnitNorm : IO Unit := do
  let x0 ← randn #[16] false
  let x := Sphere.project 16 x0
  let n := linalg.l2Norm x.coords
  LeanTest.assertTrue (Float.abs (n - 1.0) < 1e-5)
    s!"Expected sphere norm 1, got {n}"

@[test]
def testObliqueColumnsUnit : IO Unit := do
  let w0 ← randn #[12, 5] false
  let w := Oblique.project 12 5 w0
  let wtW := nn.mm (nn.transpose2d w.matrix) w.matrix
  let d := linalg.diag wtW
  let onesV := ones #[5]
  LeanTest.assertTrue (allclose d onesV (rtol := 1e-4) (atol := 1e-5))
    "Expected oblique columns to have unit norm"

@[test]
def testPositiveProjectionPositiveEntries : IO Unit := do
  let a0 ← randn #[8, 8] false
  let p := Positive.project 8 8 a0
  let minv := nn.item (nn.minAll p.matrix)
  LeanTest.assertTrue (minv > 0.0)
    s!"Expected all entries > 0, min was {minv}"

@[test]
def testPoincareProjectionInsideBall : IO Unit := do
  let x0 ← randn #[10] false
  let x := PoincareBall.project 10 (mul_scalar x0 10.0)
  let n := linalg.l2Norm x.coords
  LeanTest.assertTrue (n < 1.0)
    s!"Expected poincare point norm < 1, got {n}"

@[test]
def testSPDProjectionSymmetric : IO Unit := do
  let a0 ← randn #[6, 6] false
  let x := SymmetricPositiveDefinite.project 6 a0
  let symErr := x.matrix - nn.transpose2d x.matrix
  let e := linalg.frobeniusNorm symErr
  LeanTest.assertTrue (e < 1e-5)
    s!"Expected SPD projection symmetric, error {e}"

@[test]
def testElliptopeRowsUnit : IO Unit := do
  let y0 ← randn #[9, 4] false
  let y := Elliptope.project 9 4 y0
  let rn := linalg.rowNorms y.factor
  let onesV := ones #[9]
  LeanTest.assertTrue (allclose rn onesV (rtol := 1e-4) (atol := 1e-5))
    "Expected elliptope rows to have unit norm"

@[test]
def testFixedRankProjectionIdempotent : IO Unit := do
  let a0 ← randn #[8, 6] false
  let x := FixedRankEmbedded.project 8 6 3 a0
  let fn := linalg.frobeniusNorm x.matrix
  LeanTest.assertTrue (Float.isFinite fn && fn > 0.0)
    s!"Expected finite non-zero fixed-rank projection norm, got {fn}"

@[test]
def testProductGradientStepCompiles : IO Unit := do
  let x0 ← randn #[8] false
  let y0 ← randn #[4, 4] false
  let x : Sphere 8 := Sphere.project 8 x0
  let y : Positive 4 4 := Positive.project 4 4 y0
  let gx : SphereTangent 8 := SphereTangent.project x (zeros #[8])
  let gy : PositiveTangent 4 4 := ⟨zeros #[4, 4]⟩
  let z' := DifferentiableManifold.gradientStep (x, y) (gx, gy) 0.01
  let xn := linalg.l2Norm z'.1.coords
  LeanTest.assertTrue (Float.abs (xn - 1.0) < 1e-5)
    "Product manifold step should keep sphere component on manifold"

end Tests.AdditionalManifolds
