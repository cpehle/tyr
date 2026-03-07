import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqInterpolationParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertApprox (label : String) (actual expected tol : Float) : IO Unit :=
  LeanTest.assertTrue (approx actual expected tol)
    s!"{label}: expected {expected}, got {actual}"

private def finalSavedValue {S C : Type}
    (label : String) (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[ys.size - 1]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure 0.0
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure 0.0

@[test] def testLinearPathEndpointIncrementAndDerivativeParity : IO Unit := do
  let path := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let y0 := path.evaluate path.t0 none true
  let y1 := path.evaluate path.t1 none true
  assertApprox "Linear path endpoint t0" y0 1.0 1e-12
  assertApprox "Linear path endpoint t1" y1 5.0 1e-12

  let inc01 := path.evaluate path.t0 (some path.t1) true
  assertApprox "Linear path full increment equals endpoint difference" inc01 (y1 - y0) 1e-12

  let tA : Time := 0.35
  let tB : Time := 1.45
  let incAB := path.evaluate tA (some tB) true
  let diffAB := path.evaluate tB none true - path.evaluate tA none true
  assertApprox "Linear path increment consistency" incAB diffAB 1e-12

  match path.derivative 0.73 true with
  | some d =>
      let eps : Time := 1e-4
      let fd := (path.evaluate (0.73 + eps) none true - path.evaluate (0.73 - eps) none true) / (2.0 * eps)
      assertApprox "Linear path derivative finite-difference sanity" d fd 1e-9
  | none =>
      LeanTest.fail "Linear path should expose derivative"

@[test] def testCubicPathEndpointIncrementAndDerivativeParity : IO Unit := do
  let path :=
    AbstractPath.cubicHermiteInterpolation 0.0 1.0
      (0.0 : Float) (1.0 : Float) (0.0 : Float) (0.0 : Float)
  let y0 := path.evaluate path.t0 none true
  let y1 := path.evaluate path.t1 none true
  assertApprox "Cubic path endpoint t0" y0 0.0 1e-12
  assertApprox "Cubic path endpoint t1" y1 1.0 1e-12

  let tA : Time := 0.25
  let tB : Time := 0.75
  let incAB := path.evaluate tA (some tB) true
  let diffAB := path.evaluate tB none true - path.evaluate tA none true
  assertApprox "Cubic path increment consistency" incAB diffAB 1e-12

  match path.derivative 0.4 true with
  | some d =>
      let eps : Time := 1e-4
      let fd := (path.evaluate (0.4 + eps) none true - path.evaluate (0.4 - eps) none true) / (2.0 * eps)
      assertApprox "Cubic path derivative finite-difference sanity" d fd 1e-5
  | none =>
      LeanTest.fail "Cubic Hermite path should expose derivative"

@[test] def testDenseSolutionEndpointIncrementAndDerivativeParity : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t y _ => -y }
  let solver :=
    Midpoint.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (1.0 : Float) ()
      (saveat := { dense := true, t1 := true })
  LeanTest.assertTrue (sol.result == Result.successful)
    "Dense Midpoint solve should succeed"

  let yStart := sol.evaluate 0.0
  assertApprox "Dense solution endpoint t0" yStart 1.0 1e-12

  let yEndDense := sol.evaluate 1.0
  let yEndSaved ← finalSavedValue "Dense Midpoint" sol
  assertApprox "Dense solution endpoint t1 equals saved y(t1)" yEndDense yEndSaved 1e-12

  let tA : Time := 0.35
  let tB : Time := 0.85
  let incAB := sol.evaluate tA (some tB) true
  let diffAB := sol.evaluate tB - sol.evaluate tA
  assertApprox "Dense solution increment consistency" incAB diffAB 1e-10

  let tProbe : Time := 0.37
  let eps : Time := 1e-4
  let d := sol.derivative tProbe
  let fd := (sol.evaluate (tProbe + eps) - sol.evaluate (tProbe - eps)) / (2.0 * eps)
  assertApprox "Dense solution derivative finite-difference sanity" d fd 1e-7

@[test] def testPiecewiseDenseBoundaryLeftRightParity : IO Unit := do
  let segLeft :=
    LocalLinearDenseInfo.toInterpolation
      ({ t0 := 0.0, t1 := 1.0, y0 := (0.0 : Float), y1 := (1.0 : Float) } :
        LocalLinearDenseInfo Float)
  let segRight :=
    LocalLinearDenseInfo.toInterpolation
      ({ t0 := 1.0, t1 := 2.0, y0 := (10.0 : Float), y1 := (12.0 : Float) } :
        LocalLinearDenseInfo Float)
  let interp :=
    PiecewiseDenseInterpolation.toDense
      ({ ts := #[0.0, 1.0, 2.0], segments := #[segLeft, segRight] } :
        PiecewiseDenseInterpolation Float)

  let vLeft := interp.evaluate 1.0 none true
  let vRight := interp.evaluate 1.0 none false
  assertApprox "Piecewise dense left value at knot" vLeft 1.0 1e-12
  assertApprox "Piecewise dense right value at knot" vRight 10.0 1e-12

  let dLeft := interp.derivative 1.0 true
  let dRight := interp.derivative 1.0 false
  assertApprox "Piecewise dense left derivative at knot" dLeft 1.0 1e-12
  assertApprox "Piecewise dense right derivative at knot" dRight 2.0 1e-12

def run : IO Unit := do
  testLinearPathEndpointIncrementAndDerivativeParity
  testCubicPathEndpointIncrementAndDerivativeParity
  testDenseSolutionEndpointIncrementAndDerivativeParity
  testPiecewiseDenseBoundaryLeftRightParity

end Tests.DiffEqInterpolationParity

unsafe def main : IO Unit := do
  Tests.DiffEqInterpolationParity.run
  IO.println "TestDiffEqInterpolationParity: ok"
