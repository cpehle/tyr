import LeanTest
import Tyr.DiffEq.Path
import Tyr.DiffEq.Term

namespace Tests.DiffEqPathParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

@[test] def testLinearInterpolationEvaluateParity : IO Unit := do
  let path := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let value := path.evaluate 0.75 none true
  LeanTest.assertTrue (approx value 2.5 1e-12)
    s!"LinearPath.evaluate(t, none) expected 2.5, got {value}"
  let inc := path.evaluate 0.75 (some 1.25) true
  LeanTest.assertTrue (approx inc 1.0 1e-12)
    s!"LinearPath.evaluate(t0, some t1) expected increment 1.0, got {inc}"

@[test] def testCubicHermiteEvaluateParity : IO Unit := do
  let path := AbstractPath.cubicHermiteInterpolation 0.0 1.0
    (0.0 : Float) (1.0 : Float) (0.0 : Float) (0.0 : Float)
  let value := path.evaluate 0.5 none true
  LeanTest.assertTrue (approx value 0.5 1e-12)
    s!"CubicHermitePath.evaluate(t, none) expected 0.5, got {value}"
  let inc := path.evaluate 0.25 (some 0.75) true
  LeanTest.assertTrue (approx inc 0.6875 1e-12)
    s!"CubicHermitePath.evaluate(t0, some t1) expected increment 0.6875, got {inc}"

@[test] def testOfPositionEvaluateParity : IO Unit := do
  let path : AbstractPath Float :=
    AbstractPath.ofPosition 0.0 2.0 (fun t => t * t + 1.0)
  let value := path.evaluate 0.75 none true
  LeanTest.assertTrue (approx value 1.5625 1e-12)
    s!"AbstractPath.ofPosition evaluate(t, none) expected 1.5625, got {value}"
  let inc := path.increment 0.75 1.25
  LeanTest.assertTrue (approx inc 1.0 1e-12)
    s!"AbstractPath.ofPosition increment expected 1.0, got {inc}"

@[test] def testControlTermContrStillUsesIncrement : IO Unit := do
  let path : AbstractPath Float :=
    AbstractPath.ofPosition 0.0 2.0 (fun t => t * t + 1.0)
      (some (fun t _left => 2.0 * t))
  let term : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath (fun _t y _ => y) path (fun vf control => vf * control)
  let dControl := term.control 0.75 1.25
  LeanTest.assertTrue (approx dControl 1.0 1e-12)
    s!"ControlTerm control increment expected 1.0, got {dControl}"
  match term.toODE? with
  | some ode =>
      let vf := ode.vectorField 0.5 2.0 ()
      LeanTest.assertTrue (approx vf 2.0 1e-12)
        s!"ControlTerm.toODE derivative branch expected 2.0, got {vf}"
  | none =>
      LeanTest.fail "Expected ControlTerm.toODE? for differentiable path"

@[test] def testPathIncrementAdditivityAndAntisymmetryParity : IO Unit := do
  let path := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let tA : Time := 0.3
  let tB : Time := 1.1
  let tC : Time := 1.7
  let incAB := path.increment tA tB
  let incBC := path.increment tB tC
  let incAC := path.increment tA tC
  LeanTest.assertTrue (approx incAC (incAB + incBC) 1e-12)
    s!"Path increment additivity failed: incAC={incAC}, incAB+incBC={incAB + incBC}"
  let incBA := path.increment tB tA
  LeanTest.assertTrue (approx incBA (-incAB) 1e-12)
    s!"Path increment antisymmetry failed: incBA={incBA}, -incAB={-incAB}"

@[test] def testPathDerivativeFiniteDifferenceParity : IO Unit := do
  let linear := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let cubic :=
    AbstractPath.cubicHermiteInterpolation 0.0 1.0
      (0.0 : Float) (1.0 : Float) (0.0 : Float) (0.0 : Float)
  let smooth :=
    AbstractPath.ofDifferentiablePosition 0.0 2.0
      (fun t => t * t * t + 1.0)
      (fun t _left => 3.0 * t * t)
  let eps : Time := 1.0e-4

  let check := fun (label : String) (path : AbstractPath Float) (t : Time) => do
    match path.derivative t with
    | some d =>
        let fd :=
          (path.evaluate (t + eps) none true - path.evaluate (t - eps) none true) / (2.0 * eps)
        LeanTest.assertTrue (approx d fd 5.0e-4)
          s!"{label}: derivative/FD mismatch at t={t}: derivative={d}, fd={fd}"
    | none =>
        LeanTest.fail s!"{label}: expected derivative"

  check "linear path" linear 0.73
  check "cubic path" cubic 0.41
  check "ofDifferentiablePosition path" smooth 1.2

@[test] def testPathClearDerivativePreservesValues : IO Unit := do
  let base := AbstractPath.linearInterpolation 0.0 2.0 (1.0 : Float) (5.0 : Float)
  let cleared := base.clearDerivative
  LeanTest.assertTrue cleared.derivativeFn?.isNone
    "clearDerivative should remove derivative metadata"

  let t : Time := 0.8
  let t' : Time := 1.6
  let basePos := base.evaluate t none true
  let clearedPos := cleared.evaluate t none true
  LeanTest.assertTrue (approx basePos clearedPos 1e-12)
    s!"clearDerivative should preserve point evaluation: {basePos} vs {clearedPos}"
  let baseInc := base.increment t t'
  let clearedInc := cleared.increment t t'
  LeanTest.assertTrue (approx baseInc clearedInc 1e-12)
    s!"clearDerivative should preserve increments: {baseInc} vs {clearedInc}"

def run : IO Unit := do
  testLinearInterpolationEvaluateParity
  testCubicHermiteEvaluateParity
  testOfPositionEvaluateParity
  testControlTermContrStillUsesIncrement
  testPathIncrementAdditivityAndAntisymmetryParity
  testPathDerivativeFiniteDifferenceParity
  testPathClearDerivativePreservesValues

end Tests.DiffEqPathParity
