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

def run : IO Unit := do
  testLinearInterpolationEvaluateParity
  testCubicHermiteEvaluateParity
  testOfPositionEvaluateParity
  testControlTermContrStillUsesIncrement

end Tests.DiffEqPathParity
