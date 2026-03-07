import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Core

namespace Tests.DiffEqAdjointParity

open LeanTest
open torch
open torch.DiffEq

private def getStat (name : String) (stats : List (String × Nat)) : Nat :=
  match stats.find? (fun kv => kv.fst == name) with
  | some (_, v) => v
  | none => 0

section

  -- Valid for `f(t, y, a) = a * y`.
  local instance : AdjointBackend Float Float where
    vjp vf t y a adjY :=
      let f := vf t y a
      let gradY := a * adjY
      let gradA := y * adjY
      (f, gradY, gradA)

  @[test] def testImplicitAdjointRejectsNonT1SaveAt : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let adjSolver :=
      RK4.solver
        (Term := ODETerm (AdjointState Float Float) Float)
        (Y := AdjointState Float Float)
        (VF := AdjointState Float Float)
        (Args := Float)
    let adjoint : BacksolveAdjoint Float Float := { adjSolver := adjSolver }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (Controller := ConstantStepSize)
        (mode := {})
        term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := { t0 := true, t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ImplicitAdjoint should reject non-`SaveAt(t1=True)` save configs"
    LeanTest.assertTrue adjOpt.isNone
      "ImplicitAdjoint saveat contract failure should not return adjoint values"
    LeanTest.assertTrue (getStat "adjoint_error" sol.stats == 1)
      "ImplicitAdjoint saveat contract failure should mark adjoint_error stat"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint_saveat" sol.stats == 1)
      "ImplicitAdjoint saveat contract failure should mark unsupported_implicit_adjoint_saveat stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue (msg.contains "SaveAt(t1=True)")
          s!"Expected SaveAt(t1=True) contract message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for invalid ImplicitAdjoint saveat config"

  @[test] def testImplicitAdjointDefaultSaveAtStillWorks : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let adjSolver :=
      RK4.solver
        (Term := ODETerm (AdjointState Float Float) Float)
        (Y := AdjointState Float Float)
        (VF := AdjointState Float Float)
        (Args := Float)
    let adjoint : BacksolveAdjoint Float Float := { adjSolver := adjSolver }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (Controller := ConstantStepSize)
        (mode := {})
        term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.successful)
      "ImplicitAdjoint should still succeed with default `saveat := { t1 := true }`"
    LeanTest.assertTrue errOpt.isNone
      "ImplicitAdjoint default saveat run should not report unsupported errors"
    match adjOpt with
    | some _ => pure ()
    | none => LeanTest.fail "Expected adjoint result for default implicit saveat config"

end

def run : IO Unit := do
  testImplicitAdjointRejectsNonT1SaveAt
  testImplicitAdjointDefaultSaveAtStillWorks

end Tests.DiffEqAdjointParity
