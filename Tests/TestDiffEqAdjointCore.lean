import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Core

namespace Tests.DiffEqAdjointCore

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def getStat (name : String) (stats : List (String × Nat)) : Nat :=
  match stats.find? (fun kv => kv.fst == name) with
  | some (_, v) => v
  | none => 0

private def getY1 {S C : Type} (label : String) (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size == 0 then
        LeanTest.fail s!"Expected y1 for {label}"
        return 0.0
      else
        return ys[ys.size - 1]!
  | none =>
      LeanTest.fail s!"Expected ys for {label}"
      return 0.0

section
  -- Valid for the linear vf used in this test: f(t,y,a) = a*y.
  local instance : AdjointBackend Float Float where
    vjp vf t y a adjY :=
      let f := vf t y a
      let gradY := a * adjY
      let gradA := y * adjY
      (f, gradY, gradA)

  local instance : AdjointFnBackend Float Float where
    vjpFn f y a adjY :=
      let eps := 1.0e-4
      let dy := ((f (y + eps) a) - (f (y - eps) a)) / (2.0 * eps)
      let da := ((f y (a + eps)) - (f y (a - eps))) / (2.0 * eps)
      (dy * adjY, da * adjY)

  @[test] def testBacksolveAdjointLinearODE : IO Unit := do
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
    let y0 := 2.0
    let a := 0.3
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.01
    let sol :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 a (saveat := { steps := true })
    let adjY1 := 1.0
    match backsolveAdjoint term adjSolver sol a adjY1 with
    | none => LeanTest.fail "Expected adjoint result"
    | some adj => do
        let eps := 1.0e-4
        let solp :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 (y0 + eps) a (saveat := { t1 := true })
        let solm :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 (y0 - eps) a (saveat := { t1 := true })
        let y1p ← getY1 "y0+eps" solp
        let y1m ← getY1 "y0-eps" solm
        let fdY0 := (y1p - y1m) / (2.0 * eps)

        let solap :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 y0 (a + eps) (saveat := { t1 := true })
        let solam :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 y0 (a - eps) (saveat := { t1 := true })
        let y1ap ← getY1 "a+eps" solap
        let y1am ← getY1 "a-eps" solam
        let fdA := (y1ap - y1am) / (2.0 * eps)

        LeanTest.assertTrue (approx adj.adjY0 fdY0 1.0e-3)
          s!"adjoint dy0 expected {fdY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs fdA 1.0e-3)
          s!"adjoint da expected {fdA}, got {adj.adjArgs}"

  @[test] def testBacksolveAdjointTerminalOnlySave : IO Unit := do
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
    let y0 := 2.0
    let a := 0.3
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.01
    let sol :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 a (saveat := { t1 := true })
    let adjY1 := 1.0
    match backsolveAdjoint term adjSolver sol a adjY1 with
    | none =>
        LeanTest.fail "Expected adjoint result for terminal-only save"
    | some adj => do
        let exactDy0 := Float.exp (a * (t1 - t0))
        let exactDa := (t1 - t0) * y0 * exactDy0
        LeanTest.assertTrue (approx adj.adjY0 exactDy0 1.0e-2)
          s!"terminal-only backsolve dy0 expected {exactDy0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs exactDa 1.0e-2)
          s!"terminal-only backsolve da expected {exactDa}, got {adj.adjArgs}"

  @[test] def testBacksolveAdjointWrapper : IO Unit := do
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
    let y0 := 2.0
    let a := 0.3
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.01
    let adjY1 := 1.0
    let (solAdj, adj) :=
      diffeqsolveBacksolveAdjoint (Controller := ConstantStepSize)
        term solver adjoint t0 t1 dt0 y0 a adjY1 (saveat := { t1 := true })
    let sol :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 a (saveat := { t1 := true })
    let yAdj ← getY1 "wrapper" solAdj
    let yDir ← getY1 "direct" sol
    LeanTest.assertTrue (approx yAdj yDir 1.0e-6)
      s!"Wrapper solution expected {yDir}, got {yAdj}"
    match adj with
    | some _ => pure ()
    | none => LeanTest.fail "Expected adjoint result from wrapper"

  @[test] def testImplicitAdjointWithoutFallbackUsesSolveFnAdjoint : IO Unit := do
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
    let mode : ImplicitAdjoint := { useBacksolveFallback := false }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (Controller := ConstantStepSize)
        mode term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0 (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.successful)
      "ImplicitAdjoint without fallback should use solve-function adjoints"
    LeanTest.assertTrue errOpt.isNone
      "ImplicitAdjoint without fallback should not report unsupported errors"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint" sol.stats == 0)
      "ImplicitAdjoint solve-function path should not tag unsupported_implicit_adjoint"
    match adjOpt with
    | none =>
        LeanTest.fail "Expected adjoint result from ImplicitAdjoint without fallback"
    | some adj =>
        let expectedY0 := Float.exp 0.3
        let expectedA := 2.0 * Float.exp 0.3
        LeanTest.assertTrue (approx adj.adjY0 expectedY0 5.0e-2)
          s!"ImplicitAdjoint no-fallback adjY0 expected ~{expectedY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs expectedA 8.0e-2)
          s!"ImplicitAdjoint no-fallback adjArgs expected ~{expectedA}, got {adj.adjArgs}"

  @[test] def testImplicitAdjointRecursiveCheckpoint : IO Unit := do
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
    let mode : ImplicitAdjoint := {
      useBacksolveFallback := true
      recursiveCheckpoint := some { checkpointEvery := 2, recomputeSegments := true }
    }
    let (solAdj, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (Controller := ConstantStepSize)
        mode term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0 (saveat := { t1 := true })

    LeanTest.assertTrue (solAdj.result == Result.successful)
      "ImplicitAdjoint recursive checkpoint primal solve should succeed"
    LeanTest.assertTrue errOpt.isNone
      "ImplicitAdjoint recursive checkpoint should not report an error"
    match adjOpt with
    | none => LeanTest.fail "Expected adjoint result from recursive checkpoint mode"
    | some adj =>
        let expectedY0 := Float.exp 0.3
        let expectedA := 2.0 * Float.exp 0.3
        LeanTest.assertTrue (approx adj.adjY0 expectedY0 5.0e-2)
          s!"Recursive checkpoint adjY0 expected ~{expectedY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs expectedA 8.0e-2)
          s!"Recursive checkpoint adjArgs expected ~{expectedA}, got {adj.adjArgs}"

  @[test] def testForwardModePIDControllerWithoutDt0IsUnsupported : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Dopri5.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let controller : PIDController := { rtol := 1.0e-6, atol := 1.0e-8 }
    let y0 := 2.0
    let a := 0.3
    let adjY1 := 1.0
    let (solAdj, adjOpt, errOpt) :=
      diffeqsolveForwardModeWithController
        (Controller := PIDController)
        (mode := {})
        term solver 0.0 1.0 none y0 a adjY1
        (saveat := { t1 := true })
        (controller := controller)
    LeanTest.assertTrue (solAdj.result == Result.internalError)
      "ForwardMode should report unsupported for PIDController until controller adjoints are implemented"
    LeanTest.assertTrue adjOpt.isNone
      "Unsupported ForwardMode PIDController path should not return adjoints"
    LeanTest.assertTrue (getStat "adjoint_error" solAdj.stats == 1)
      "Unsupported ForwardMode PIDController path should set adjoint_error"
    LeanTest.assertTrue (getStat "unsupported_forward_mode" solAdj.stats == 1)
      "Unsupported ForwardMode PIDController path should set unsupported_forward_mode"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue (msg.contains "ForwardMode")
          s!"Expected ForwardMode unsupported message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported ForwardMode PIDController message"

  @[test] def testForwardModeConstantStepMissingDt0DefersToControllerValidation : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Euler.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let (sol, adjOpt, errOpt) :=
      diffeqsolveForwardMode
        (mode := {})
        term solver 0.0 1.0 none 2.0 0.3 1.0 (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ForwardMode with ConstantStepSize and missing dt0 should fail via controller validation"
    LeanTest.assertTrue adjOpt.isNone
      "ForwardMode controller-validation failure should not return adjoints"
    LeanTest.assertTrue errOpt.isNone
      "ForwardMode controller-validation failure should not produce an unsupported-mode error"
    LeanTest.assertTrue (getStat "adjoint_error" sol.stats == 0)
      "ForwardMode controller-validation failure should not mark adjoint_error"
    LeanTest.assertTrue (getStat "unsupported_forward_mode" sol.stats == 0)
      "ForwardMode controller-validation failure should not mark unsupported_forward_mode"

  @[test] def testForwardModeRequireDt0ErrorReporting : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Euler.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let (sol, adjOpt, errOpt) :=
      diffeqsolveForwardMode
        (mode := { requireDt0 := true })
        term solver 0.0 1.0 none 2.0 0.3 1.0 (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ForwardMode without dt0 should fail when requireDt0=true"
    LeanTest.assertTrue adjOpt.isNone
      "ForwardMode unsupported run should not return adjoints"
    LeanTest.assertTrue (getStat "adjoint_error" sol.stats == 1)
      "ForwardMode unsupported run should mark adjoint_error stat"
    LeanTest.assertTrue (getStat "unsupported_forward_mode" sol.stats == 1)
      "ForwardMode unsupported run should mark unsupported_forward_mode stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue (msg.contains "ForwardMode.requireDt0 := true")
          s!"Expected dt0 requirement message for ForwardMode, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for ForwardMode without dt0"

  @[test] def testImplicitAdjointRecursiveNeedsBacksolveFallbackMessage : IO Unit := do
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
    let mode : ImplicitAdjoint := {
      useBacksolveFallback := false
      recursiveCheckpoint := some { checkpointEvery := 4, recomputeSegments := true }
    }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (Controller := ConstantStepSize)
        mode term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0 (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ImplicitAdjoint recursive mode without fallback should report unsupported mode"
    LeanTest.assertTrue adjOpt.isNone
      "ImplicitAdjoint recursive unsupported mode should not return adjoints"
    LeanTest.assertTrue (getStat "adjoint_error" sol.stats == 1)
      "ImplicitAdjoint unsupported run should mark adjoint_error stat"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint" sol.stats == 1)
      "ImplicitAdjoint unsupported run should mark unsupported_implicit_adjoint stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue (msg.contains "requires `useBacksolveFallback := true`")
          s!"Expected recursive fallback requirement message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for recursive implicit adjoint without fallback"
end

end Tests.DiffEqAdjointCore
