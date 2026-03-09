import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Core

namespace Tests.DiffEqAdjointParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) <= tol

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

  -- Valid for `f(t, y, a) = a * y`.
  local instance : AdjointBackend Float Float where
    vjp vf t y a adjY :=
      let f := vf t y a
      let gradY := a * adjY
      let gradA := y * adjY
      (f, gradY, gradA)

  /-
  Finite-difference fallback backend for direct/forward adjoint mode tests.
  This keeps test coverage independent of a torch-backed autodiff runtime.
  -/
  local instance : AdjointFnBackend Float Float where
    vjpFn vf y a adjY :=
      let eps : Float := 1.0e-6
      let gradY := ((vf (y + eps) a) - (vf (y - eps) a)) / (2.0 * eps)
      let gradA := ((vf y (a + eps)) - (vf y (a - eps))) / (2.0 * eps)
      (gradY * adjY, gradA * adjY)

  @[test] def testForwardModePIDControllerWithoutDt0IsUnsupported : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Dopri5.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let controller : PIDController := { rtol := 1.0e-6, atol := 1.0e-8 }
    let (solAdj, adjOpt, errOpt) :=
      diffeqsolveForwardModeWithController
        (Controller := PIDController)
        (mode := {})
        term solver 0.0 1.0 none 2.0 0.3 1.0
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

  @[test] def testForwardModeRequireDt0RejectsMissingDt0 : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let mode : ForwardMode := { requireDt0 := true }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveForwardMode
        (mode := mode)
        term solver 0.0 1.0 none 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ForwardMode should reject missing dt0 when requireDt0=true"
    LeanTest.assertTrue adjOpt.isNone
      "ForwardMode requireDt0 failure should not return adjoint values"
    LeanTest.assertTrue (getStat "adjoint_error" sol.stats == 1)
      "ForwardMode requireDt0 failure should set adjoint_error stat"
    LeanTest.assertTrue (getStat "unsupported_forward_mode" sol.stats == 1)
      "ForwardMode requireDt0 failure should set unsupported_forward_mode stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue
          (msg.startsWith "Adjoint solve failed: `ForwardMode.requireDt0 := true`")
          s!"Expected ForwardMode requireDt0 message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for missing ForwardMode dt0"

  @[test] def testDirectAdjointDegenerateIntervalMissingDt0Succeeds : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let (sol, adjOpt, errOpt) :=
      diffeqsolveDirectAdjointMode
        (mode := {})
        term solver 1.0 1.0 none 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.successful)
      "DirectAdjoint should succeed on a degenerate interval without consulting controller dt0 requirements"
    LeanTest.assertTrue errOpt.isNone
      "DirectAdjoint degenerate interval should not report unsupported errors"
    LeanTest.assertTrue (getStat "unsupported_direct_adjoint_mode" sol.stats == 0)
      "DirectAdjoint degenerate interval should not set unsupported_direct_adjoint_mode"
    let y1 ← getY1 "direct degenerate interval" sol
    LeanTest.assertTrue (approx y1 2.0 1.0e-12)
      s!"DirectAdjoint degenerate interval should preserve y0, got {y1}"
    match adjOpt with
    | none =>
        LeanTest.fail "Expected adjoint values for DirectAdjoint on a degenerate interval"
    | some adj =>
        LeanTest.assertTrue (approx adj.adjY0 1.0 2.0e-3)
          s!"DirectAdjoint degenerate interval adjY0 should be identity, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs 0.0 2.0e-3)
          s!"DirectAdjoint degenerate interval adjArgs should vanish, got {adj.adjArgs}"

  @[test] def testDirectAdjointConstantStepUsesDiscreteLoopVjp : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Euler.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.01
    let y0 := 2.0
    let a := 0.3
    let adjY1 := 1.0
    let (sol, adjOpt, errOpt) :=
      diffeqsolveDirectAdjointMode
        (mode := {})
        term solver t0 t1 dt0 y0 a adjY1
        (saveat := { t1 := true })
    LeanTest.assertTrue errOpt.isNone
      "DirectAdjoint discrete-loop path should not report unsupported errors"
    LeanTest.assertTrue (sol.result == Result.successful)
      "DirectAdjoint should succeed on the discrete loop path"
    let eps : Float := 1.0e-6
    let solYp :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 (y0 + eps) a
        (saveat := { t1 := true })
    let solYm :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 (y0 - eps) a
        (saveat := { t1 := true })
    let solAp :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 (a + eps)
        (saveat := { t1 := true })
    let solAm :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 (a - eps)
        (saveat := { t1 := true })
    let yYp ← getY1 "direct loop y0+eps" solYp
    let yYm ← getY1 "direct loop y0-eps" solYm
    let yAp ← getY1 "direct loop a+eps" solAp
    let yAm ← getY1 "direct loop a-eps" solAm
    let fdY0 := (yYp - yYm) / (2.0 * eps)
    let fdA := (yAp - yAm) / (2.0 * eps)
    match adjOpt with
    | none =>
        LeanTest.fail "Expected DirectAdjoint result on discrete-loop path"
    | some adj =>
        LeanTest.assertTrue (approx adj.adjY0 fdY0 2.0e-3)
          s!"DirectAdjoint discrete-loop dy0 expected {fdY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs fdA 2.0e-3)
          s!"DirectAdjoint discrete-loop da expected {fdA}, got {adj.adjArgs}"

  @[test] def testForwardModeConstantStepUsesDiscreteLoopVjp : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.05
    let y0 := 2.0
    let a := 0.3
    let adjY1 := 1.0
    let (sol, adjOpt, errOpt) :=
      diffeqsolveForwardMode
        (mode := {})
        term solver t0 t1 dt0 y0 a adjY1
        (saveat := { t1 := true })
    LeanTest.assertTrue errOpt.isNone
      "ForwardMode discrete-loop path should not report unsupported errors"
    LeanTest.assertTrue (sol.result == Result.successful)
      "ForwardMode should succeed on the discrete loop path"
    let eps : Float := 1.0e-6
    let solYp :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 (y0 + eps) a
          (saveat := { t1 := true })
      let solYm :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 (y0 - eps) a
          (saveat := { t1 := true })
      let solAp :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 (a + eps)
          (saveat := { t1 := true })
      let solAm :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 (a - eps)
          (saveat := { t1 := true })
    let yYp ← getY1 "forward loop y0+eps" solYp
    let yYm ← getY1 "forward loop y0-eps" solYm
    let yAp ← getY1 "forward loop a+eps" solAp
    let yAm ← getY1 "forward loop a-eps" solAm
    let fdY0 := (yYp - yYm) / (2.0 * eps)
    let fdA := (yAp - yAm) / (2.0 * eps)
    match adjOpt with
    | none =>
        LeanTest.fail "Expected ForwardMode result on discrete-loop path"
    | some adj =>
        LeanTest.assertTrue (approx adj.adjY0 fdY0 5.0e-3)
          s!"ForwardMode discrete-loop dy0 expected {fdY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs fdA 5.0e-3)
          s!"ForwardMode discrete-loop da expected {fdA}, got {adj.adjArgs}"

  @[test] def testDirectAdjointStepToUsesDiscreteLoopVjp : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let controller : StepTo := { ts := #[0.0, 0.1, 0.23, 0.6, 1.0] }
    let t0 := 0.0
    let t1 := 1.0
    let y0 := 2.0
    let a := 0.3
    let adjY1 := 1.0
    let (sol, adjOpt, errOpt) :=
      diffeqsolveDirectAdjointModeWithController
        (Controller := StepTo)
        (mode := {})
        term solver t0 t1 none y0 a adjY1
        (saveat := { t1 := true })
        (controller := controller)
    LeanTest.assertTrue errOpt.isNone
      "DirectAdjoint StepTo path should not report unsupported errors"
    LeanTest.assertTrue (sol.result == Result.successful)
      "DirectAdjoint StepTo path should succeed"
    let eps : Float := 1.0e-6
    let solYp :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none (y0 + eps) a
          (saveat := { t1 := true })
          (controller := controller)
      let solYm :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none (y0 - eps) a
          (saveat := { t1 := true })
          (controller := controller)
      let solAp :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none y0 (a + eps)
          (saveat := { t1 := true })
          (controller := controller)
      let solAm :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none y0 (a - eps)
          (saveat := { t1 := true })
          (controller := controller)
    let yYp ← getY1 "direct StepTo y0+eps" solYp
    let yYm ← getY1 "direct StepTo y0-eps" solYm
    let yAp ← getY1 "direct StepTo a+eps" solAp
    let yAm ← getY1 "direct StepTo a-eps" solAm
    let fdY0 := (yYp - yYm) / (2.0 * eps)
    let fdA := (yAp - yAm) / (2.0 * eps)
    match adjOpt with
    | none =>
        LeanTest.fail "Expected DirectAdjoint result on StepTo discrete-loop path"
    | some adj =>
        LeanTest.assertTrue (approx adj.adjY0 fdY0 5.0e-3)
          s!"DirectAdjoint StepTo dy0 expected {fdY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs fdA 5.0e-3)
          s!"DirectAdjoint StepTo da expected {fdA}, got {adj.adjArgs}"

  @[test] def testForwardModeStepToUsesDiscreteLoopVjp : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Euler.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let controller : StepTo := { ts := #[0.0, 0.08, 0.21, 0.59, 1.0] }
    let t0 := 0.0
    let t1 := 1.0
    let y0 := 2.0
    let a := 0.3
    let adjY1 := 1.0
    let (sol, adjOpt, errOpt) :=
      diffeqsolveForwardModeWithController
        (Controller := StepTo)
        (mode := {})
        term solver t0 t1 none y0 a adjY1
        (saveat := { t1 := true })
        (controller := controller)
    LeanTest.assertTrue errOpt.isNone
      "ForwardMode StepTo path should not report unsupported errors"
    LeanTest.assertTrue (sol.result == Result.successful)
      "ForwardMode StepTo path should succeed"
    let eps : Float := 1.0e-6
    let solYp :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none (y0 + eps) a
          (saveat := { t1 := true })
          (controller := controller)
      let solYm :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none (y0 - eps) a
          (saveat := { t1 := true })
          (controller := controller)
      let solAp :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none y0 (a + eps)
          (saveat := { t1 := true })
          (controller := controller)
      let solAm :=
        diffeqsolve
          (Term := ODETerm Float Float)
          (Y := Float)
          (VF := Float)
          (Control := Time)
          (Args := Float)
          (Controller := StepTo)
          term solver t0 t1 none y0 (a - eps)
          (saveat := { t1 := true })
          (controller := controller)
    let yYp ← getY1 "forward StepTo y0+eps" solYp
    let yYm ← getY1 "forward StepTo y0-eps" solYm
    let yAp ← getY1 "forward StepTo a+eps" solAp
    let yAm ← getY1 "forward StepTo a-eps" solAm
    let fdY0 := (yYp - yYm) / (2.0 * eps)
    let fdA := (yAp - yAm) / (2.0 * eps)
    match adjOpt with
    | none =>
        LeanTest.fail "Expected ForwardMode result on StepTo discrete-loop path"
    | some adj =>
        LeanTest.assertTrue (approx adj.adjY0 fdY0 5.0e-3)
          s!"ForwardMode StepTo dy0 expected {fdY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs fdA 5.0e-3)
          s!"ForwardMode StepTo da expected {fdA}, got {adj.adjArgs}"

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

  @[test] def testImplicitAdjointAllowsEmptyTsSaveAt : IO Unit := do
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
        (saveat := { t1 := true, ts := some #[] })
    LeanTest.assertTrue (sol.result == Result.successful)
      "ImplicitAdjoint should treat `saveat.ts := #[]` like no `ts` and still succeed"
    LeanTest.assertTrue errOpt.isNone
      "ImplicitAdjoint empty-ts saveat run should not report unsupported errors"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint_saveat" sol.stats == 0)
      "ImplicitAdjoint empty-ts saveat run should not mark unsupported_implicit_adjoint_saveat stat"
    match adjOpt with
    | some _ => pure ()
    | none => LeanTest.fail "Expected adjoint result for implicit saveat config with empty ts"

  @[test] def testImplicitAdjointWithoutFallbackAllowsGeneralSaveAt : IO Unit := do
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
    let saveat : SaveAt := { t0 := true, t1 := true, ts := some #[0.25, 0.5, 0.75] }
    let (solAdj, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (mode := mode)
        (Controller := ConstantStepSize)
        term solver  adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := saveat)
    let solBase :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver 0.0 1.0 (some 0.01) 2.0 0.3
        (saveat := saveat)
    LeanTest.assertTrue (solAdj.result == Result.successful)
      "ImplicitAdjoint without fallback should allow general saveat payloads"
    LeanTest.assertTrue errOpt.isNone
      "ImplicitAdjoint no-fallback general saveat path should not report unsupported errors"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint_saveat" solAdj.stats == 0)
      "ImplicitAdjoint no-fallback general saveat path should not tag unsupported_implicit_adjoint_saveat"
    match solAdj.ts, solBase.ts, solAdj.ys, solBase.ys with
    | some tsAdj, some tsBase, some ysAdj, some ysBase =>
        LeanTest.assertTrue (tsAdj == tsBase)
          s!"ImplicitAdjoint no-fallback ts mismatch: {tsAdj} vs {tsBase}"
        LeanTest.assertTrue (ysAdj.size == ysBase.size)
          s!"ImplicitAdjoint no-fallback ys size mismatch: {ysAdj.size} vs {ysBase.size}"
        for i in [:ysAdj.size] do
          LeanTest.assertTrue (Float.abs (ysAdj[i]! - ysBase[i]!) <= 1.0e-12)
            s!"ImplicitAdjoint no-fallback saved y mismatch at {i}: {ysAdj[i]!} vs {ysBase[i]!}"
    | _, _, _, _ =>
        LeanTest.fail "Expected matching ts/ys payloads for general implicit saveat"
    match adjOpt with
    | none =>
        LeanTest.fail "Expected adjoint result for ImplicitAdjoint without fallback and general saveat"
    | some adj =>
        let expectedY0 := Float.exp 0.3
        let expectedA := 2.0 * Float.exp 0.3
        LeanTest.assertTrue (Float.abs (adj.adjY0 - expectedY0) <= 5.0e-2)
          s!"ImplicitAdjoint general-saveat adjY0 expected ~{expectedY0}, got {adj.adjY0}"
        LeanTest.assertTrue (Float.abs (adj.adjArgs - expectedA) <= 8.0e-2)
          s!"ImplicitAdjoint general-saveat adjArgs expected ~{expectedA}, got {adj.adjArgs}"

  @[test] def testBacksolveAdjointRejectsSubSaveAt : IO Unit := do
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
    let nested : SubSaveAt := { t1 := true }
    let (sol, adjOpt) :=
      diffeqsolveAdjoint
        (Controller := ConstantStepSize)
        term solver adjSolver 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := { t1 := false, subs := #[nested] })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "Backsolve adjoint should reject nested SaveAt.subs payloads"
    LeanTest.assertTrue adjOpt.isNone
      "Backsolve adjoint SaveAt.subs contract failure should not return adjoint values"

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
    let mode : ImplicitAdjoint := {
      useBacksolveFallback := false
      recursiveCheckpoint := none
    }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (mode := mode)
        (Controller := ConstantStepSize)
        term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.successful)
      "ImplicitAdjoint without fallback should use solve-function adjoints"
    LeanTest.assertTrue errOpt.isNone
      "ImplicitAdjoint without fallback should not report unsupported errors"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint" sol.stats == 0)
      "ImplicitAdjoint no-fallback solve-function path should not set unsupported_implicit_adjoint"
    match adjOpt with
    | none =>
        LeanTest.fail "Expected adjoint values for ImplicitAdjoint without fallback"
    | some adj =>
        let expectedY0 := Float.exp 0.3
        let expectedA := 2.0 * Float.exp 0.3
        LeanTest.assertTrue (Float.abs (adj.adjY0 - expectedY0) <= 5.0e-2)
          s!"ImplicitAdjoint no-fallback adjY0 expected ~{expectedY0}, got {adj.adjY0}"
        LeanTest.assertTrue (Float.abs (adj.adjArgs - expectedA) <= 8.0e-2)
          s!"ImplicitAdjoint no-fallback adjArgs expected ~{expectedA}, got {adj.adjArgs}"

  @[test] def testImplicitAdjointRecursiveRequiresFallback : IO Unit := do
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
      recursiveCheckpoint := some {}
    }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveImplicitAdjoint
        (mode := mode)
        (Controller := ConstantStepSize)
        term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ImplicitAdjoint recursive-checkpoint mode should require backsolve fallback"
    LeanTest.assertTrue adjOpt.isNone
      "ImplicitAdjoint recursive-checkpoint contract failure should not return adjoint values"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint" sol.stats == 1)
      "ImplicitAdjoint recursive-checkpoint contract failure should set unsupported_implicit_adjoint stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue
          (msg.startsWith "Adjoint solve failed: `ImplicitAdjoint.recursiveCheckpoint` requires")
          s!"Expected ImplicitAdjoint recursive fallback contract message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for ImplicitAdjoint recursive fallback contract"

  @[test] def testRecursiveCheckpointAdjointRejectsBacksolveIncompatibleSaveAt : IO Unit := do
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
    let nested : SubSaveAt := { t1 := true }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveRecursiveCheckpointAdjoint
        (Controller := ConstantStepSize)
        (mode := {})
        term solver adjoint 0.0 1.0 (some 0.01) 2.0 0.3 1.0
        (saveat := { t1 := false, subs := #[nested] })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "RecursiveCheckpointAdjoint should reject backsolve-incompatible SaveAt payloads"
    LeanTest.assertTrue adjOpt.isNone
      "RecursiveCheckpointAdjoint contract failure should not return adjoint values"
    LeanTest.assertTrue (getStat "unsupported_recursive_checkpoint_backsolve_contract" sol.stats == 1)
      "RecursiveCheckpointAdjoint contract failure should set unsupported_recursive_checkpoint_backsolve_contract stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue
          (msg.startsWith "Adjoint solve failed: backsolve-style adjoints do not yet support nested")
          s!"Expected RecursiveCheckpointAdjoint backsolve-contract message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for RecursiveCheckpointAdjoint saveat contract failure"

end

def run : IO Unit := do
  testForwardModePIDControllerWithoutDt0IsUnsupported
  testForwardModeRequireDt0RejectsMissingDt0
  testDirectAdjointDegenerateIntervalMissingDt0Succeeds
  testDirectAdjointConstantStepUsesDiscreteLoopVjp
  testForwardModeConstantStepUsesDiscreteLoopVjp
  testImplicitAdjointRejectsNonT1SaveAt
  testImplicitAdjointDefaultSaveAtStillWorks
  testImplicitAdjointAllowsEmptyTsSaveAt
  testImplicitAdjointWithoutFallbackAllowsGeneralSaveAt
  testBacksolveAdjointRejectsSubSaveAt
  testImplicitAdjointWithoutFallbackUsesSolveFnAdjoint
  testImplicitAdjointRecursiveRequiresFallback
  testRecursiveCheckpointAdjointRejectsBacksolveIncompatibleSaveAt

end Tests.DiffEqAdjointParity
