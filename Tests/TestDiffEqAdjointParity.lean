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

  @[test] def testForwardModeInfersDt0WhenAllowed : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let mode : ForwardMode := { requireDt0 := false }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveForwardMode
        (mode := mode)
        term solver 0.0 1.0 none 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.successful)
      "ForwardMode should infer dt0 when requireDt0=false"
    LeanTest.assertTrue errOpt.isNone
      "ForwardMode inferred-dt0 path should not return unsupported error"
    LeanTest.assertTrue (getStat "unsupported_forward_mode" sol.stats == 0)
      "ForwardMode inferred-dt0 path should not tag unsupported_forward_mode"
    match adjOpt with
    | some _ => pure ()
    | none => LeanTest.fail "Expected ForwardMode adjoint output when dt0 inference succeeds"

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
          (msg.startsWith "Adjoint solve failed: direct/forward mode currently requires")
          s!"Expected ForwardMode requireDt0 message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for missing ForwardMode dt0"

  @[test] def testDirectAdjointDegenerateIntervalMissingDt0IsUnsupported : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let mode : DirectAdjoint := { requireDt0 := false }
    let (sol, adjOpt, errOpt) :=
      diffeqsolveDirectAdjointMode
        (mode := mode)
        term solver 1.0 1.0 none 2.0 0.3 1.0
        (saveat := { t1 := true })
    LeanTest.assertTrue (sol.result == Result.internalError)
      "DirectAdjoint should fail when dt0 inference is impossible on degenerate interval"
    LeanTest.assertTrue adjOpt.isNone
      "DirectAdjoint degenerate missing-dt0 failure should not return adjoint values"
    LeanTest.assertTrue (getStat "adjoint_error" sol.stats == 1)
      "DirectAdjoint degenerate missing-dt0 failure should set adjoint_error stat"
    LeanTest.assertTrue (getStat "unsupported_direct_adjoint_mode" sol.stats == 1)
      "DirectAdjoint degenerate missing-dt0 failure should set unsupported_direct_adjoint_mode stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue
          (msg.startsWith "Adjoint solve failed: `DirectAdjoint` without `dt0` could not infer")
          s!"Expected DirectAdjoint dt0 inference failure message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for DirectAdjoint degenerate missing dt0"

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

  @[test] def testImplicitAdjointWithoutFallbackUnsupported : IO Unit := do
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
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ImplicitAdjoint should report unsupported when backsolve fallback is disabled"
    LeanTest.assertTrue adjOpt.isNone
      "ImplicitAdjoint unsupported mode should not return adjoint values"
    LeanTest.assertTrue (getStat "unsupported_implicit_adjoint" sol.stats == 1)
      "ImplicitAdjoint unsupported mode should set unsupported_implicit_adjoint stat"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue
          (msg.startsWith "Adjoint solve failed: `ImplicitAdjoint` without backsolve fallback")
          s!"Expected ImplicitAdjoint unsupported fallback message, got: {msg}"
    | none =>
        LeanTest.fail "Expected unsupported message for ImplicitAdjoint without fallback"

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
  testForwardModeInfersDt0WhenAllowed
  testForwardModeRequireDt0RejectsMissingDt0
  testDirectAdjointDegenerateIntervalMissingDt0IsUnsupported
  testImplicitAdjointRejectsNonT1SaveAt
  testImplicitAdjointDefaultSaveAtStillWorks
  testImplicitAdjointAllowsEmptyTsSaveAt
  testBacksolveAdjointRejectsSubSaveAt
  testImplicitAdjointWithoutFallbackUnsupported
  testImplicitAdjointRecursiveRequiresFallback
  testRecursiveCheckpointAdjointRejectsBacksolveIncompatibleSaveAt

end Tests.DiffEqAdjointParity
