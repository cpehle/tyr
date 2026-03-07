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

  @[test] def testImplicitAdjointFallbackDisabled : IO Unit := do
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
    LeanTest.assertTrue (sol.result == Result.internalError)
      "ImplicitAdjoint without fallback should report unsupported mode"
    LeanTest.assertTrue adjOpt.isNone
      "ImplicitAdjoint unsupported mode should not return adjoint values"
    match errOpt with
    | some msg =>
        LeanTest.assertTrue (msg.contains "without backsolve fallback")
          s!"Expected unsupported fallback message, got: {msg}"
    | none =>
        LeanTest.fail "Expected error message for unsupported implicit adjoint mode"

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

  @[test] def testForwardModeInferredDt0MatchesExplicitDt0 : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      Euler.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let mode : ForwardMode := { requireDt0 := false }
    let y0 := 2.0
    let a := 0.3
    let adjY1 := 1.0
    let steps := 16
    let inferredDt := (1.0 : Float) / Float.ofNat (steps / 2)
    let (solInferred, adjInferredOpt, errInferredOpt) :=
      diffeqsolveForwardMode
        mode term solver 0.0 1.0 none y0 a adjY1
        (saveat := { t1 := true }) (maxSteps := steps)
    let (solExplicit, adjExplicitOpt, errExplicitOpt) :=
      diffeqsolveForwardMode
        mode term solver 0.0 1.0 (some inferredDt) y0 a adjY1
        (saveat := { t1 := true }) (maxSteps := steps)

    LeanTest.assertTrue (solInferred.result == Result.successful)
      "ForwardMode with inferred dt0 should succeed when requireDt0=false"
    LeanTest.assertTrue errInferredOpt.isNone
      "ForwardMode with inferred dt0 should not report unsupported errors"
    LeanTest.assertTrue (solExplicit.result == Result.successful)
      "ForwardMode with explicit dt0 should succeed"
    LeanTest.assertTrue errExplicitOpt.isNone
      "ForwardMode with explicit dt0 should not report unsupported errors"

    let yInferred ← getY1 "forward inferred dt0" solInferred
    let yExplicit ← getY1 "forward explicit dt0" solExplicit
    LeanTest.assertTrue (approx yInferred yExplicit 1.0e-12)
      s!"ForwardMode inferred/explicit primal mismatch: {yInferred} vs {yExplicit}"
    match adjInferredOpt, adjExplicitOpt with
    | some adjInferred, some adjExplicit =>
        LeanTest.assertTrue (approx adjInferred.adjY0 adjExplicit.adjY0 2.0e-3)
          s!"ForwardMode inferred/explicit adjY0 mismatch: {adjInferred.adjY0} vs {adjExplicit.adjY0}"
        LeanTest.assertTrue (approx adjInferred.adjArgs adjExplicit.adjArgs 2.0e-3)
          s!"ForwardMode inferred/explicit adjArgs mismatch: {adjInferred.adjArgs} vs {adjExplicit.adjArgs}"
    | _, _ =>
        LeanTest.fail "Expected adjoint results for both inferred/explicit forward mode runs"

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
        LeanTest.assertTrue (msg.contains "requires `dt0`")
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
