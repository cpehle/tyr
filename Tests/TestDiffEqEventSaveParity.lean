import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqEventSaveParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) <= tol

private def solveLinearEvent (saveat : SaveAt) :=
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .real (fun t _y _ => t - 3.5)
    terminate := true
  }
  diffeqsolve
    (Term := ODETerm Float Unit)
    (Y := Float)
    (VF := Float)
    (Control := Time)
    (Args := Unit)
    (Controller := ConstantStepSize)
    term solver 0.0 6.0 (some 1.0) (0.0 : Float) ()
    (saveat := saveat)
    (event := some ev)
    (maxSteps := 16)

private def assertEventOccurredAndMasked {S C : Type}
    (label : String) (sol : Solution Float S C) : IO Unit := do
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    s!"{label}: expected Result.eventOccurred, got {repr sol.result}"
  LeanTest.assertTrue (approx sol.t1 3.5 1.0e-4)
    s!"{label}: expected terminal time near 3.5, got {sol.t1}"
  match sol.eventMask, sol.eventMaskLast with
  | some mask, some lastMask =>
      LeanTest.assertTrue (mask.size == 1)
        s!"{label}: expected one event mask entry, got {mask.size}"
      LeanTest.assertTrue (lastMask.size == 1)
        s!"{label}: expected one last-event mask entry, got {lastMask.size}"
      LeanTest.assertTrue mask[0]! s!"{label}: eventMask should record hit"
      LeanTest.assertTrue lastMask[0]! s!"{label}: eventMaskLast should record hit"
  | _, _ =>
      LeanTest.fail s!"{label}: expected eventMask and eventMaskLast"

private def assertNoSavesAfterTerminal {S C : Type}
    (label : String) (sol : Solution Float S C) : IO Unit := do
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == ys.size)
        s!"{label}: ts/ys size mismatch {ts.size}/{ys.size}"
      for i in [:ts.size] do
        let t := ts[i]!
        LeanTest.assertTrue (t <= sol.t1 + 1.0e-9)
          s!"{label}: ts[{i}]={t} exceeds terminal time {sol.t1}"
  | _, _ =>
      LeanTest.fail s!"{label}: expected ts/ys outputs"

@[test] def testEventSaveTsIgnoresPostEventTimes : IO Unit := do
  let saveat : SaveAt := {
    t1 := false
    ts := some #[0.5, 3.5, 5.5]
  }
  let sol := solveLinearEvent saveat
  assertEventOccurredAndMasked "saveat.ts" sol
  assertNoSavesAfterTerminal "saveat.ts" sol
  match sol.ts with
  | some ts =>
      LeanTest.assertTrue (ts.any (fun t => approx t 0.5 1.0e-12))
        "saveat.ts: expected pre-event ts=0.5 to be saved"
      LeanTest.assertTrue (ts.any (fun t => approx t sol.t1 1.0e-4))
        "saveat.ts: expected event terminal time to be represented"
  | none =>
      LeanTest.fail "saveat.ts: expected saved times"

@[test] def testEventSaveStepsStopsAtEvent : IO Unit := do
  let saveat : SaveAt := {
    t1 := false
    steps := (2 : Nat)
  }
  let sol := solveLinearEvent saveat
  assertEventOccurredAndMasked "saveat.steps" sol
  assertNoSavesAfterTerminal "saveat.steps" sol
  match sol.ts with
  | some ts =>
      LeanTest.assertTrue (ts.any (fun t => approx t sol.t1 1.0e-4))
        "saveat.steps: expected terminal event time to be saved"
      LeanTest.assertTrue (ts.any (fun t => t < sol.t1 - 1.0e-9))
        "saveat.steps: expected at least one strictly pre-event save"
  | none =>
      LeanTest.fail "saveat.steps: expected saved times"

@[test] def testEventSaveStepsCadenceHonorsT1FlagAtEvent : IO Unit := do
  let solNoT1 := solveLinearEvent { t1 := false, steps := (3 : Nat) }
  let solWithT1 := solveLinearEvent { t1 := true, steps := (3 : Nat) }
  assertEventOccurredAndMasked "saveat.steps(3,t1=false)" solNoT1
  assertEventOccurredAndMasked "saveat.steps(3,t1=true)" solWithT1
  assertNoSavesAfterTerminal "saveat.steps(3,t1=false)" solNoT1
  assertNoSavesAfterTerminal "saveat.steps(3,t1=true)" solWithT1
  match solNoT1.ts with
  | some ts =>
      LeanTest.assertTrue (ts.any (fun t => approx t 3.0 1.0e-12))
        "saveat.steps(3,t1=false): expected cadence save at t=3.0"
      LeanTest.assertTrue (!(ts.any (fun t => approx t solNoT1.t1 1.0e-4)))
        "saveat.steps(3,t1=false): should not force terminal event time when t1=false"
  | none =>
      LeanTest.fail "saveat.steps(3,t1=false): expected saved times"
  match solWithT1.ts with
  | some ts =>
      LeanTest.assertTrue (ts.any (fun t => approx t 3.0 1.0e-12))
        "saveat.steps(3,t1=true): expected cadence save at t=3.0"
      LeanTest.assertTrue (ts.any (fun t => approx t solWithT1.t1 1.0e-4))
        "saveat.steps(3,t1=true): expected terminal event time when t1=true"
  | none =>
      LeanTest.fail "saveat.steps(3,t1=true): expected saved times"

@[test] def testEventSaveSubsIgnorePostEventTimes : IO Unit := do
  let tsSub : SubSaveAt := { ts := some #[0.5, 3.5, 5.5] }
  let stepsSub : SubSaveAt := { steps := (2 : Nat) }
  let saveat : SaveAt := {
    t1 := false
    subs := #[tsSub, stepsSub]
  }
  let sol := solveLinearEvent saveat
  assertEventOccurredAndMasked "saveat.subs" sol
  assertNoSavesAfterTerminal "saveat.subs" sol
  match sol.ts with
  | some ts =>
      LeanTest.assertTrue (ts.any (fun t => approx t 0.5 1.0e-12))
        "saveat.subs: expected pre-event ts=0.5 to be saved"
      LeanTest.assertTrue (ts.any (fun t => approx t sol.t1 1.0e-4))
        "saveat.subs: expected event terminal time to be represented"
  | none =>
      LeanTest.fail "saveat.subs: expected saved times"

def run : IO Unit := do
  testEventSaveTsIgnoresPostEventTimes
  testEventSaveStepsStopsAtEvent
  testEventSaveStepsCadenceHonorsT1FlagAtEvent
  testEventSaveSubsIgnorePostEventTimes

end Tests.DiffEqEventSaveParity
