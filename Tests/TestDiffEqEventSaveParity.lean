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

private def getStat (key : String) (stats : List (String × Nat)) : Nat :=
  match stats.find? (fun entry => entry.fst == key) with
  | some (_, value) => value
  | none => 0

private def countApprox (xs : Array Time) (target tol : Time) : Nat := Id.run do
  let mut count := 0
  for x in xs do
    if approx x target tol then
      count := count + 1
  return count

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

@[test] def testEventSaveTsT1NoDuplicateTerminalWhenTsContainsEvent : IO Unit := do
  let saveat : SaveAt := {
    t1 := true
    ts := some #[1.0, 3.5, 5.5]
  }
  let sol := solveLinearEvent saveat
  assertEventOccurredAndMasked "saveat.ts+t1 no-dup terminal" sol
  assertNoSavesAfterTerminal "saveat.ts+t1 no-dup terminal" sol
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == ys.size)
        s!"saveat.ts+t1 no-dup terminal: ts/ys size mismatch {ts.size}/{ys.size}"
      LeanTest.assertTrue (ts.any (fun t => approx t 1.0 1.0e-12))
        "saveat.ts+t1 no-dup terminal: expected pre-event ts=1.0"
      LeanTest.assertTrue (countApprox ts sol.t1 1.0e-9 == 1)
        s!"saveat.ts+t1 no-dup terminal: expected exactly one terminal save at {sol.t1}, got {ts}"
      LeanTest.assertTrue (ts.size == 2)
        s!"saveat.ts+t1 no-dup terminal: expected exactly [1.0, t_event], got {ts}"
  | _, _ =>
      LeanTest.fail "saveat.ts+t1 no-dup terminal: expected ts/ys outputs"

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

@[test] def testBooleanEventAtStartTerminatesWithoutStepping : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.0)
    terminate := true
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) ()
      (saveat := { t1 := true })
      (event := some ev)
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Boolean event true at start should terminate immediately"
  LeanTest.assertTrue (approx sol.t1 0.0 1.0e-12)
    s!"Expected terminal time at t0=0.0, got {sol.t1}"
  LeanTest.assertTrue (getStat "num_steps" sol.stats == 0)
    s!"Boolean start-hit should not attempt steps, got num_steps={getStat "num_steps" sol.stats}"
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == 1 && ys.size == 1)
        s!"Expected one saved t/y at t0, got sizes {ts.size}/{ys.size}"
      LeanTest.assertTrue (approx ts[0]! 0.0 1.0e-12)
        s!"Expected saved time t0=0.0, got {ts[0]!}"
      LeanTest.assertTrue (approx ys[0]! 0.0 1.0e-12)
        s!"Expected saved state y0=0.0, got {ys[0]!}"
  | _, _ =>
      LeanTest.fail "Boolean start-hit should produce a single saved endpoint"

@[test] def testRealEventAtStartRunsStepThenLocalizes : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let ev : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y)
    terminate := true
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.5) (0.0 : Float) ()
      (saveat := { t1 := true })
      (event := some ev)
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Real event with cond(t0)=0 should terminate via step sign-change localization"
  LeanTest.assertTrue (approx sol.t1 0.0 1.0e-6)
    s!"Expected localized root at t0=0.0, got {sol.t1}"
  LeanTest.assertTrue (getStat "num_steps" sol.stats > 0)
    "Real start-root should not short-circuit before stepping"
  match sol.eventMask, sol.eventMaskLast with
  | some mask, some lastMask =>
      LeanTest.assertTrue (mask.size == 1 && lastMask.size == 1)
        s!"Expected single-event masks, got {mask.size}/{lastMask.size}"
      LeanTest.assertTrue (mask[0]! && lastMask[0]!)
        "Localized real start-root should be recorded in event masks"
  | _, _ =>
      LeanTest.fail "Expected event masks for real start-root case"

def run : IO Unit := do
  testEventSaveTsIgnoresPostEventTimes
  testEventSaveTsT1NoDuplicateTerminalWhenTsContainsEvent
  testEventSaveStepsStopsAtEvent
  testEventSaveStepsCadenceHonorsT1FlagAtEvent
  testEventSaveSubsIgnorePostEventTimes
  testBooleanEventAtStartTerminatesWithoutStepping
  testRealEventAtStartRunsStepThenLocalizes

end Tests.DiffEqEventSaveParity
