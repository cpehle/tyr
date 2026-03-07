import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqProgressMeterParity2

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) <= tol

private def getStat (key : String) (stats : List (String × Nat)) : Nat :=
  match stats.find? (fun (k, _) => k == key) with
  | some (_, v) => v
  | none => 0

private def assertProgressStatsForMode
    (label : String)
    (mode : ProgressMeter)
    (stats : List (String × Nat)) : IO Unit := do
  let numSteps := getStat "num_steps" stats
  let numAccepted := getStat "num_accepted_steps" stats
  let numRejected := getStat "num_rejected_steps" stats
  LeanTest.assertTrue (numSteps == numAccepted + numRejected)
    s!"{label}: expected num_steps = accepted + rejected, got {numSteps}, {numAccepted}, {numRejected}"
  match mode with
  | .none =>
      LeanTest.assertTrue (getStat "progress_meter_start" stats == 0)
        s!"{label}: .none should not emit progress_meter_start"
      LeanTest.assertTrue (getStat "progress_meter_updates" stats == 0)
        s!"{label}: .none should not emit progress_meter_updates"
      LeanTest.assertTrue (getStat "progress_meter_close" stats == 0)
        s!"{label}: .none should not emit progress_meter_close"
      LeanTest.assertTrue (getStat "progress_meter_tqdm_alias" stats == 0)
        s!"{label}: .none should not emit progress_meter_tqdm_alias"
  | .text =>
      LeanTest.assertTrue (getStat "progress_meter_start" stats == 1)
        s!"{label}: .text should emit one progress_meter_start"
      LeanTest.assertTrue (getStat "progress_meter_updates" stats == numSteps)
        s!"{label}: .text progress_meter_updates should equal num_steps"
      LeanTest.assertTrue (getStat "progress_meter_close" stats == 1)
        s!"{label}: .text should emit one progress_meter_close"
      LeanTest.assertTrue (getStat "progress_meter_tqdm_alias" stats == 0)
        s!"{label}: .text should not emit tqdm alias stat"
  | .tqdm =>
      LeanTest.assertTrue (getStat "progress_meter_start" stats == 1)
        s!"{label}: .tqdm should emit one progress_meter_start"
      LeanTest.assertTrue (getStat "progress_meter_updates" stats == numSteps)
        s!"{label}: .tqdm progress_meter_updates should equal num_steps"
      LeanTest.assertTrue (getStat "progress_meter_close" stats == 1)
        s!"{label}: .tqdm should emit one progress_meter_close"
      LeanTest.assertTrue (getStat "progress_meter_tqdm_alias" stats == 1)
        s!"{label}: .tqdm should emit compatibility alias stat"

private def assertSavedSeriesApproxEq {S C : Type}
    (label : String)
    (a b : Solution Float S C) : IO Unit := do
  match a.ts, b.ts, a.ys, b.ys with
  | some ats, some bts, some ays, some bys =>
      LeanTest.assertTrue (ats.size == bts.size)
        s!"{label}: ts sizes differ: {ats.size} vs {bts.size}"
      LeanTest.assertTrue (ays.size == bys.size)
        s!"{label}: ys sizes differ: {ays.size} vs {bys.size}"
      LeanTest.assertTrue (ats.size == ays.size && bts.size == bys.size)
        s!"{label}: ts/ys alignment mismatch"
      for i in [:ats.size] do
        LeanTest.assertTrue (approx ats[i]! bts[i]! 1.0e-10)
          s!"{label}: ts[{i}] mismatch: {ats[i]!} vs {bts[i]!}"
        LeanTest.assertTrue (approx ays[i]! bys[i]! 1.0e-10)
          s!"{label}: ys[{i}] mismatch: {ays[i]!} vs {bys[i]!}"
  | _, _, _, _ =>
      LeanTest.fail s!"{label}: expected both solutions to contain ts/ys"

private def assertEventMaskParity {S C : Type}
    (label : String)
    (noneSol textSol tqdmSol : Solution Float S C) :
    IO Unit := do
  match noneSol.eventMask, textSol.eventMask, tqdmSol.eventMask with
  | some mNone, some mText, some mTqdm =>
      LeanTest.assertTrue (mNone.size == mText.size && mText.size == mTqdm.size)
        s!"{label}: eventMask size mismatch"
      for i in [:mNone.size] do
        let n := mNone[i]!
        let t := mText[i]!
        let q := mTqdm[i]!
        LeanTest.assertTrue (n == t && t == q)
          s!"{label}: eventMask[{i}] mismatch: none={n}, text={t}, tqdm={q}"
  | _, _, _ =>
      LeanTest.fail s!"{label}: expected eventMask in all modes"
  match noneSol.eventMaskLast, textSol.eventMaskLast, tqdmSol.eventMaskLast with
  | some mNone, some mText, some mTqdm =>
      LeanTest.assertTrue (mNone.size == mText.size && mText.size == mTqdm.size)
        s!"{label}: eventMaskLast size mismatch"
      for i in [:mNone.size] do
        let n := mNone[i]!
        let t := mText[i]!
        let q := mTqdm[i]!
        LeanTest.assertTrue (n == t && t == q)
          s!"{label}: eventMaskLast[{i}] mismatch: none={n}, text={t}, tqdm={q}"
  | _, _, _ =>
      LeanTest.fail s!"{label}: expected eventMaskLast in all modes"

private def assertNoSavesAfterTerminal {S C : Type}
    (label : String)
    (sol : Solution Float S C) : IO Unit := do
  match sol.ts with
  | some ts =>
      for i in [:ts.size] do
        LeanTest.assertTrue (ts[i]! <= sol.t1 + 1.0e-10)
          s!"{label}: ts[{i}]={ts[i]!} exceeds terminal time {sol.t1}"
  | none =>
      LeanTest.fail s!"{label}: expected saved times"

private def solveLinearWithSaveSteps
    (mode : ProgressMeter)
    (event : Option (EventSpec Float Unit) := none) :=
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  diffeqsolve
    (Term := ODETerm Float Unit)
    (Y := Float)
    (VF := Float)
    (Control := Time)
    (Args := Unit)
    (Controller := ConstantStepSize)
    term solver 0.0 6.0 (some 1.0) (0.0 : Float) ()
    (saveat := { t1 := false, steps := (2 : Nat) })
    (event := event)
    (maxSteps := 16)
    (progress_meter := mode)

private def assertCoreParityAcrossModes {S C : Type}
    (label : String)
    (noneSol textSol tqdmSol : Solution Float S C)
    (expectedResult : Result) : IO Unit := do
  LeanTest.assertTrue (noneSol.result == expectedResult)
    s!"{label}: .none result mismatch: {repr noneSol.result}"
  LeanTest.assertTrue (textSol.result == expectedResult)
    s!"{label}: .text result mismatch: {repr textSol.result}"
  LeanTest.assertTrue (tqdmSol.result == expectedResult)
    s!"{label}: .tqdm result mismatch: {repr tqdmSol.result}"
  LeanTest.assertTrue (approx noneSol.t1 textSol.t1 1.0e-10)
    s!"{label}: terminal time mismatch none/text: {noneSol.t1} vs {textSol.t1}"
  LeanTest.assertTrue (approx noneSol.t1 tqdmSol.t1 1.0e-10)
    s!"{label}: terminal time mismatch none/tqdm: {noneSol.t1} vs {tqdmSol.t1}"
  assertSavedSeriesApproxEq s!"{label}: none vs text saved payload" noneSol textSol
  assertSavedSeriesApproxEq s!"{label}: none vs tqdm saved payload" noneSol tqdmSol
  LeanTest.assertTrue (getStat "num_steps" noneSol.stats == getStat "num_steps" textSol.stats)
    s!"{label}: num_steps mismatch none/text"
  LeanTest.assertTrue (getStat "num_steps" noneSol.stats == getStat "num_steps" tqdmSol.stats)
    s!"{label}: num_steps mismatch none/tqdm"
  LeanTest.assertTrue
    (getStat "num_accepted_steps" noneSol.stats == getStat "num_accepted_steps" textSol.stats)
    s!"{label}: num_accepted_steps mismatch none/text"
  LeanTest.assertTrue
    (getStat "num_accepted_steps" noneSol.stats == getStat "num_accepted_steps" tqdmSol.stats)
    s!"{label}: num_accepted_steps mismatch none/tqdm"
  LeanTest.assertTrue
    (getStat "num_rejected_steps" noneSol.stats == getStat "num_rejected_steps" textSol.stats)
    s!"{label}: num_rejected_steps mismatch none/text"
  LeanTest.assertTrue
    (getStat "num_rejected_steps" noneSol.stats == getStat "num_rejected_steps" tqdmSol.stats)
    s!"{label}: num_rejected_steps mismatch none/tqdm"
  assertProgressStatsForMode s!"{label}: .none" .none noneSol.stats
  assertProgressStatsForMode s!"{label}: .text" .text textSol.stats
  assertProgressStatsForMode s!"{label}: .tqdm" .tqdm tqdmSol.stats

@[test] def testProgressMeterSaveAtStepsParity : IO Unit := do
  /-
  Diffrax reference: `../diffrax/test/test_progress_meter.py`.
  Signal we can assert in Tyr parity tests:
  - solve payload/result are mode-invariant
  - progress-meter stats reflect compatibility surface, not numerics
  -/
  let solNone := solveLinearWithSaveSteps .none
  let solText := solveLinearWithSaveSteps .text
  let solTqdm := solveLinearWithSaveSteps .tqdm
  assertCoreParityAcrossModes "saveat.steps parity" solNone solText solTqdm Result.successful
  LeanTest.assertTrue (approx solNone.t1 6.0 1.0e-12)
    s!"saveat.steps parity: expected terminal t1=6.0, got {solNone.t1}"

@[test] def testProgressMeterEventTerminationSaveAtStepsParity : IO Unit := do
  /-
  Diffrax reference: `../diffrax/test/test_progress_meter.py`.
  Event-termination compatibility + saveat.steps should remain invariant across meter modes.
  -/
  let ev : EventSpec Float Unit := {
    condition := .real (fun t _y _ => t - 3.5)
    terminate := true
  }
  let solNone := solveLinearWithSaveSteps .none (event := some ev)
  let solText := solveLinearWithSaveSteps .text (event := some ev)
  let solTqdm := solveLinearWithSaveSteps .tqdm (event := some ev)

  assertCoreParityAcrossModes
    "event + saveat.steps parity" solNone solText solTqdm Result.eventOccurred
  LeanTest.assertTrue (approx solNone.t1 3.5 1.0e-4)
    s!"event + saveat.steps parity: expected localized terminal time near 3.5, got {solNone.t1}"
  assertNoSavesAfterTerminal "event + saveat.steps .none" solNone
  assertNoSavesAfterTerminal "event + saveat.steps .text" solText
  assertNoSavesAfterTerminal "event + saveat.steps .tqdm" solTqdm
  assertEventMaskParity "event + saveat.steps parity" solNone solText solTqdm
  match solNone.ts with
  | some ts =>
      LeanTest.assertTrue (ts.any (fun t => t < solNone.t1 - 1.0e-9))
        "event + saveat.steps parity: expected at least one strictly pre-event save"
      LeanTest.assertTrue (ts.any (fun t => approx t solNone.t1 1.0e-4))
        "event + saveat.steps parity: expected terminal event time to be saved"
  | none =>
      LeanTest.fail "event + saveat.steps parity: expected saved times"

def run : IO Unit := do
  testProgressMeterSaveAtStepsParity
  testProgressMeterEventTerminationSaveAtStepsParity

end Tests.DiffEqProgressMeterParity2
