import Tyr.DiffEq.SaveAt
import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Interpolation
import Tyr.DiffEq.StepSizeController

namespace torch
namespace DiffEq

/-! ## Integration Entry Point -/

inductive EventCondition (Y Args : Type) where
  | boolean : (Time → Y → Args → Bool) → EventCondition Y Args
  | real : (Time → Y → Args → Float) → EventCondition Y Args

structure EventSpec (Y Args : Type) where
  condition : EventCondition Y Args
  terminate : Bool := true
  direction : Option Bool := none
  rootMaxIters : Nat := 24
  rootTol : Time := 1.0e-6

/-- Progress meter compatibility surface for `diffeqsolve`. -/
inductive ProgressMeter where
  | none
  | text
  | tqdm
  deriving Repr, BEq, DecidableEq, Inhabited

namespace ProgressMeter

/--
Normalize compatibility aliases to concrete behavior.
Currently `tqdm` falls back to lightweight text logging.
-/
def normalize (mode : ProgressMeter) : ProgressMeter :=
  match mode with
  | .tqdm => .text
  | other => other

def textEnabled (mode : ProgressMeter) : Bool :=
  mode.normalize == .text

end ProgressMeter

private def normalizeSaveTs (ts : Option (Array Time)) : Option (Array Time) :=
  match ts with
  | some xs => if xs.size == 0 then none else some xs
  | none => none

private def isMonotoneInSolveDirection (t0 t1 : Time) (ts : Array Time) : Bool := Id.run do
  if ts.size <= 1 then
    return true
  let forward := t1 >= t0
  let mut ok := true
  for i in [:ts.size - 1] do
    let lhs := ts[i]!
    let rhs := ts[i + 1]!
    let monotone := if forward then rhs >= lhs else rhs <= lhs
    ok := ok && monotone
  return ok

private def constantInterpolation [DiffEqSpace Y] (y : Y) : DenseInterpolation Y := {
  evaluate := fun _t0 t1 _left =>
    match t1 with
    | none => y
    | some _ => DiffEqSpace.sub y y
  derivative := fun _t _left =>
    DiffEqSpace.scale 0.0 (DiffEqSpace.sub y y)
}

private def floatSign (v : Float) : Int :=
  if v > 0.0 then
    1
  else if v < 0.0 then
    -1
  else
    0

private def hasSignChange (v0 v1 : Float) : Bool :=
  floatSign v0 != floatSign v1

private def hasDirectedSignChange (v0 v1 : Float) (direction : Option Bool) : Bool :=
  match direction with
  | none => hasSignChange v0 v1
  | some true => floatSign v0 <= 0 && floatSign v1 > 0
  | some false => floatSign v0 > 0 && floatSign v1 <= 0

private def maxTol (a b : Time) : Time :=
  if a >= b then a else b

private def timesWithinTol (t0 t1 tol : Time) : Bool :=
  Float.abs (t0 - t1) <= tol

private def hasNoTimeProgress (t0 t1 tol : Time) : Bool :=
  let tol := if tol > 0.0 then tol else 1.0e-12
  timesWithinTol t0 t1 tol

private def eventHitPrecedes (forward : Bool) (lhs rhs : Time) : Bool :=
  if forward then lhs < rhs else lhs > rhs

private structure StepEventHit (Y : Type) where
  idx : Nat
  time : Time
  y : Y
  terminate : Bool
  tol : Time

private def preferStepEventHit {Y : Type}
    (forward : Bool)
    (candidate best : StepEventHit Y) : Bool :=
  if eventHitPrecedes forward candidate.time best.time then
    true
  else
    let tol := maxTol candidate.tol best.tol
    if timesWithinTol candidate.time best.time tol then
      candidate.terminate && !best.terminate
    else
      false

private def chooseStepEventHit? {Y : Type}
    (forward : Bool)
    (hits : Array (StepEventHit Y)) : Option (StepEventHit Y) := Id.run do
  let mut chosen : Option (StepEventHit Y) := none
  for hit in hits do
    match chosen with
    | none => chosen := some hit
    | some best =>
        if preferStepEventHit forward hit best then
          chosen := some hit
  return chosen

private def maskFromStepHits {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (hits : Array (StepEventHit Y)) : Array Bool := Id.run do
  let mut mask := Array.replicate events.size false
  for hit in hits do
    if hit.idx < mask.size then
      mask := mask.set! hit.idx true
  return mask

private def maskAtEventTime {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (hits : Array (StepEventHit Y))
    (chosen : StepEventHit Y) : Array Bool := Id.run do
  let mut mask := Array.replicate events.size false
  for hit in hits do
    let tol := maxTol hit.tol chosen.tol
    if timesWithinTol hit.time chosen.time tol then
      if hit.idx < mask.size then
        mask := mask.set! hit.idx true
  return mask

private def configuredEvents {Y Args : Type}
    (event : Option (EventSpec Y Args))
    (events : Array (EventSpec Y Args)) : Array (EventSpec Y Args) :=
  match event with
  | none => events
  | some ev => #[ev] ++ events

private def eventHitAtStart {Y Args : Type}
    (ev : EventSpec Y Args)
    (t0 : Time)
    (y0 : Y)
    (args : Args) : Bool :=
  match ev.condition with
  | .boolean cond => cond t0 y0 args
  | .real cond => cond t0 y0 args == 0.0

private def initialEventMask {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (t0 : Time)
    (y0 : Y)
    (args : Args) : Array Bool :=
  events.map (fun ev => eventHitAtStart ev t0 y0 args)

private def anyTerminatingEvent {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (mask : Array Bool) : Bool := Id.run do
  let mut hit := false
  for i in [:events.size] do
    if mask.getD i false then
      match events[i]? with
      | some ev =>
          if ev.terminate then
            hit := true
      | none => pure ()
  return hit

private def mergeEventMasks (base delta : Array Bool) : Array Bool :=
  base.mapIdx (fun i seen => seen || delta.getD i false)

private def eventMaskOption {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (mask : Array Bool) : Option (Array Bool) :=
  if events.size == 0 then none else some mask

private def eventMaskHitOption {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (mask : Array Bool) : Option (Array Bool) :=
  if events.size == 0 || !(mask.any (fun hit => hit)) then none else some mask

/-!
`max_steps=None` parity mode is implemented with a large finite cap so the core solver
loop remains structurally terminating.
-/
private def unboundedStepSafetyCap : Nat := 1000000

private def progressMeterStart (mode : ProgressMeter) (t0 t1 : Time) : Unit := Id.run do
  if mode.textEnabled then
    if mode == .tqdm then
      dbg_trace "[DiffEq progress_meter=tqdm] using text compatibility fallback."
    dbg_trace s!"[DiffEq progress_meter=text] start t0={t0} t1={t1}"

private def progressMeterShouldLogUpdate (attempted : Nat) : Bool :=
  attempted == 1 || attempted % 64 == 0

private def progressMeterUpdate
    (mode : ProgressMeter)
    (attempted : Nat)
    (tNow : Time) : Unit := Id.run do
  if mode.textEnabled && progressMeterShouldLogUpdate attempted then
    dbg_trace s!"[DiffEq progress_meter=text] update steps={attempted} t={tNow}"

private def progressMeterClose
    (mode : ProgressMeter)
    (result : Result)
    (numSteps numAcceptedSteps numRejectedSteps : Nat) : Unit := Id.run do
  if mode.textEnabled then
    dbg_trace
      s!"[DiffEq progress_meter=text] close steps={numSteps} accepted={numAcceptedSteps} rejected={numRejectedSteps} result={repr result}"

private def progressMeterStats (mode : ProgressMeter) (numSteps : Nat) : List (String × Nat) :=
  if mode.textEnabled then
    let aliasStats :=
      if mode == .tqdm then
        [("progress_meter_tqdm_alias", 1)]
      else
        []
    [
      ("progress_meter_start", 1),
      ("progress_meter_updates", numSteps),
      ("progress_meter_close", 1)
    ] ++ aliasStats
  else
    []

private def maybeThrowOnFailure {Y SolverState ControllerState : Type}
    (throwOnFailure : Bool)
    (sol : Solution Y SolverState ControllerState) :
    Solution Y SolverState ControllerState :=
  if throwOnFailure && sol.result.isFailure then
    panic! sol.result.message
  else
    sol

private def localizeSignChange {Y Args : Type}
    [DiffEqSpace Y] [Inhabited Y]
    (interp : DenseInterpolation Y)
    (eventFn : Time → Y → Args → Float)
    (args : Args)
    (t0 t1 : Time)
    (v0 v1 : Float)
    (maxIters : Nat)
    (tol : Time) : Time × Y := Id.run do
  let mut leftT := t0
  let mut rightT := t1
  let mut leftV := v0
  let mut rightV := v1
  let secantT :=
    if rightV == leftV then
      0.5 * (leftT + rightT)
    else
      let raw := leftT - leftV * (rightT - leftT) / (rightV - leftV)
      let tmin := if leftT <= rightT then leftT else rightT
      let tmax := if leftT <= rightT then rightT else leftT
      if raw < tmin then tmin else if raw > tmax then tmax else raw
  let mut midT := secantT
  let mut midY := interp.evaluate midT none true
  let mut midV := eventFn midT midY args
  if midV == 0.0 then
    return (midT, midY)
  for _ in [:maxIters] do
    if Float.abs (rightT - leftT) <= tol then
      return (midT, midY)
    if hasSignChange leftV midV then
      rightT := midT
      rightV := midV
    else
      leftT := midT
      leftV := midV
    midT := 0.5 * (leftT + rightT)
    midY := interp.evaluate midT none true
    midV := eventFn midT midY args
    if midV == 0.0 then
      return (midT, midY)
  return (midT, midY)

private def stepEventHit? {Y Args : Type}
    [DiffEqSpace Y] [Inhabited Y]
    (denseStep : DenseInterpolation Y)
    (ev : EventSpec Y Args)
    (idx : Nat)
    (t0 t1 : Time)
    (y0 y1 : Y)
    (args : Args) : Option (StepEventHit Y) :=
  match ev.condition with
  | .boolean cond =>
      let c0 := cond t0 y0 args
      let c1 := cond t1 y1 args
      if (!c0) && c1 then
        some {
          idx := idx
          time := t1
          y := y1
          terminate := ev.terminate
          tol := ev.rootTol
        }
      else
        none
  | .real cond =>
      let v0 := cond t0 y0 args
      let v1 := cond t1 y1 args
      if hasDirectedSignChange v0 v1 ev.direction then
        let timeY := localizeSignChange denseStep cond args t0 t1 v0 v1 ev.rootMaxIters ev.rootTol
        some {
          idx := idx
          time := timeY.fst
          y := timeY.snd
          terminate := ev.terminate
          tol := ev.rootTol
        }
      else
        none

def diffeqsolve {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (terms : Term)
    (solver : AbstractSolver Term Y VF Control Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default)
    (event : Option (EventSpec Y Args) := none)
    (initialSolverState : Option solver.SolverState := none)
    (initialControllerState :
      Option (StepSizeController.State (C := Controller)) := none)
    (initialMadeJump : Bool := false)
    (events : Array (EventSpec Y Args) := #[])
    (saveFn : Option (Time → Y → Args → Y) := none)
    (throwOnFailure : Bool := false)
    (maxStepsOpt : Option Nat := some maxSteps)
    (progress_meter : ProgressMeter := .none) :
    Solution Y solver.SolverState (StepSizeController.State (C := Controller)) := by
  let activeEvents := configuredEvents event events
  let eventMask0 := initialEventMask activeEvents t0 y0 args
  let terminateAtStart := anyTerminatingEvent activeEvents eventMask0
  let eventMask0? := eventMaskOption activeEvents eventMask0
  let eventMaskLast0? := eventMaskHitOption activeEvents eventMask0
  let saveatTs := normalizeSaveTs saveat.effectiveTs
  let saveDense := saveat.effectiveDense
  let saveSteps := saveat.stepsEnabled
  let saveAtT0 := saveat.effectiveT0
  let saveAtT1 := saveat.effectiveT1
  let saveSolverState := saveat.effectiveSolverState
  let saveControllerState := saveat.effectiveControllerState
  let saveMadeJump := saveat.effectiveMadeJump
  let isUnbounded := maxStepsOpt.isNone
  let maxStepBudget :=
    match maxStepsOpt with
    | some n => n
    | none => unboundedStepSafetyCap
  let saveConfigIncompatibleWithUnbounded := isUnbounded && (saveSteps || saveDense)
  let saveValue : Time → Y → Y :=
    match saveFn with
    | none => fun _ y => y
    | some f => fun t y => f t y args
  let stepStats0 : List (String × Nat) :=
    [("num_steps", 0), ("num_accepted_steps", 0), ("num_rejected_steps", 0)]
  let controllerInvalid :=
    (inferInstance : StepSizeControllerValidation Controller).validate controller t0 t1 dt0
  let outOfRange :=
    match saveatTs with
    | none => false
    | some ts =>
        let tmin := if t0 <= t1 then t0 else t1
        let tmax := if t0 <= t1 then t1 else t0
        ts.foldl (init := false) (fun acc t => acc || t < tmin || t > tmax)
  let badDirection :=
    match saveatTs with
    | none => false
    | some ts => !(isMonotoneInSolveDirection t0 t1 ts)
  if controllerInvalid.isSome || outOfRange || badDirection || saveConfigIncompatibleWithUnbounded then
    exact maybeThrowOnFailure throwOnFailure {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := stepStats0
      result := Result.internalError
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := eventMask0?
      eventMaskLast := eventMaskLast0?
    }
  else if dt0 == some 0.0 then
    exact maybeThrowOnFailure throwOnFailure {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := stepStats0
      result := Result.dtMinReached
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := eventMask0?
      eventMaskLast := eventMaskLast0?
    }
  else
    let computeOutputs :=
      fun (tf : Time) (yf : Y) (stepTs : Array Time) (stepYs : Array Y)
          (denseInterp : Option (DenseInterpolation Y)) =>
        let (stepTs, stepYs) :=
          if saveSteps then
            match stepTs.back?, stepYs.back? with
            | some lastT, some _ =>
                if lastT == tf then
                  (stepTs, stepYs)
                else
                  (stepTs.push tf, stepYs.push (saveValue tf yf))
            | _, _ => (#[t0, tf], #[saveValue t0 y0, saveValue tf yf])
          else
            (stepTs, stepYs)
        match saveatTs with
        | some ts =>
            let tsOut :=
              if saveAtT0 || saveAtT1 then
                let pre := if saveAtT0 then #[t0] else #[]
                let post := if saveAtT1 then #[tf] else #[]
                pre ++ ts ++ post
              else
                ts
            let ys :=
              match denseInterp with
              | none => #[]
              | some interp =>
                  tsOut.map (fun t => saveValue t (interp.evaluate t none true))
            (some tsOut, some ys)
        | none =>
            if saveSteps then
              (some stepTs, some stepYs)
            else if saveAtT0 || saveAtT1 then
              let ts :=
                if saveAtT0 && saveAtT1 then #[t0, tf]
                else if saveAtT0 then #[t0] else #[tf]
              let ys :=
                if saveAtT0 && saveAtT1 then #[saveValue t0 y0, saveValue tf yf]
                else if saveAtT0 then #[saveValue t0 y0] else #[saveValue tf yf]
              (some ts, some ys)
            else
              (none, none)
    if t0 == t1 then
      let denseInterp :=
        if saveDense || saveatTs.isSome then
          some (constantInterpolation y0)
        else
          none
      let stepTs := if saveSteps then #[t0] else #[]
      let stepYs := if saveSteps then #[saveValue t0 y0] else #[]
      let (tsOut, ysOut) := computeOutputs t0 y0 stepTs stepYs denseInterp
      exact maybeThrowOnFailure throwOnFailure {
        t0 := t0
        t1 := t1
        ts := tsOut
        ys := ysOut
        interpolation := denseInterp
        stats := stepStats0
        result := if terminateAtStart then Result.eventOccurred else Result.successful
        solverState := if saveSolverState then initialSolverState else none
        controllerState := if saveControllerState then initialControllerState else none
        madeJump := if saveMadeJump then some initialMadeJump else none
        eventMask := eventMask0?
        eventMaskLast := eventMaskLast0?
      }
    else
    let errorOrder := solver.errorOrder terms
    let initCtrlBase :=
      StepSizeController.init controller terms t0 t1 y0 args dt0 solver.func errorOrder
    let dtAbs := if initCtrlBase.dt < 0.0 then -initCtrlBase.dt else initCtrlBase.dt
    let dt := if t1 >= t0 then dtAbs else -dtAbs
    let initCtrlBase := { initCtrlBase with dt := dt }
    let initCtrl : StepSizeState (StepSizeController.State (C := Controller)) :=
      match initialControllerState with
      | some st => { dt := initCtrlBase.dt, state := st }
      | none => initCtrlBase
    let initState :=
      match initialSolverState with
      | some st => st
      | none => solver.init terms t0 t1 y0 args
    if terminateAtStart then
      let denseInterp :=
        if saveDense || saveatTs.isSome then
          some (constantInterpolation y0)
        else
          none
      let stepTs := if saveSteps then #[t0] else #[]
      let stepYs := if saveSteps then #[saveValue t0 y0] else #[]
      let (tsOut, ysOut) := computeOutputs t0 y0 stepTs stepYs denseInterp
      exact maybeThrowOnFailure throwOnFailure {
        t0 := t0
        t1 := t0
        ts := tsOut
        ys := ysOut
        interpolation := denseInterp
        stats := stepStats0
        result := Result.eventOccurred
        solverState := if saveSolverState then some initState else none
        controllerState := if saveControllerState then some initCtrl.state else none
        madeJump := if saveMadeJump then some true else none
        eventMask := eventMask0?
        eventMaskLast := eventMaskLast0?
      }
    else
    let denseTsInit := #[t0]
    let denseYsInit := #[y0]
    let denseSegsInit : Array (DenseInterpolation Y) := #[]
    let stepTsInit := if saveSteps then #[t0] else #[]
    let stepYsInit := if saveSteps then #[saveValue t0 y0] else #[]
    let eventMaskLastInit := eventMask0
    let _ := progressMeterStart progress_meter t0 t1
    let rec loop (attempted : Nat) (accepted : Nat) (rejected : Nat) (t : Time) (y : Y)
        (state : solver.SolverState)
        (ctrlState : StepSizeState (StepSizeController.State (C := Controller)))
        (madeJump : Bool)
        (denseTs : Array Time) (denseYs : Array Y)
        (denseSegs : Array (DenseInterpolation Y))
        (stepTs : Array Time) (stepYs : Array Y)
        (eventMaskAcc : Array Bool)
        (eventMaskLastAcc : Array Bool) :
        (Time × Y × solver.SolverState ×
          StepSizeState (StepSizeController.State (C := Controller)) ×
          Bool × Array Time × Array Y × Array (DenseInterpolation Y) ×
          Array Time × Array Y × Result × Nat × Nat × Nat × Array Bool × Array Bool) :=
      if attempted >= maxStepBudget then
        (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
          Result.maxStepsReached, attempted, accepted, rejected, eventMaskAcc, eventMaskLastAcc)
      else if rejected >= maxStepBudget then
        (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
          Result.maxStepsRejected, attempted, accepted, rejected, eventMaskAcc, eventMaskLastAcc)
      else
        let dt := ctrlState.dt
        let done := if dt > 0.0 then t >= t1 else t <= t1
        if done then
          (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
            Result.successful, attempted, accepted, rejected, eventMaskAcc, eventMaskLastAcc)
        else
          let tCandidate := t + dt
          let tNext :=
            if dt > 0.0 then
              if tCandidate > t1 then t1 else tCandidate
            else
              if tCandidate < t1 then t1 else tCandidate
          let attemptedNext := attempted + 1
          let _ := progressMeterUpdate progress_meter attemptedNext tNext
          if tNext == t then
            (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
              Result.dtMinReached, attemptedNext, accepted, rejected, eventMaskAcc, eventMaskLastAcc)
          else
            let step := solver.step terms t tNext y args state madeJump
            if step.result != Result.successful then
              (tNext, step.y1, step.solverState, ctrlState, madeJump, denseTs, denseYs, denseSegs,
                stepTs, stepYs, step.result, attemptedNext, accepted, rejected,
                eventMaskAcc, eventMaskLastAcc)
            else
              let decision :=
                StepSizeController.adapt controller ctrlState t tNext y step.y1 step.yError errorOrder
              if decision.result != Result.successful then
                (tNext, step.y1, step.solverState, ctrlState, madeJump, denseTs, denseYs, denseSegs,
                  stepTs, stepYs, decision.result, attemptedNext, accepted, rejected,
                  eventMaskAcc, eventMaskLastAcc)
              else if decision.accept then
                let ctrlState := { dt := decision.dt, state := decision.state }
                let denseStep := solver.interpolation step.denseInfo
                let madeJumpNext := decision.madeJump
                let acceptedNext := accepted + 1
                let forward := tNext >= t
                let stepHits : Array (StepEventHit Y) := Id.run do
                  let mut hits : Array (StepEventHit Y) := #[]
                  for i in [:activeEvents.size] do
                    match activeEvents[i]? with
                    | none => pure ()
                    | some ev =>
                        match stepEventHit? denseStep ev i t tNext y step.y1 args with
                        | none => pure ()
                        | some hit => hits := hits.push hit
                  return hits
                let stepMaskAll := maskFromStepHits activeEvents stepHits
                let eventMaskAcc := mergeEventMasks eventMaskAcc stepMaskAll
                let chosenHit? := chooseStepEventHit? forward stepHits
                let stepMaskChosen :=
                  match chosenHit? with
                  | none => Array.replicate activeEvents.size false
                  | some chosen => maskAtEventTime activeEvents stepHits chosen
                let eventMaskLastAcc :=
                  match chosenHit? with
                  | none => eventMaskLastAcc
                  | some _ => stepMaskChosen
                let chosenTerminates := anyTerminatingEvent activeEvents stepMaskChosen
                let (eventHit, chosenTime, chosenY, chosenTol) :=
                  match chosenHit? with
                  | none => (false, tNext, step.y1, 0.0)
                  | some chosen => (true, chosen.time, chosen.y, chosen.tol)
                let noProgressEvent := eventHit && hasNoTimeProgress t chosenTime chosenTol
                let restartAtLocalized := eventHit && !chosenTerminates && !noProgressEvent
                let stepOutTime :=
                  if restartAtLocalized || chosenTerminates then chosenTime else tNext
                let stepOutY :=
                  if restartAtLocalized || chosenTerminates then chosenY else step.y1
                let denseTs := denseTs.push stepOutTime
                let denseYs := denseYs.push stepOutY
                let denseSegs := denseSegs.push denseStep
                let stepSaved := saveat.shouldSaveAcceptedStep acceptedNext
                let (stepTs, stepYs) :=
                  if stepSaved then
                    (stepTs.push stepOutTime, stepYs.push (saveValue stepOutTime stepOutY))
                  else
                    (stepTs, stepYs)
                if eventHit then
                  if chosenTerminates then
                    (stepOutTime, stepOutY, step.solverState, ctrlState, true,
                      denseTs, denseYs, denseSegs, stepTs, stepYs,
                      Result.eventOccurred, attemptedNext, acceptedNext, rejected,
                      eventMaskAcc, eventMaskLastAcc)
                  else
                    loop attemptedNext acceptedNext 0 stepOutTime stepOutY
                      step.solverState ctrlState true
                      denseTs denseYs denseSegs stepTs stepYs eventMaskAcc eventMaskLastAcc
                else
                  loop attemptedNext acceptedNext 0 tNext step.y1 step.solverState ctrlState madeJumpNext
                    denseTs denseYs denseSegs stepTs stepYs eventMaskAcc eventMaskLastAcc
              else
                let ctrlState := { dt := decision.dt, state := decision.state }
                loop attemptedNext accepted (rejected + 1) t y state ctrlState madeJump
                  denseTs denseYs denseSegs stepTs stepYs eventMaskAcc eventMaskLastAcc
    let (tf, yf, statef, ctrlState, madeJumpf, denseTs, denseYs, denseSegs, stepTs, stepYs,
        result, numSteps, numAcceptedSteps, numRejectedSteps, eventMaskF, eventMaskLastF) :=
      loop 0 0 0 t0 y0 initState initCtrl initialMadeJump
        denseTsInit denseYsInit denseSegsInit stepTsInit stepYsInit eventMask0 eventMaskLastInit
    let _ := progressMeterClose progress_meter result numSteps numAcceptedSteps numRejectedSteps
    let denseInterp :=
      if saveDense || saveatTs.isSome then
        if denseSegs.size > 0 then
          some (PiecewiseDenseInterpolation.toDense { ts := denseTs, segments := denseSegs })
        else
          some (constantInterpolation yf)
      else
        none
    let (tsOut, ysOut) := computeOutputs tf yf stepTs stepYs denseInterp
    let stats :=
      [
        ("num_steps", numSteps),
        ("num_accepted_steps", numAcceptedSteps),
        ("num_rejected_steps", numRejectedSteps)
      ] ++ progressMeterStats progress_meter numSteps
    exact maybeThrowOnFailure throwOnFailure {
      t0 := t0
      t1 := tf
      ts := tsOut
      ys := ysOut
      interpolation := denseInterp
      stats := stats
      result := result
      solverState := if saveSolverState then some statef else none
      controllerState := if saveControllerState then some ctrlState.state else none
      madeJump := if saveMadeJump then some madeJumpf else none
      eventMask := eventMaskOption activeEvents eventMaskF
      eventMaskLast := eventMaskHitOption activeEvents eventMaskLastF
    }

end DiffEq
end torch
