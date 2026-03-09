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

private def clampNonnegativeTime (value : Time) : Time :=
  if value < 0.0 then 0.0 else value

namespace EventSpec

/--
Construct a steady-state event condition from a term bundle and solver function.

The condition is `vfNorm(f(t, y, args)) < atol + rtol * stateNorm(y)`, where
`f` is `solver.func terms`.

- `rtol` and `atol` are clamped to nonnegative values.
- This is a boolean event: checks occur on accepted step endpoints (plus `(t0, y0)`),
  so no real-valued root localization is performed.
- With default `direction := none`, a true condition at `(t0, y0)` triggers
  immediately.
-/
def steadyStateWithNorms {Term Y VF Control Args : Type}
    (terms : Term)
    (solver : AbstractSolver Term Y VF Control Args)
    (vfNorm : VF → Time)
    (stateNorm : Y → Time)
    (rtol atol : Time)
    (terminate : Bool := true)
    (direction : Option Bool := none) : EventSpec Y Args :=
  let rtol := clampNonnegativeTime rtol
  let atol := clampNonnegativeTime atol
  {
    condition := .boolean (fun t y args =>
      let vf := solver.func terms t y args
      vfNorm vf < atol + rtol * stateNorm y)
    terminate := terminate
    direction := direction
  }

/--
Construct a steady-state event condition using `DiffEqSeminorm.rms` for both state
and vector-field norms.
-/
def steadyState {Term Y VF Control Args : Type}
    [DiffEqSeminorm Y]
    [DiffEqSeminorm VF]
    (terms : Term)
    (solver : AbstractSolver Term Y VF Control Args)
    (rtol atol : Time)
    (terminate : Bool := true)
    (direction : Option Bool := none) : EventSpec Y Args :=
  steadyStateWithNorms
    terms solver DiffEqSeminorm.rms DiffEqSeminorm.rms rtol atol
    terminate direction

end EventSpec

/-- Tree-structured event configuration for PyTree-style parity with diffrax. -/
inductive EventTree (Y Args : Type) where
  | leaf (event : EventSpec Y Args)
  | branch (children : Array (EventTree Y Args))
  deriving Inhabited

/-- Tree-structured event mask aligned with an `EventTree` layout. -/
inductive EventMaskTree where
  | leaf (hit : Bool)
  | branch (children : Array EventMaskTree)
  deriving Inhabited, Repr, BEq

namespace EventTree

private def flattenAux {Y Args : Type}
    (tree : EventTree Y Args)
    (acc : Array (EventSpec Y Args)) : Array (EventSpec Y Args) :=
  match tree with
  | .leaf ev => acc.push ev
  | .branch children =>
      children.foldl (fun acc child => flattenAux child acc) acc

/-- Flatten a tree of events in left-to-right depth-first leaf order. -/
def flatten {Y Args : Type} (tree : EventTree Y Args) : Array (EventSpec Y Args) :=
  flattenAux tree #[]

private def maskTreeFromFlatAux {Y Args : Type}
    (tree : EventTree Y Args)
    (mask : Array Bool)
    (idx : Nat) : EventMaskTree × Nat :=
  match tree with
  | .leaf _ =>
      (.leaf (mask.getD idx false), idx + 1)
  | .branch children =>
      let (out, next) :=
        children.foldl
          (fun (state : Array EventMaskTree × Nat) child =>
            let (acc, i) := state
            let (childTree, iNext) := maskTreeFromFlatAux child mask i
            (acc.push childTree, iNext))
          (#[], idx)
      (.branch out, next)

/-- Reconstruct a tree-shaped mask from a flattened event mask. -/
def maskTreeFromFlat {Y Args : Type}
    (tree : EventTree Y Args)
    (mask : Array Bool) : EventMaskTree :=
  (maskTreeFromFlatAux tree mask 0).fst

def liftMaskTree? {Y Args : Type}
    (tree : EventTree Y Args)
    (mask? : Option (Array Bool)) : Option EventMaskTree :=
  mask?.map (fun mask => maskTreeFromFlat tree mask)

end EventTree

structure EventTreeSolution (Y SolverState ControllerState : Type) where
  base : Solution Y SolverState ControllerState
  eventMaskTree : Option EventMaskTree
  eventMaskLastTree : Option EventMaskTree
  deriving Inhabited

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

private def hasDirectedBoolEdge (c0 c1 : Bool) (direction : Option Bool) : Bool :=
  match direction with
  | none => (!c0) && c1
  | some true => (!c0) && c1
  | some false => c0 && (!c1)

private def booleanHitAtStart (c0 : Bool) (direction : Option Bool) : Bool :=
  match direction with
  | some false => false
  | _ => c0

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
    (_hits : Array (StepEventHit Y))
    (chosen : StepEventHit Y) : Array Bool := Id.run do
  let mut mask := Array.replicate events.size false
  if chosen.idx < mask.size then
    -- diffrax-style event_mask semantics: commit only the selected trigger.
    mask := mask.set! chosen.idx true
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
  | .boolean cond => booleanHitAtStart (cond t0 y0 args) ev.direction
  | .real _ => false

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

private def clampProgressUnitInterval (value : Time) : Time :=
  if value < 0.0 then
    0.0
  else if value > 1.0 then
    1.0
  else
    value

private def solveProgressFraction (t0 t1 tNow : Time) : Time :=
  if t1 == t0 then
    1.0
  else
    clampProgressUnitInterval ((tNow - t0) / (t1 - t0))

private def progressMeterMinimumIncrease : Time :=
  0.0999

private def progressMeterTerminalTol : Time :=
  1.0e-12

private def progressMeterReachedTerminal (progress : Time) : Bool :=
  progress >= 1.0 - progressMeterTerminalTol

private structure ProgressMeterRuntime where
  mode : ProgressMeter
  t0 : Time
  t1 : Time
  lastProgress : Time := 0.0
  renderedUpdates : Nat := 0
  renderedCloseTerminal : Nat := 0
  renderedPoints : Nat := 0

private def progressMeterStart (mode : ProgressMeter) (t0 t1 : Time) : ProgressMeterRuntime := Id.run do
  if mode.textEnabled then
    if mode == .tqdm then
      dbg_trace "[DiffEq progress_meter=tqdm] using text compatibility fallback."
    dbg_trace "[DiffEq progress_meter=text] 0.00%"
    return {
      mode := mode
      t0 := t0
      t1 := t1
      renderedPoints := 1
    }
  return {
    mode := mode
    t0 := t0
    t1 := t1
  }

private def progressMeterShouldRenderUpdate (lastProgress nextProgress : Time) : Bool :=
  (nextProgress - lastProgress > progressMeterMinimumIncrease) || progressMeterReachedTerminal nextProgress

private def progressMeterUpdate
    (state : ProgressMeterRuntime)
    (attempted : Nat)
    (tNow : Time) : ProgressMeterRuntime := Id.run do
  if state.mode.textEnabled then
    let progress := solveProgressFraction state.t0 state.t1 tNow
    if progressMeterShouldRenderUpdate state.lastProgress progress then
      dbg_trace s!"[DiffEq progress_meter=text] {100.0 * progress}% (steps={attempted})"
      return {
        state with
        lastProgress := progress
        renderedUpdates := state.renderedUpdates + 1
        renderedPoints := state.renderedPoints + 1
      }
    else
      return state
  else
    return state

private def progressMeterClose
    (state : ProgressMeterRuntime)
    (result : Result)
    (numSteps numAcceptedSteps numRejectedSteps : Nat) : ProgressMeterRuntime := Id.run do
  if state.mode.textEnabled then
    let mut next := state
    if !progressMeterReachedTerminal state.lastProgress then
      dbg_trace "[DiffEq progress_meter=text] 100.00% (close)"
      next := {
        next with
        lastProgress := 1.0
        renderedCloseTerminal := 1
        renderedPoints := next.renderedPoints + 1
      }
    dbg_trace
      s!"[DiffEq progress_meter=text] close steps={numSteps} accepted={numAcceptedSteps} rejected={numRejectedSteps} result={repr result}"
    return next
  else
    return state

private def progressMeterStats
    (state : ProgressMeterRuntime)
    (numSteps : Nat) : List (String × Nat) :=
  if state.mode.textEnabled then
    let aliasStats :=
      if state.mode == .tqdm then
        [("progress_meter_tqdm_alias", 1)]
      else
        []
    [
      ("progress_meter_start", 1),
      ("progress_meter_updates", numSteps),
      ("progress_meter_close", 1),
      ("progress_meter_rendered_updates", state.renderedUpdates),
      ("progress_meter_rendered_close_terminal", state.renderedCloseTerminal),
      ("progress_meter_rendered_points", state.renderedPoints)
    ] ++ aliasStats
  else
    []

private def maybeThrowOnFailure {Y SolverState ControllerState : Type}
    (throwOnFailure : Bool)
    (sol : Solution Y SolverState ControllerState) :
    Solution Y SolverState ControllerState :=
  if throwOnFailure then
    match sol.toExcept with
    | .ok okSol => okSol
    | .error _ => sol
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
      if hasDirectedBoolEdge c0 c1 ev.direction then
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

private def collectStepHits {Y Args : Type}
    [DiffEqSpace Y] [Inhabited Y]
    (denseStep : DenseInterpolation Y)
    (activeEvents : Array (EventSpec Y Args))
    (t0 t1 : Time)
    (y0 y1 : Y)
    (args : Args) : Array (StepEventHit Y) := Id.run do
  let mut hits : Array (StepEventHit Y) := #[]
  for i in [:activeEvents.size] do
    match activeEvents[i]? with
    | none => pure ()
    | some ev =>
        match stepEventHit? denseStep ev i t0 t1 y0 y1 args with
        | none => pure ()
        | some hit => hits := hits.push hit
  return hits

structure SolveLoopAttempt (Y ControllerState : Type) where
  tStart : Time
  yStart : Y
  tAttempt : Time
  yAttempt : Y
  tAfter : Time
  yAfter : Y
  controllerStateBefore : StepSizeState ControllerState
  controllerStateAfter : StepSizeState ControllerState
  accepted : Bool
  stepResult : Result
  decisionResult : Result
  madeJumpBefore : Bool
  madeJumpAfter : Bool

structure SolveLoopState (Y SolverState ControllerState : Type) where
  attempted : Nat := 0
  accepted : Nat := 0
  rejected : Nat := 0
  t : Time
  y : Y
  solverState : SolverState
  controllerState : StepSizeState ControllerState
  madeJump : Bool := false
  denseTs : Array Time := #[]
  denseYs : Array Y := #[]
  denseSegs : Array (DenseInterpolation Y) := #[]
  stepTs : Array Time := #[]
  stepYs : Array Y := #[]
  attempts : Array (SolveLoopAttempt Y ControllerState) := #[]
  eventMaskAcc : Array Bool := #[]
  eventMaskLastAcc : Array Bool := #[]
  result : Result := Result.successful
  continueLoop : Bool := true
  progressState : ProgressMeterRuntime

def initialSolveLoopState
    {Y SolverState ControllerState : Type}
    (t0 : Time)
    (t1 : Time)
    (y0 : Y)
    (solverState : SolverState)
    (controllerState : StepSizeState ControllerState)
    (madeJump : Bool)
    (eventMask0 : Array Bool)
    (progress_meter : ProgressMeter := .none) :
    SolveLoopState Y SolverState ControllerState := {
      t := t0
      y := y0
      solverState := solverState
      controllerState := controllerState
      madeJump := madeJump
      denseTs := #[t0]
      denseYs := #[y0]
      stepTs := #[t0]
      stepYs := #[y0]
      attempts := #[]
      eventMaskAcc := eventMask0
      eventMaskLastAcc := eventMask0
      progressState := progressMeterStart progress_meter t0 t1
    }

def solveLoopStep
    {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    (terms : Term)
    (solver : AbstractSolver Term Y VF Control Args)
    (controller : Controller)
    (t1 : Time)
    (args : Args)
    (maxStepsOpt : Option Nat)
    (activeEvents : Array (EventSpec Y Args))
    (errorOrder : Float)
    (loop : SolveLoopState Y solver.SolverState (StepSizeController.State (C := Controller))) :
    SolveLoopState Y solver.SolverState (StepSizeController.State (C := Controller)) :=
  let loop :=
    match maxStepsOpt with
    | some maxStepBudget =>
        if loop.attempted >= maxStepBudget then
          { loop with result := Result.maxStepsReached, continueLoop := false }
        else if loop.rejected >= maxStepBudget then
          { loop with result := Result.maxStepsRejected, continueLoop := false }
        else
          loop
    | none => loop
  if !loop.continueLoop then
    loop
  else
    let controllerStateBefore := loop.controllerState
    let dt := controllerStateBefore.dt
    let done := if dt > 0.0 then loop.t >= t1 else loop.t <= t1
    if done then
      { loop with result := Result.successful, continueLoop := false }
    else
      let tCandidate := loop.t + dt
      let tNext :=
        if dt > 0.0 then
          if tCandidate > t1 then t1 else tCandidate
        else
          if tCandidate < t1 then t1 else tCandidate
      let attemptedNext := loop.attempted + 1
      if tNext == loop.t then
        { loop with
          attempted := attemptedNext
          progressState := progressMeterUpdate loop.progressState attemptedNext loop.t
          result := Result.dtMinReached
          continueLoop := false
        }
      else
        let step := solver.step terms loop.t tNext loop.y args loop.solverState loop.madeJump
        if step.result != Result.successful then
          let attempt : SolveLoopAttempt Y (StepSizeController.State (C := Controller)) := {
            tStart := loop.t
            yStart := loop.y
            tAttempt := tNext
            yAttempt := step.y1
            tAfter := tNext
            yAfter := step.y1
            controllerStateBefore := controllerStateBefore
            controllerStateAfter := controllerStateBefore
            accepted := false
            stepResult := step.result
            decisionResult := step.result
            madeJumpBefore := loop.madeJump
            madeJumpAfter := loop.madeJump
          }
          { loop with
            attempted := attemptedNext
            t := tNext
            y := step.y1
            solverState := step.solverState
            attempts := loop.attempts.push attempt
            progressState := progressMeterUpdate loop.progressState attemptedNext tNext
            result := step.result
            continueLoop := false
          }
        else
          let decision :=
            StepSizeController.adapt controller loop.controllerState
              loop.t tNext loop.y step.y1 step.yError errorOrder
          if decision.result != Result.successful then
            let attempt : SolveLoopAttempt Y (StepSizeController.State (C := Controller)) := {
              tStart := loop.t
              yStart := loop.y
              tAttempt := tNext
              yAttempt := step.y1
              tAfter := tNext
              yAfter := step.y1
              controllerStateBefore := controllerStateBefore
              controllerStateAfter := controllerStateBefore
              accepted := false
              stepResult := step.result
              decisionResult := decision.result
              madeJumpBefore := loop.madeJump
              madeJumpAfter := loop.madeJump
            }
            { loop with
              attempted := attemptedNext
              t := tNext
              y := step.y1
              solverState := step.solverState
              attempts := loop.attempts.push attempt
              progressState := progressMeterUpdate loop.progressState attemptedNext tNext
              result := decision.result
              continueLoop := false
            }
          else if decision.accept then
            let controllerStateNext : StepSizeState (StepSizeController.State (C := Controller)) := {
              dt := decision.dt
              state := decision.state
            }
            let denseStep := solver.interpolation step.denseInfo
            let stepHits := collectStepHits denseStep activeEvents loop.t tNext loop.y step.y1 args
            let forward := tNext >= loop.t
            let chosenHit? := chooseStepEventHit? forward stepHits
            let stepMaskChosen :=
              match chosenHit? with
              | none => Array.replicate activeEvents.size false
              | some chosen => maskAtEventTime activeEvents stepHits chosen
            let eventMaskAccNext := mergeEventMasks loop.eventMaskAcc stepMaskChosen
            let eventMaskLastAccNext :=
              match chosenHit? with
              | none => loop.eventMaskLastAcc
              | some _ => stepMaskChosen
            let chosenTerminates := anyTerminatingEvent activeEvents stepMaskChosen
            let (eventHit, chosenTime, chosenY, chosenTol) :=
              match chosenHit? with
              | none => (false, tNext, step.y1, 0.0)
              | some chosen => (true, chosen.time, chosen.y, chosen.tol)
            let noProgressEvent := eventHit && hasNoTimeProgress loop.t chosenTime chosenTol
            let restartAtLocalized := eventHit && !chosenTerminates && !noProgressEvent
            let stepOutTime :=
              if restartAtLocalized || chosenTerminates then chosenTime else tNext
            let stepOutY :=
              if restartAtLocalized || chosenTerminates then chosenY else step.y1
            let nextT := if eventHit then stepOutTime else tNext
            let nextY := if eventHit then stepOutY else step.y1
            let nextMadeJump := if eventHit then true else decision.madeJump
            let nextResult :=
              if eventHit && chosenTerminates then Result.eventOccurred else Result.successful
            let nextContinue := !(eventHit && chosenTerminates)
            let attempt : SolveLoopAttempt Y (StepSizeController.State (C := Controller)) := {
              tStart := loop.t
              yStart := loop.y
              tAttempt := tNext
              yAttempt := step.y1
              tAfter := nextT
              yAfter := nextY
              controllerStateBefore := controllerStateBefore
              controllerStateAfter := controllerStateNext
              accepted := true
              stepResult := step.result
              decisionResult := decision.result
              madeJumpBefore := loop.madeJump
              madeJumpAfter := nextMadeJump
            }
            { loop with
              attempted := attemptedNext
              accepted := loop.accepted + 1
              rejected := 0
              t := nextT
              y := nextY
              solverState := step.solverState
              controllerState := controllerStateNext
              madeJump := nextMadeJump
              denseTs := loop.denseTs.push stepOutTime
              denseYs := loop.denseYs.push stepOutY
              denseSegs := loop.denseSegs.push denseStep
              stepTs := loop.stepTs.push stepOutTime
              stepYs := loop.stepYs.push stepOutY
              attempts := loop.attempts.push attempt
              eventMaskAcc := eventMaskAccNext
              eventMaskLastAcc := eventMaskLastAccNext
              result := nextResult
              continueLoop := nextContinue
              progressState := progressMeterUpdate loop.progressState attemptedNext nextT
            }
          else
            let controllerStateNext : StepSizeState (StepSizeController.State (C := Controller)) := {
              dt := decision.dt
              state := decision.state
            }
            let attempt : SolveLoopAttempt Y (StepSizeController.State (C := Controller)) := {
              tStart := loop.t
              yStart := loop.y
              tAttempt := tNext
              yAttempt := step.y1
              tAfter := loop.t
              yAfter := loop.y
              controllerStateBefore := controllerStateBefore
              controllerStateAfter := controllerStateNext
              accepted := false
              stepResult := step.result
              decisionResult := decision.result
              madeJumpBefore := loop.madeJump
              madeJumpAfter := loop.madeJump
            }
            { loop with
              attempted := attemptedNext
              rejected := loop.rejected + 1
              controllerState := controllerStateNext
              attempts := loop.attempts.push attempt
              progressState := progressMeterUpdate loop.progressState attemptedNext loop.t
            }

def runSolveLoop
    {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
    (terms : Term)
    (solver : AbstractSolver Term Y VF Control Args)
    (controller : Controller)
    (t1 : Time)
    (args : Args)
    (maxStepsOpt : Option Nat)
    (activeEvents : Array (EventSpec Y Args))
    (errorOrder : Float)
    (loop0 : SolveLoopState Y solver.SolverState (StepSizeController.State (C := Controller))) :
    SolveLoopState Y solver.SolverState (StepSizeController.State (C := Controller)) := Id.run do
  let mut loop := loop0
  while loop.continueLoop do
    loop := solveLoopStep terms solver controller t1 args maxStepsOpt activeEvents errorOrder loop
  return loop

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
    (eventTree : Option (EventTree Y Args) := none)
    (saveFn : Option (Time → Y → Args → Y) := none)
    (throwOnFailure : Bool := false)
    (maxStepsOpt : Option Nat := some maxSteps)
    (progress_meter : ProgressMeter := .none) :
    Solution Y solver.SolverState (StepSizeController.State (C := Controller)) := by
  let treeEvents :=
    match eventTree with
    | some tree => EventTree.flatten tree
    | none => #[]
  let activeEvents := configuredEvents event (events ++ treeEvents)
  let eventMask0 := initialEventMask activeEvents t0 y0 args
  let terminateAtStart := anyTerminatingEvent activeEvents eventMask0
  let eventMask0? := eventMaskOption activeEvents eventMask0
  let eventMaskLast0? := eventMaskHitOption activeEvents eventMask0
  let payloadSubs := saveat.payloadSubs
  let saveatTs := normalizeSaveTs saveat.effectiveTs
  let saveDense := saveat.effectiveDense
  let saveSteps := saveat.stepsEnabled
  let saveSolverState := saveat.effectiveSolverState
  let saveControllerState := saveat.effectiveControllerState
  let saveMadeJump := saveat.effectiveMadeJump
  let isUnbounded := maxStepsOpt.isNone
  let saveConfigIncompatibleWithUnbounded := isUnbounded && (saveSteps || saveDense)
  let hasEmptyLeafSub := saveat.hasEmptyLeafSub
  let saveValue : Time → Y → Y :=
    match saveFn with
    | none => fun _ y => y
    | some f => fun t y => f t y args
  let stepStats0 : List (String × Nat) :=
    [("num_steps", 0), ("num_accepted_steps", 0), ("num_rejected_steps", 0)]
  let tmin := if t0 <= t1 then t0 else t1
  let tmax := if t0 <= t1 then t1 else t0
  let outOfRange :=
    payloadSubs.any (fun sub =>
      match sub.ts with
      | none => false
      | some ts => ts.any (fun t => t < tmin || t > tmax))
  let badDirection :=
    payloadSubs.any (fun sub =>
      match sub.ts with
      | none => false
      | some ts => !(isMonotoneInSolveDirection t0 t1 ts))
  if outOfRange || badDirection || saveConfigIncompatibleWithUnbounded || hasEmptyLeafSub then
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
  else
    let computeOutputs :=
      fun (tf : Time) (yf : Y) (allStepTs : Array Time) (allStepYs : Array Y)
          (denseInterp : Option (DenseInterpolation Y)) =>
        let terminalTol : Time := 1.0e-6
        let forward := tf >= t0
        let clippedTerminal := !timesWithinTol tf t1 terminalTol
        let timeNotPastTerminal : Time → Bool :=
          if forward then
            fun t => t <= tf + terminalTol
          else
            fun t => t >= tf - terminalTol
        let evalFromDense : Time → Y :=
          fun t =>
            match denseInterp with
            | some interp => saveValue t (interp.evaluate t none true)
            | none =>
                if t == t0 then
                  saveValue t0 y0
                else if t == tf then
                  saveValue tf yf
                else
                  saveValue t yf
        let (flatTs, flatYs) := Id.run do
          let mut outTs : Array Time := #[]
          let mut outYs : Array Y := #[]
          let stepCount := Nat.min allStepTs.size allStepYs.size
          for sub in payloadSubs do
            let mut subTs : Array Time := #[]
            let mut subYs : Array Y := #[]
            if sub.t0 then
              subTs := subTs.push t0
              subYs := subYs.push (saveValue t0 y0)
            match sub.ts with
            | none => pure ()
            | some ts =>
                for t in ts do
                  if timeNotPastTerminal t then
                    subTs := subTs.push t
                    subYs := subYs.push (evalFromDense t)
            let cadence : Nat := sub.steps
            if cadence != 0 then
              for i in [:stepCount] do
                if i > 0 && i % cadence == 0 then
                  let tStep := allStepTs[i]!
                  let yStep := allStepYs[i]!
                  subTs := subTs.push tStep
                  subYs := subYs.push (saveValue tStep yStep)
            if sub.t1 then
              let t1AlreadySaved :=
                match subTs.back? with
                | some lastT =>
                    timesWithinTol lastT tf terminalTol &&
                      (clippedTerminal || cadence != 0)
                | none => false
              if !t1AlreadySaved then
                subTs := subTs.push tf
                subYs := subYs.push (saveValue tf yf)
            outTs := outTs ++ subTs
            outYs := outYs ++ subYs
          return (outTs, outYs)
        if flatTs.size == 0 then
          (none, none)
        else
          (some flatTs, some flatYs)
    if t0 == t1 then
      let result0 := if terminateAtStart then Result.eventOccurred else Result.successful
      let progressState0 := progressMeterStart progress_meter t0 t1
      let progressStateF := progressMeterClose progressState0 result0 0 0 0
      let stats0 := stepStats0 ++ progressMeterStats progressStateF 0
      let denseInterp :=
        if saveDense || saveatTs.isSome then
          some (constantInterpolation y0)
        else
          none
      let stepTs := #[t0]
      let stepYs := #[y0]
      let (tsOut, ysOut) := computeOutputs t0 y0 stepTs stepYs denseInterp
      exact maybeThrowOnFailure throwOnFailure {
        t0 := t0
        t1 := t1
        ts := tsOut
        ys := ysOut
        interpolation := denseInterp
        stats := stats0
        result := result0
        solverState := if saveSolverState then initialSolverState else none
        controllerState := if saveControllerState then initialControllerState else none
        madeJump := if saveMadeJump then some initialMadeJump else none
        eventMask := eventMask0?
        eventMaskLast := eventMaskLast0?
      }
    else
      let controllerInvalid :=
        (inferInstance : StepSizeControllerValidation Controller).validate controller t0 t1 dt0
      if controllerInvalid.isSome then
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
          let progressState0 := progressMeterStart progress_meter t0 t1
          let progressStateF := progressMeterClose progressState0 Result.eventOccurred 0 0 0
          let stats0 := stepStats0 ++ progressMeterStats progressStateF 0
          let denseInterp :=
            if saveDense || saveatTs.isSome then
              some (constantInterpolation y0)
            else
              none
          let stepTs := #[t0]
          let stepYs := #[y0]
          let (tsOut, ysOut) := computeOutputs t0 y0 stepTs stepYs denseInterp
          exact maybeThrowOnFailure throwOnFailure {
            t0 := t0
            t1 := t0
            ts := tsOut
            ys := ysOut
            interpolation := denseInterp
            stats := stats0
            result := Result.eventOccurred
            solverState := if saveSolverState then some initState else none
            controllerState := if saveControllerState then some initCtrl.state else none
            madeJump := if saveMadeJump then some true else none
            eventMask := eventMask0?
            eventMaskLast := eventMaskLast0?
          }
        else
          let loopInit :=
            initialSolveLoopState t0 t1 y0 initState initCtrl initialMadeJump eventMask0 progress_meter
          let loopResult :=
            runSolveLoop
              terms solver controller t1 args maxStepsOpt activeEvents errorOrder loopInit
          let tf := loopResult.t
          let yf := loopResult.y
          let statef := loopResult.solverState
          let ctrlState := loopResult.controllerState
          let madeJumpf := loopResult.madeJump
          let denseTs := loopResult.denseTs
          let denseYs := loopResult.denseYs
          let denseSegs := loopResult.denseSegs
          let stepTs := loopResult.stepTs
          let stepYs := loopResult.stepYs
          let result := loopResult.result
          let numSteps := loopResult.attempted
          let numAcceptedSteps := loopResult.accepted
          let numRejectedSteps := loopResult.rejected
          let eventMaskF := loopResult.eventMaskAcc
          let eventMaskLastF := loopResult.eventMaskLastAcc
          let progressStateRaw := loopResult.progressState
          let progressStateF :=
            progressMeterClose progressStateRaw result numSteps numAcceptedSteps numRejectedSteps
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
            ] ++ progressMeterStats progressStateF numSteps
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

/--
Solve with a tree-structured event configuration.

This is a compatibility surface for diffrax-style event PyTrees: the solver runs on
flattened leaves, then event masks are reconstructed with the original tree shape.
-/
def diffeqsolveEventTree {Term Y VF Control Args Controller : Type}
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
    (eventTree : EventTree Y Args)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default)
    (initialSolverState : Option solver.SolverState := none)
    (initialControllerState :
      Option (StepSizeController.State (C := Controller)) := none)
    (initialMadeJump : Bool := false)
    (saveFn : Option (Time → Y → Args → Y) := none)
    (throwOnFailure : Bool := false)
    (maxStepsOpt : Option Nat := some maxSteps)
    (progress_meter : ProgressMeter := .none) :
    EventTreeSolution Y solver.SolverState (StepSizeController.State (C := Controller)) :=
  let base :=
    diffeqsolve
      (Term := Term)
      (Y := Y)
      (VF := VF)
      (Control := Control)
      (Args := Args)
      (Controller := Controller)
      terms solver t0 t1 dt0 y0 args
      (saveat := saveat)
      (maxSteps := maxSteps)
      (controller := controller)
      (event := none)
      (initialSolverState := initialSolverState)
      (initialControllerState := initialControllerState)
      (initialMadeJump := initialMadeJump)
      (events := #[])
      (eventTree := some eventTree)
      (saveFn := saveFn)
      (throwOnFailure := throwOnFailure)
      (maxStepsOpt := maxStepsOpt)
      (progress_meter := progress_meter)
  {
    base := base
    eventMaskTree := EventTree.liftMaskTree? eventTree base.eventMask
    eventMaskLastTree := EventTree.liftMaskTree? eventTree base.eventMaskLast
  }

def diffeqsolveOrError {Term Y VF Control Args Controller : Type}
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
    (eventTree : Option (EventTree Y Args) := none)
    (saveFn : Option (Time → Y → Args → Y) := none)
    (maxStepsOpt : Option Nat := some maxSteps)
    (progress_meter : ProgressMeter := .none) :
    Except SolveError (Solution Y solver.SolverState (StepSizeController.State (C := Controller))) :=
  let sol :=
    diffeqsolve
      (Term := Term)
      (Y := Y)
      (VF := VF)
      (Control := Control)
      (Args := Args)
      (Controller := Controller)
      terms solver t0 t1 dt0 y0 args
      (saveat := saveat)
      (maxSteps := maxSteps)
      (controller := controller)
      (event := event)
      (initialSolverState := initialSolverState)
      (initialControllerState := initialControllerState)
      (initialMadeJump := initialMadeJump)
      (events := events)
      (eventTree := eventTree)
      (saveFn := saveFn)
      (throwOnFailure := false)
      (maxStepsOpt := maxStepsOpt)
      (progress_meter := progress_meter)
  sol.toExcept

end DiffEq
end torch
