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

private def normalizeSaveTs (ts : Option (Array Time)) : Option (Array Time) :=
  match ts with
  | some xs => if xs.size == 0 then none else some xs
  | none => none

private def constantInterpolation [DiffEqSpace Y] (y : Y) : DenseInterpolation Y := {
  evaluate := fun _t0 t1 _left =>
    match t1 with
    | none => y
    | some _ => DiffEqSpace.sub y y
  derivative := fun _t _left =>
    DiffEqSpace.scale 0.0 (DiffEqSpace.sub y y)
}

private def hasSignChange (v0 v1 : Float) : Bool :=
  v0 == 0.0 || v1 == 0.0 || (v0 < 0.0 && v1 > 0.0) || (v0 > 0.0 && v1 < 0.0)

private def hasDirectedSignChange (v0 v1 : Float) (direction : Option Bool) : Bool :=
  match direction with
  | none => hasSignChange v0 v1
  | some true => v0 <= 0.0 && v1 > 0.0
  | some false => v0 > 0.0 && v1 <= 0.0

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

private def firstTrueIdx? (mask : Array Bool) : Option Nat := Id.run do
  let mut out : Option Nat := none
  for i in [:mask.size] do
    if out.isNone && mask[i]! then
      out := some i
  return out

private def firstTerminatingTrueIdx? {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (mask : Array Bool) : Option Nat := Id.run do
  let mut out : Option Nat := none
  for i in [:events.size] do
    if out.isNone && mask.getD i false then
      match events[i]? with
      | some ev =>
          if ev.terminate then
            out := some i
      | none => pure ()
  return out

private def eventMaskOption {Y Args : Type}
    (events : Array (EventSpec Y Args))
    (mask : Array Bool) : Option (Array Bool) :=
  if events.size == 0 then none else some mask

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

def diffeqsolve {Term Y VF Control Args Controller : Type}
    [DiffEqSpace Y]
    [DiffEqSeminorm Y]
    [DiffEqElem Y]
    [Inhabited Y]
    [StepSizeController Controller]
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
    (throwOnFailure : Bool := false) :
    Solution Y solver.SolverState (StepSizeController.State (C := Controller)) := by
  let activeEvents := configuredEvents event events
  let eventMask0 := initialEventMask activeEvents t0 y0 args
  let terminateAtStart := anyTerminatingEvent activeEvents eventMask0
  let eventMask0? := eventMaskOption activeEvents eventMask0
  let saveatTs := normalizeSaveTs saveat.effectiveTs
  let saveDense := saveat.effectiveDense
  let saveSteps := saveat.stepsEnabled
  let saveAtT0 := saveat.effectiveT0
  let saveAtT1 := saveat.effectiveT1
  let saveValue : Time → Y → Y :=
    match saveFn with
    | none => fun _ y => y
    | some f => fun t y => f t y args
  let stepStats0 : List (String × Nat) :=
    [("num_steps", 0), ("num_accepted_steps", 0), ("num_rejected_steps", 0)]
  let outOfRange :=
    match saveatTs with
    | none => false
    | some ts =>
        let tmin := if t0 <= t1 then t0 else t1
        let tmax := if t0 <= t1 then t1 else t0
        ts.foldl (init := false) (fun acc t => acc || t < tmin || t > tmax)
  if outOfRange then
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
        solverState := if saveat.solverState then initialSolverState else none
        controllerState := if saveat.controllerState then initialControllerState else none
        madeJump := if saveat.madeJump then some initialMadeJump else none
        eventMask := eventMask0?
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
        solverState := if saveat.solverState then some initState else none
        controllerState := if saveat.controllerState then some initCtrl.state else none
        madeJump := if saveat.madeJump then some true else none
        eventMask := eventMask0?
      }
    else
    let denseTsInit := #[t0]
    let denseYsInit := #[y0]
    let denseSegsInit : Array (DenseInterpolation Y) := #[]
    let stepTsInit := if saveSteps then #[t0] else #[]
    let stepYsInit := if saveSteps then #[saveValue t0 y0] else #[]
    let rec loop (attempted : Nat) (accepted : Nat) (rejected : Nat) (t : Time) (y : Y)
        (state : solver.SolverState)
        (ctrlState : StepSizeState (StepSizeController.State (C := Controller)))
        (madeJump : Bool)
        (denseTs : Array Time) (denseYs : Array Y)
        (denseSegs : Array (DenseInterpolation Y))
        (stepTs : Array Time) (stepYs : Array Y)
        (eventMaskAcc : Array Bool) :
        (Time × Y × solver.SolverState ×
          StepSizeState (StepSizeController.State (C := Controller)) ×
          Bool × Array Time × Array Y × Array (DenseInterpolation Y) ×
          Array Time × Array Y × Result × Nat × Nat × Nat × Array Bool) :=
      if attempted >= maxSteps then
        (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
          Result.maxStepsReached, attempted, accepted, rejected, eventMaskAcc)
      else if rejected >= maxSteps then
        (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
          Result.maxStepsRejected, attempted, accepted, rejected, eventMaskAcc)
      else
        let dt := ctrlState.dt
        let done := if dt > 0.0 then t >= t1 else t <= t1
        if done then
          (t, y, state, ctrlState, madeJump, denseTs, denseYs, denseSegs, stepTs, stepYs,
            Result.successful, attempted, accepted, rejected, eventMaskAcc)
        else
          let tCandidate := t + dt
          let tNext :=
            if dt > 0.0 then
              if tCandidate > t1 then t1 else tCandidate
            else
              if tCandidate < t1 then t1 else tCandidate
          let attemptedNext := attempted + 1
          let step := solver.step terms t tNext y args state madeJump
          if step.result != Result.successful then
            (tNext, step.y1, step.solverState, ctrlState, madeJump, denseTs, denseYs, denseSegs,
              stepTs, stepYs, step.result, attemptedNext, accepted, rejected, eventMaskAcc)
          else
            let decision :=
              StepSizeController.adapt controller ctrlState t tNext y step.y1 step.yError errorOrder
            if decision.result != Result.successful then
              (tNext, step.y1, step.solverState, ctrlState, madeJump, denseTs, denseYs, denseSegs,
                stepTs, stepYs, decision.result, attemptedNext, accepted, rejected, eventMaskAcc)
            else if decision.accept then
              let ctrlState := { dt := decision.dt, state := decision.state }
              let denseStep := solver.interpolation step.denseInfo
              let madeJumpNext := decision.madeJump
              let acceptedNext := accepted + 1
              let eventHits :=
                activeEvents.map (fun ev =>
                  match ev.condition with
                  | .boolean cond =>
                      let c0 := cond t y args
                      let c1 := cond tNext step.y1 args
                      (!c0) && c1
                  | .real cond =>
                      let v0 := cond t y args
                      let v1 := cond tNext step.y1 args
                      hasDirectedSignChange v0 v1 ev.direction)
              let eventMaskAcc := mergeEventMasks eventMaskAcc eventHits
              let terminatingHitIdx? := firstTerminatingTrueIdx? activeEvents eventHits
              let chosenHitIdx? :=
                match terminatingHitIdx? with
                | some idx => some idx
                | none => firstTrueIdx? eventHits
              let (eventHit, chosenTerminates, eventTime, eventY) :=
                match chosenHitIdx? with
                | none => (false, false, tNext, step.y1)
                | some idx =>
                    match activeEvents[idx]? with
                    | none => (false, false, tNext, step.y1)
                    | some ev =>
                        let eventTimeY :=
                          match ev.condition with
                          | .boolean _ => (tNext, step.y1)
                          | .real cond =>
                              let v0 := cond t y args
                              let v1 := cond tNext step.y1 args
                              if hasDirectedSignChange v0 v1 ev.direction then
                                localizeSignChange
                                  denseStep cond args t tNext v0 v1 ev.rootMaxIters ev.rootTol
                              else
                                (tNext, step.y1)
                        let terminates :=
                          match terminatingHitIdx? with
                          | some _ => true
                          | none => false
                        (true, terminates, eventTimeY.fst, eventTimeY.snd)
              let denseTs := denseTs.push eventTime
              let denseYs := denseYs.push eventY
              let denseSegs := denseSegs.push denseStep
              let stepSaved := saveat.shouldSaveAcceptedStep acceptedNext
              let (stepTs, stepYs) :=
                if stepSaved then
                  (stepTs.push eventTime, stepYs.push (saveValue eventTime eventY))
                else
                  (stepTs, stepYs)
              if eventHit then
                if chosenTerminates then
                  (eventTime, eventY, step.solverState, ctrlState, true, denseTs, denseYs, denseSegs,
                    stepTs, stepYs, Result.eventOccurred, attemptedNext, acceptedNext, rejected, eventMaskAcc)
                else
                  loop attemptedNext acceptedNext 0 eventTime eventY step.solverState ctrlState true
                    denseTs denseYs denseSegs stepTs stepYs eventMaskAcc
              else
                loop attemptedNext acceptedNext 0 tNext step.y1 step.solverState ctrlState madeJumpNext
                  denseTs denseYs denseSegs stepTs stepYs eventMaskAcc
            else
              let ctrlState := { dt := decision.dt, state := decision.state }
              loop attemptedNext accepted (rejected + 1) t y state ctrlState madeJump
                denseTs denseYs denseSegs stepTs stepYs eventMaskAcc
    let (tf, yf, statef, ctrlState, madeJumpf, denseTs, denseYs, denseSegs, stepTs, stepYs,
        result, numSteps, numAcceptedSteps, numRejectedSteps, eventMaskF) :=
      loop 0 0 0 t0 y0 initState initCtrl initialMadeJump
        denseTsInit denseYsInit denseSegsInit stepTsInit stepYsInit eventMask0
    let denseInterp :=
      if saveDense || saveatTs.isSome then
        if denseSegs.size > 0 then
          some (PiecewiseDenseInterpolation.toDense { ts := denseTs, segments := denseSegs })
        else
          some (constantInterpolation yf)
      else
        none
    let (tsOut, ysOut) := computeOutputs tf yf stepTs stepYs denseInterp
    exact maybeThrowOnFailure throwOnFailure {
      t0 := t0
      t1 := tf
      ts := tsOut
      ys := ysOut
      interpolation := denseInterp
      stats := [
        ("num_steps", numSteps),
        ("num_accepted_steps", numAcceptedSteps),
        ("num_rejected_steps", numRejectedSteps)
      ]
      result := result
      solverState := if saveat.solverState then some statef else none
      controllerState := if saveat.controllerState then some ctrlState.state else none
      madeJump := if saveat.madeJump then some madeJumpf else none
      eventMask := eventMaskOption activeEvents eventMaskF
    }

end DiffEq
end torch
