import Tyr.DiffEq.SaveAt
import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Interpolation
import Tyr.DiffEq.StepSizeController

namespace torch
namespace DiffEq

/-! ## Integration Entry Point -/

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
    (controller : Controller := default) :
    Solution Y solver.SolverState (StepSizeController.State (C := Controller)) := by
  let saveatTs :=
    match saveat.ts with
    | some ts => if ts.size == 0 then none else some ts
    | none => none
  let outOfRange :=
    match saveatTs with
    | none => false
    | some ts =>
        let tmin := if t0 <= t1 then t0 else t1
        let tmax := if t0 <= t1 then t1 else t0
        ts.foldl (init := false) (fun acc t => acc || t < tmin || t > tmax)
  if outOfRange then
    exact {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := []
      result := Result.internalError
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := none
    }
  else if dt0 == some 0.0 then
    exact {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := []
      result := Result.dtMinReached
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := none
    }
  else
    let computeOutputs :=
      fun (tf : Time) (yf : Y) (denseTs : Array Time) (denseYs : Array Y)
          (denseInterp : Option (DenseInterpolation Y)) =>
        match saveatTs with
        | some ts =>
            let tsOut :=
              if saveat.t0 || saveat.t1 then
                let pre := if saveat.t0 then #[t0] else #[]
                let post := if saveat.t1 then #[tf] else #[]
                pre ++ ts ++ post
              else
                ts
            let ys :=
              match denseInterp with
              | none => #[]
              | some interp => tsOut.map (fun t => interp.evaluate t none true)
            (some tsOut, some ys)
        | none =>
            if saveat.steps then
              (some denseTs, some denseYs)
            else if saveat.t0 || saveat.t1 then
              let ts :=
                if saveat.t0 && saveat.t1 then #[t0, tf]
                else if saveat.t0 then #[t0] else #[tf]
              let ys :=
                if saveat.t0 && saveat.t1 then #[y0, yf]
                else if saveat.t0 then #[y0] else #[yf]
              (some ts, some ys)
            else
              (none, none)
    if t0 == t1 then
      let denseTs := #[t0]
      let denseYs := #[y0]
      let denseInterp :=
        if saveat.dense || saveatTs.isSome then
          some ({
            evaluate := fun _t t1 _left =>
              match t1 with
              | none => y0
              | some _t1 => DiffEqSpace.sub y0 y0
            derivative := fun _t _left =>
              DiffEqSpace.scale 0.0 (DiffEqSpace.sub y0 y0)
          } : DenseInterpolation Y)
        else
          none
      let (tsOut, ysOut) := computeOutputs t0 y0 denseTs denseYs denseInterp
      exact {
        t0 := t0
        t1 := t1
        ts := tsOut
        ys := ysOut
        interpolation := denseInterp
        stats := [("num_steps", 0)]
        result := Result.successful
        solverState := none
        controllerState := none
        madeJump := none
        eventMask := none
      }
    else
    let errorOrder := solver.errorOrder terms
    let initCtrl := StepSizeController.init controller terms t0 t1 y0 args dt0 solver.func errorOrder
    let dtAbs := if initCtrl.dt < 0.0 then -initCtrl.dt else initCtrl.dt
    let dt := if t1 >= t0 then dtAbs else -dtAbs
    let initCtrl := { initCtrl with dt := dt }
    let initState := solver.init terms t0 t1 y0 args
    let denseTsInit := #[t0]
    let denseYsInit := #[y0]
    let rec loop (i : Nat) (rejects : Nat) (t : Time) (y : Y)
        (state : solver.SolverState)
        (ctrlState : StepSizeState (StepSizeController.State (C := Controller)))
        (denseTs : Array Time) (denseYs : Array Y) :
        (Time × Y × solver.SolverState ×
          StepSizeState (StepSizeController.State (C := Controller)) ×
          Array Time × Array Y × Result × Nat × Nat) :=
      if i >= maxSteps then
        (t, y, state, ctrlState, denseTs, denseYs, Result.maxStepsReached, i, rejects)
      else if rejects >= maxSteps then
        (t, y, state, ctrlState, denseTs, denseYs, Result.maxStepsRejected, i, rejects)
      else
        let dt := ctrlState.dt
        let done := if dt > 0.0 then t >= t1 else t <= t1
        if done then
          (t, y, state, ctrlState, denseTs, denseYs, Result.successful, i, rejects)
        else
          let tCandidate := t + dt
          let tNext :=
            if dt > 0.0 then
              if tCandidate > t1 then t1 else tCandidate
            else
              if tCandidate < t1 then t1 else tCandidate
          let step := solver.step terms t tNext y args state false
          if step.result != Result.successful then
            (tNext, step.y1, step.solverState, ctrlState, denseTs, denseYs, step.result, i, rejects)
          else
            let decision := StepSizeController.adapt controller ctrlState t tNext y step.y1 step.yError errorOrder
            if decision.result != Result.successful then
              (tNext, step.y1, step.solverState, ctrlState, denseTs, denseYs, decision.result, i, rejects)
            else if decision.accept then
              let ctrlState := { dt := decision.dt, state := decision.state }
              let denseTs := denseTs.push tNext
              let denseYs := denseYs.push step.y1
              loop (i + 1) 0 tNext step.y1 step.solverState ctrlState denseTs denseYs
            else
              let ctrlState := { dt := decision.dt, state := decision.state }
              loop i (rejects + 1) t y state ctrlState denseTs denseYs
    let (tf, yf, statef, ctrlState, denseTs, denseYs, result, steps, _rejects) :=
      loop 0 0 t0 y0 initState initCtrl denseTsInit denseYsInit
    let denseInterp :=
      if saveat.dense || saveatTs.isSome then
        some (LinearInterpolation.toDense { ts := denseTs, ys := denseYs })
      else
        none
    let (tsOut, ysOut) := computeOutputs tf yf denseTs denseYs denseInterp
    exact {
      t0 := t0
      t1 := tf
      ts := tsOut
      ys := ysOut
      interpolation := denseInterp
      stats := [("num_steps", steps)]
      result := result
      solverState := if saveat.solverState then some statef else none
      controllerState := if saveat.controllerState then some ctrlState.state else none
      madeJump := none
      eventMask := none
    }

end DiffEq
end torch
