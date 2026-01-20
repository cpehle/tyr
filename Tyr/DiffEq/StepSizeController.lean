import Tyr.DiffEq.Types
import Tyr.DiffEq.Solution
import Tyr.DiffEq.Term

namespace torch
namespace DiffEq

/-! ## Step Size Controllers -/

structure StepSizeState (State : Type) where
  dt : Time
  state : State
  deriving Inhabited

structure StepSizeDecision (State : Type) where
  accept : Bool
  dt : Time
  state : State
  result : Result

class InitialStepSelector (Term Y VF Args : Type) where
  select : Term → Time → Y → Args → (Term → Time → Y → Args → VF) →
    Float → Float → Float → Option Time

instance (priority := 5) : InitialStepSelector Term Y VF Args where
  select _ _ _ _ _ _ _ _ := none

private def selectInitialStepODE {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y]
    (term : ODETerm Y Args) (t0 : Time) (y0 : Y) (args : Args)
    (func : ODETerm Y Args → Time → Y → Args → Y)
    (rtol atol errorOrder : Float) : Time :=
  let scale := DiffEqElem.addScalar atol (DiffEqSpace.scale rtol (DiffEqElem.abs y0))
  let d0 := DiffEqSeminorm.rms (DiffEqElem.div y0 scale)
  let f0 := func term t0 y0 args
  let d1 := DiffEqSeminorm.rms (DiffEqElem.div f0 scale)
  let small := d0 < 1.0e-5 || d1 < 1.0e-5
  let d1ForH0 := if small then 1.0 else d1
  let h0 := if small then 1.0e-6 else 0.01 * (d0 / d1ForH0)
  let t1 := t0 + h0
  let y1 := DiffEqSpace.add y0 (DiffEqSpace.scale h0 f0)
  let f1 := func term t1 y1 args
  let diff := DiffEqSpace.sub f1 f0
  let d2 := (DiffEqSeminorm.rms (DiffEqElem.div diff scale)) / h0
  let maxd := if d1 > d2 then d1 else d2
  let h1 :=
    if maxd <= 1.0e-15 then
      let hmin := if h0 * 1.0e-3 > 1.0e-6 then h0 * 1.0e-3 else 1.0e-6
      hmin
    else
      Float.pow (0.01 / maxd) (1.0 / errorOrder)
  if 100.0 * h0 < h1 then 100.0 * h0 else h1

instance [DiffEqSpace Y] [DiffEqSeminorm Y] [DiffEqElem Y] :
    InitialStepSelector (ODETerm Y Args) Y Y Args where
  select term t0 y0 args func rtol atol errorOrder :=
    some (selectInitialStepODE term t0 y0 args func rtol atol errorOrder)

class StepSizeController (C : Type) where
  State : Type
  init : {Term Y VF Args : Type} → [DiffEqSpace Y] → [DiffEqSeminorm Y] → [DiffEqElem Y] →
    C → Term → Time → Time → Y → Args → Option Time →
    (Term → Time → Y → Args → VF) → Float → StepSizeState State
  adapt : {Y : Type} → [DiffEqSpace Y] → [DiffEqSeminorm Y] → [DiffEqElem Y] →
    C → StepSizeState State → Time → Time → Y → Y → Option Y → Float →
    StepSizeDecision State

structure ConstantStepSize where
  deriving Inhabited

instance : StepSizeController ConstantStepSize where
  State := Unit
  init _ _terms _t0 _t1 _y0 _args dt0 _func _errorOrder :=
    match dt0 with
    | some dt => { dt := dt, state := () }
    | none => panic! "ConstantStepSize requires dt0; pass `dt0 := some ...`."
  adapt _ state _t0 _t1 _y0 _y1 _yError _errorOrder :=
    { accept := true, dt := state.dt, state := state.state, result := Result.successful }

structure PIDState where
  prevInvError : Float := 1.0
  prevPrevInvError : Float := 1.0
  deriving Inhabited

structure PIDController where
  rtol : Float := 1.0e-5
  atol : Float := 1.0e-8
  pcoeff : Float := 0.7
  icoeff : Float := 0.4
  dcoeff : Float := 0.0
  dtmin : Option Float := none
  dtmax : Option Float := none
  safety : Float := 0.9
  factormin : Float := 0.2
  factormax : Float := 10.0
  force_dtmin : Bool := false
  deriving Inhabited

private def clampDt (dt : Time) (dtmin dtmax : Option Time) : Time :=
  let sign := if dt < 0.0 then -1.0 else 1.0
  let absDt := if dt < 0.0 then -dt else dt
  let absDt :=
    match dtmin with
    | none => absDt
    | some mn => if absDt < mn then mn else absDt
  let absDt :=
    match dtmax with
    | none => absDt
    | some mx => if absDt > mx then mx else absDt
  sign * absDt

private def errorScale {Y : Type} [DiffEqSpace Y] [DiffEqElem Y]
    (rtol atol : Float) (y0 y1 : Y) : Y :=
  let yAbs := DiffEqElem.max (DiffEqElem.abs y0) (DiffEqElem.abs y1)
  let scaled := DiffEqSpace.scale rtol yAbs
  DiffEqElem.addScalar atol scaled

instance : StepSizeController PIDController where
  State := PIDState
  init ctrl terms t0 _t1 y0 args dt0 func errorOrder :=
    let dt0 :=
      match dt0 with
      | some dt => dt
      | none =>
          match InitialStepSelector.select terms t0 y0 args func ctrl.rtol ctrl.atol errorOrder with
          | some dt => dt
          | none => 0.01
    let dt := clampDt dt0 ctrl.dtmin ctrl.dtmax
    { dt := dt, state := default }
  adapt ctrl state t0 t1 y0 y1 yError errorOrder :=
    match yError with
    | none =>
        { accept := false, dt := state.dt, state := state.state, result := Result.internalError }
    | some yErr =>
        let prevDt := t1 - t0
        let scale := errorScale ctrl.rtol ctrl.atol y0 y1
        let scaledError := DiffEqSeminorm.rms (DiffEqElem.div yErr scale)
        let keep :=
          if let some dtmin := ctrl.dtmin then
            scaledError < 1.0 || Float.abs prevDt <= dtmin
          else
            scaledError < 1.0
        let invScaled :=
          if scaledError <= 0.0 then 1.0 else 1.0 / scaledError
        let coeff1 := (ctrl.icoeff + ctrl.pcoeff + ctrl.dcoeff) / errorOrder
        let coeff2 := -(ctrl.pcoeff + 2.0 * ctrl.dcoeff) / errorOrder
        let coeff3 := ctrl.dcoeff / errorOrder
        let factor1 := if coeff1 == 0.0 then 1.0 else Float.pow invScaled coeff1
        let factor2 :=
          if coeff2 == 0.0 then 1.0 else Float.pow state.state.prevInvError coeff2
        let factor3 :=
          if coeff3 == 0.0 then 1.0 else Float.pow state.state.prevPrevInvError coeff3
        let factormin := if keep then 1.0 else ctrl.factormin
        let factormax := if keep then ctrl.factormax else ctrl.safety
        let factor := ctrl.safety * factor1 * factor2 * factor3
        let factor :=
          if factor < factormin then factormin
          else if factor > factormax then factormax
          else factor
        let dt := clampDt (prevDt * factor) ctrl.dtmin ctrl.dtmax
        let result :=
          match ctrl.dtmin with
          | some dtmin =>
              if !ctrl.force_dtmin && Float.abs dt < dtmin then Result.dtMinReached
              else Result.successful
          | none => Result.successful
        let invScaled := if invScaled == 0.0 then 1.0 else invScaled
        let nextState : PIDState :=
          if keep then
            { prevInvError := invScaled, prevPrevInvError := state.state.prevInvError }
          else
            { prevInvError := state.state.prevInvError
              prevPrevInvError := state.state.prevPrevInvError }
        { accept := keep, dt := dt, state := nextState, result := result }

structure StepTo where
  ts : Array Time := #[]
  deriving Inhabited

structure StepToState where
  idx : Nat := 0
  deriving Inhabited

instance : StepSizeController StepTo where
  State := StepToState
  init ctrl _terms t0 _t1 _y0 _args dt0 _func _errorOrder :=
    match dt0 with
    | some _ => panic! "StepTo requires dt0 := none."
    | none =>
        if ctrl.ts.size >= 2 then
          let t1 := ctrl.ts[1]!
          { dt := t1 - t0, state := { idx := 1 } }
        else
          panic! "StepTo requires ts.size >= 2."
  adapt ctrl state _t0 t1 _y0 _y1 _yError _errorOrder :=
    if state.state.idx + 1 < ctrl.ts.size then
      let nextIdx := state.state.idx + 1
      let nextT := ctrl.ts.getD nextIdx t1
      let dt := nextT - t1
      { accept := true, dt := dt, state := { idx := nextIdx }, result := Result.successful }
    else
      { accept := true, dt := state.dt, state := state.state, result := Result.successful }

structure ClipStepSizeController where
  dt_min : Time := 1.0e-6
  dt_max : Time := 1.0
  deriving Inhabited

instance : StepSizeController ClipStepSizeController where
  State := Unit
  init ctrl _terms _t0 _t1 _y0 _args dt0 _func _errorOrder :=
    match dt0 with
    | none => panic! "ClipStepSizeController requires dt0; pass `dt0 := some ...`."
    | some dt0 =>
        let dt :=
          if dt0 < ctrl.dt_min then ctrl.dt_min
          else if dt0 > ctrl.dt_max then ctrl.dt_max
          else dt0
        { dt := dt, state := () }
  adapt ctrl state _t0 _t1 _y0 _y1 _yError _errorOrder :=
    let dt :=
      if state.dt < ctrl.dt_min then ctrl.dt_min
      else if state.dt > ctrl.dt_max then ctrl.dt_max
      else state.dt
    { accept := true, dt := dt, state := state.state, result := Result.successful }

end DiffEq
end torch
