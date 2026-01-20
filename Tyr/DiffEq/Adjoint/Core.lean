import Tyr.DiffEq.Integrate

namespace torch
namespace DiffEq

/-! ## Adjoint Support (Core)

Backend-agnostic adjoint utilities for ODEs.
Provide an `AdjointBackend` instance to supply VJPs.

Backsolve adjoints currently mirror Diffrax constraints:
- Only ODE terms and single-term solvers.
- No `SaveAt(steps := true)` or `SaveAt(dense := true)`.
- No events; SDE adjoints are not yet supported (Stratonovich-only when added).

Direct adjoints are currently limited to constant step size ODE solves.
-/

class AdjointBackend (Y Args : Type) where
  vjp : (Time → Y → Args → Y) → Time → Y → Args → Y → (Y × Y × Args)

class AdjointFnBackend (Y Args : Type) where
  vjpFn : (Y → Args → Y) → Y → Args → Y → (Y × Args)

structure AdjointState (Y Args : Type) where
  y : Y
  adjY : Y
  adjArgs : Args

instance [Inhabited Y] [Inhabited Args] : Inhabited (AdjointState Y Args) :=
  ⟨{ y := default, adjY := default, adjArgs := default }⟩

instance [DiffEqSpace Y] [DiffEqSpace Args] : DiffEqSpace (AdjointState Y Args) where
  add a b := {
    y := DiffEqSpace.add a.y b.y
    adjY := DiffEqSpace.add a.adjY b.adjY
    adjArgs := DiffEqSpace.add a.adjArgs b.adjArgs
  }
  sub a b := {
    y := DiffEqSpace.sub a.y b.y
    adjY := DiffEqSpace.sub a.adjY b.adjY
    adjArgs := DiffEqSpace.sub a.adjArgs b.adjArgs
  }
  scale s a := {
    y := DiffEqSpace.scale s a.y
    adjY := DiffEqSpace.scale s a.adjY
    adjArgs := DiffEqSpace.scale s a.adjArgs
  }

instance [DiffEqElem Y] [DiffEqElem Args] : DiffEqElem (AdjointState Y Args) where
  abs a := {
    y := DiffEqElem.abs a.y
    adjY := DiffEqElem.abs a.adjY
    adjArgs := DiffEqElem.abs a.adjArgs
  }
  max a b := {
    y := DiffEqElem.max a.y b.y
    adjY := DiffEqElem.max a.adjY b.adjY
    adjArgs := DiffEqElem.max a.adjArgs b.adjArgs
  }
  addScalar s a := {
    y := DiffEqElem.addScalar s a.y
    adjY := DiffEqElem.addScalar s a.adjY
    adjArgs := DiffEqElem.addScalar s a.adjArgs
  }
  div a b := {
    y := DiffEqElem.div a.y b.y
    adjY := DiffEqElem.div a.adjY b.adjY
    adjArgs := DiffEqElem.div a.adjArgs b.adjArgs
  }

instance [DiffEqSeminorm Y] [DiffEqSeminorm Args] : DiffEqSeminorm (AdjointState Y Args) where
  rms a :=
    let r1 := DiffEqSeminorm.rms a.y
    let r2 := DiffEqSeminorm.rms a.adjY
    let r3 := DiffEqSeminorm.rms a.adjArgs
    max r1 (max r2 r3)

structure AdjointResult (Y Args : Type) where
  adjY0 : Y
  adjArgs : Args

def backsolveAdjointSupported {Y Args : Type}
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (saveat : SaveAt) : Bool :=
  solver.termStructure == TermStructure.single && !saveat.steps && !saveat.dense

/-! ## Adjoint Method Placeholders (Diffrax parity) -/

structure DirectAdjoint where
  deriving Inhabited

structure RecursiveCheckpointAdjoint where
  deriving Inhabited

structure ForwardMode where
  deriving Inhabited

structure ImplicitAdjoint where
  deriving Inhabited

/-! ## Backsolve Adjoint Wrapper -/

structure BacksolveAdjoint (Y Args : Type) where
  adjSolver :
    AbstractSolver (ODETerm (AdjointState Y Args) Args)
      (AdjointState Y Args) (AdjointState Y Args) Time Args

def adjointTerm
    [DiffEqSpace Y] [DiffEqSpace Args]
    [AdjointBackend Y Args]
    (term : ODETerm Y Args) : ODETerm (AdjointState Y Args) Args :=
  { vectorField := fun t state args =>
      let (f, vjpY, vjpArgs) := (AdjointBackend.vjp (Y := Y) (Args := Args))
        term.vectorField t state.y args state.adjY
      {
        y := f
        adjY := DiffEqSpace.scale (-1.0) vjpY
        adjArgs := DiffEqSpace.scale (-1.0) vjpArgs
      } }

def backsolveAdjoint
    {Y Args SolverState ControllerState : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (term : ODETerm Y Args)
    (adjSolver :
      AbstractSolver (ODETerm (AdjointState Y Args) Args)
        (AdjointState Y Args) (AdjointState Y Args) Time Args)
    (sol : Solution Y SolverState ControllerState)
    (args : Args)
    (adjY1 : Y) :
    Option (AdjointResult Y Args) :=
  if sol.result != Result.successful then
    none
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          none
        else if ts.size != ys.size then
          none
        else
          let adjTerm := adjointTerm term
          Id.run do
            let numSegs := ts.size - 1
            let mut adjY := adjY1
            let mut adjArgs := DiffEqSpace.scale 0.0 args
            let mut ok := true
            for offset in [:numSegs] do
              if ok then
                let i := ts.size - 1 - offset
                let t1 := ts[i]!
                let t0 := ts[i - 1]!
                let y1 := ys[i]!
                let dtAbs := Float.abs (t1 - t0)
                if dtAbs != 0.0 then
                  let state0 : AdjointState Y Args :=
                    { y := y1, adjY := adjY, adjArgs := adjArgs }
                  let adjSol :=
                    diffeqsolve
                      (Term := ODETerm (AdjointState Y Args) Args)
                      (Y := AdjointState Y Args)
                      (VF := AdjointState Y Args)
                      (Control := Time)
                      (Args := Args)
                      (Controller := ConstantStepSize)
                      adjTerm adjSolver t1 t0 (some dtAbs) state0 args
                      (saveat := { t1 := true }) (maxSteps := 1)
                  match adjSol.ys with
                  | some ysAdj =>
                      if ysAdj.size == 0 then
                        ok := false
                      else
                        let state1 := ysAdj[ysAdj.size - 1]!
                        adjY := state1.adjY
                        adjArgs := state1.adjArgs
                  | none => ok := false
            if ok then
              return some { adjY0 := adjY, adjArgs := adjArgs }
            else
              return none
    | _, _ => none

def diffeqsolveAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjSolver :
      AbstractSolver (ODETerm (AdjointState Y Args) Args)
        (AdjointState Y Args) (AdjointState Y Args) Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := Controller)) ×
      Option (AdjointResult Y Args)) := by
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
  if outOfRange || !backsolveAdjointSupported solver saveat then
    let sol : Solution Y solver.SolverState (StepSizeController.State (C := Controller)) := {
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
    exact (sol, none)
  else
    let saveatInternal := { saveat with steps := true, ts := none, dense := false }
    let solSteps :=
      diffeqsolve
        (Term := ODETerm Y Args)
        (Y := Y)
        (VF := Y)
        (Control := Time)
        (Args := Args)
        (Controller := Controller)
        term solver t0 t1 dt0 y0 args (saveat := saveatInternal) (maxSteps := maxSteps)
        (controller := controller)
    let adj := backsolveAdjoint term adjSolver solSteps args adjY1
    let sol :=
      match solSteps.ts, solSteps.ys with
      | some denseTs, some denseYs =>
          if denseTs.size == 0 || denseYs.size == 0 then
            { solSteps with ts := none, ys := none, interpolation := none }
          else
            let tf := solSteps.t1
            let yf := denseYs[denseYs.size - 1]!
            let denseInterp :=
              if saveat.dense || saveatTs.isSome then
                some (LinearInterpolation.toDense { ts := denseTs, ys := denseYs })
              else
                none
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
                    let ysOut :=
                      match denseInterp with
                      | none => #[]
                      | some interp => tsOut.map (fun t => interp.evaluate t none true)
                    (some tsOut, some ysOut)
                | none =>
                    if saveat.steps then
                      (some denseTs, some denseYs)
                    else if saveat.t0 || saveat.t1 then
                      let tsOut :=
                        if saveat.t0 && saveat.t1 then #[t0, tf]
                        else if saveat.t0 then #[t0] else #[tf]
                      let ysOut :=
                        if saveat.t0 && saveat.t1 then #[y0, yf]
                        else if saveat.t0 then #[y0] else #[yf]
                      (some tsOut, some ysOut)
                    else
                      (none, none)
            let (tsOut, ysOut) := computeOutputs tf yf denseTs denseYs denseInterp
            let interpOut := if saveat.dense || saveatTs.isSome then denseInterp else none
            { solSteps with ts := tsOut, ys := ysOut, interpolation := interpOut }
      | _, _ => solSteps
    exact (sol, adj)

def diffeqsolveDirectAdjoint
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointFnBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : ConstantStepSize := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := ConstantStepSize)) ×
      Option (AdjointResult Y Args)) := by
  if dt0.isNone then
    let sol : Solution Y solver.SolverState (StepSizeController.State (C := ConstantStepSize)) := {
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
    exact (sol, none)
  else
    let sol :=
      diffeqsolve
        (Term := ODETerm Y Args)
        (Y := Y)
        (VF := Y)
        (Control := Time)
        (Args := Args)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
        (controller := controller)
    if sol.result != Result.successful then
      exact (sol, none)
    else
      let solveFn := fun (y : Y) (args : Args) =>
        let sol1 :=
          diffeqsolve
            (Term := ODETerm Y Args)
            (Y := Y)
            (VF := Y)
            (Control := Time)
            (Args := Args)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 y args (saveat := { t1 := true }) (maxSteps := maxSteps)
            (controller := controller)
        match sol1.ys with
        | some ys =>
            if ys.size == 0 then y else ys[ys.size - 1]!
        | none => y
      let (adjY0, adjArgs) :=
        (AdjointFnBackend.vjpFn (Y := Y) (Args := Args)) solveFn y0 args adjY1
      exact (sol, some { adjY0 := adjY0, adjArgs := adjArgs })

def diffeqsolveBacksolveAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjoint : BacksolveAdjoint Y Args)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (y0 : Y)
    (args : Args)
    (adjY1 : Y)
    (saveat : SaveAt := {})
    (maxSteps : Nat := 4096)
    (controller : Controller := default) :
    (Solution Y solver.SolverState (StepSizeController.State (C := Controller)) ×
      Option (AdjointResult Y Args)) :=
  diffeqsolveAdjoint term solver adjoint.adjSolver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller

end DiffEq
end torch
