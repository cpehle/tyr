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

private def saveAtTsOutOfRange (t0 t1 : Time) (saveat : SaveAt) : Bool :=
  match saveat.ts with
  | none => false
  | some ts =>
      if ts.size == 0 then
        false
      else
        let tmin := if t0 <= t1 then t0 else t1
        let tmax := if t0 <= t1 then t1 else t0
        ts.foldl (init := false) (fun acc t => acc || t < tmin || t > tmax)

def backsolveAdjointUnsupportedReason {Y Args : Type}
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (t0 t1 : Time)
    (saveat : SaveAt) : Option String :=
  if saveAtTsOutOfRange t0 t1 saveat then
    some "Adjoint solve failed: `saveat.ts` contains values outside [t0, t1]."
  else if solver.termStructure != TermStructure.single then
    some "Adjoint solve failed: only single-term ODE solvers are supported; multi-term structures (for example SDE drift+diffusion) are not yet supported."
  else if saveat.steps then
    some "Adjoint solve failed: backsolve-style adjoints do not support `saveat.steps := true`."
  else if saveat.dense then
    some "Adjoint solve failed: backsolve-style adjoints do not support `saveat.dense := true`."
  else
    none

def directAdjointUnsupportedReason (dt0 : Option Time) : Option String :=
  if dt0.isNone then
    some "Adjoint solve failed: direct/forward mode currently requires `dt0` (constant-step solve)."
  else
    none

private def mkAdjointInternalErrorSolution {Y SolverState ControllerState : Type}
    (t0 t1 : Time)
    (stats : List (String × Nat) := []) :
    Solution Y SolverState ControllerState := {
      t0 := t0
      t1 := t1
      ts := none
      ys := none
      interpolation := none
      stats := stats
      result := Result.internalError
      solverState := none
      controllerState := none
      madeJump := none
      eventMask := none
    }

private def mkAdjointFailure
    {Y SolverState ControllerState Args : Type}
    (t0 t1 : Time)
    (tag : String)
    (msg : String) :
    (Solution Y SolverState ControllerState × Option (AdjointResult Y Args) × Option String) :=
  let sol :=
    mkAdjointInternalErrorSolution (Y := Y) (SolverState := SolverState)
      (ControllerState := ControllerState) t0 t1
      [("adjoint_error", 1), (tag, 1)]
  (sol, none, some msg)

abbrev AdjointSolveWithReport (Y SolverState ControllerState Args : Type) :=
  (Solution Y SolverState ControllerState × Option (AdjointResult Y Args) × Option String)

/-! ## Adjoint Method Modes -/

structure DirectAdjoint where
  /-- Keep parity with the current direct-adjoint implementation contract. -/
  requireDt0 : Bool := true
  deriving Inhabited

structure RecursiveCheckpointAdjoint where
  /-- Number of primal steps per checkpoint chunk (`0` is normalized to `1`). -/
  checkpointEvery : Nat := 64
  /--
  Recompute each checkpoint chunk before backpropagating through it.
  When `false`, this mode reduces to regular full-trajectory backsolve.
  -/
  recomputeSegments : Bool := true
  deriving Inhabited

structure ForwardMode where
  /-- Keep parity with the current forward-mode implementation contract. -/
  requireDt0 : Bool := true
  deriving Inhabited

structure ImplicitAdjoint where
  /--
  Use the current backsolve-based fallback for implicit adjoints.
  If `false`, this mode returns an explicit unsupported message.
  -/
  useBacksolveFallback : Bool := true
  /--
  Optional recursive-checkpoint configuration for implicit adjoints.
  This strategy still requires `useBacksolveFallback := true`.
  -/
  recursiveCheckpoint : Option RecursiveCheckpointAdjoint := none
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
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
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
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
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

def recursiveCheckpointChunkSize (mode : RecursiveCheckpointAdjoint) : Nat :=
  if mode.checkpointEvery == 0 then 1 else mode.checkpointEvery

def recursiveCheckpointAdjointUnsupportedReason
    (mode : RecursiveCheckpointAdjoint) : Option String :=
  let _ := mode
  none

private def recursiveCheckpointGrid
    {Y : Type}
    [Inhabited Y]
    (ts : Array Time)
    (ys : Array Y)
    (chunkSize : Nat) : Option (Array Time × Array Y) :=
  if ts.size == 0 || ys.size == 0 || ts.size != ys.size then
    none
  else
    let stride := if chunkSize == 0 then 1 else chunkSize
    let last := ts.size - 1
    Id.run do
      let mut chkTs : Array Time := #[ts[0]!]
      let mut chkYs : Array Y := #[ys[0]!]
      for i in [:ts.size] do
        if i != 0 then
          if i % stride == 0 || i == last then
            chkTs := chkTs.push ts[i]!
            chkYs := chkYs.push ys[i]!
      return some (chkTs, chkYs)

private def recursiveCheckpointBackprop
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (term : ODETerm Y Args)
    (solver : AbstractSolver (ODETerm Y Args) Y Y Time Args)
    (adjSolver :
      AbstractSolver (ODETerm (AdjointState Y Args) Args)
        (AdjointState Y Args) (AdjointState Y Args) Time Args)
    (checkTs : Array Time)
    (checkYs : Array Y)
    (dt0 : Option Time)
    (args : Args)
    (adjY1 : Y)
    (maxSteps : Nat)
    (controller : Controller) :
    (Option (AdjointResult Y Args) × Option String) :=
  if checkTs.size == 0 || checkYs.size == 0 || checkTs.size != checkYs.size then
    (none, some "Adjoint solve failed: recursive checkpoint grid is empty or malformed.")
  else
    Id.run do
      let mut adjY := adjY1
      let mut adjArgs := DiffEqSpace.scale 0.0 args
      let mut failure : Option String := none
      let numSegs := checkTs.size - 1
      for offset in [:numSegs] do
        if failure.isNone then
          let i := checkTs.size - 1 - offset
          let segT0 := checkTs[i - 1]!
          let segT1 := checkTs[i]!
          let segY0 := checkYs[i - 1]!
          let segSol :=
            diffeqsolve
              (Term := ODETerm Y Args)
              (Y := Y)
              (VF := Y)
              (Control := Time)
              (Args := Args)
              (Controller := Controller)
              term solver segT0 segT1 dt0 segY0 args (saveat := { steps := true })
              (maxSteps := maxSteps)
              (controller := controller)
          if segSol.result != Result.successful then
            failure :=
              some
                "Adjoint solve failed: recursive checkpoint segment recomputation did not finish successfully."
          else
            match backsolveAdjoint term adjSolver segSol args adjY with
            | none =>
                failure :=
                  some
                    "Adjoint solve failed: recursive checkpoint segment backsolve did not produce an adjoint state."
            | some segAdj =>
                adjY := segAdj.adjY0
                adjArgs := DiffEqSpace.add adjArgs segAdj.adjArgs
      match failure with
      | some msg => return (none, some msg)
      | none => return (some { adjY0 := adjY, adjArgs := adjArgs }, none)

def directModeUnsupportedReason
    (mode : DirectAdjoint)
    (dt0 : Option Time) : Option String :=
  if dt0.isNone then
    if mode.requireDt0 then
      directAdjointUnsupportedReason dt0
    else
      none
  else
    none

def forwardModeUnsupportedReason
    (mode : ForwardMode)
    (dt0 : Option Time) : Option String :=
  if dt0.isNone then
    if mode.requireDt0 then
      directAdjointUnsupportedReason dt0
    else
      none
  else
    none

private def inferConstantStepDt0 (t0 t1 : Time) (maxSteps : Nat) : Option Time :=
  let span := t1 - t0
  if span == 0.0 then
    none
  else
    let budget := if maxSteps <= 1 then 1 else maxSteps / 2
    let steps := Nat.min 256 budget
    let dt := span / Float.ofNat steps
    if dt == 0.0 then some span else some dt

private def resolveDirectLikeDt0
    (modeName : String)
    (requireDt0 : Bool)
    (t0 t1 : Time)
    (dt0 : Option Time)
    (maxSteps : Nat) :
    (Option Time × Option String) :=
  match dt0 with
  | some dt => (some dt, none)
  | none =>
      if requireDt0 then
        (none, directAdjointUnsupportedReason none)
      else
        match inferConstantStepDt0 t0 t1 maxSteps with
        | some inferred => (some inferred, none)
        | none =>
            (none,
              some
                s!"Adjoint solve failed: `{modeName}` without `dt0` could not infer a nonzero constant step from `t0`/`t1`.")

def implicitAdjointUnsupportedReason
    (mode : ImplicitAdjoint) : Option String :=
  if mode.useBacksolveFallback then
    none
  else
    match mode.recursiveCheckpoint with
    | some _ =>
        some
          "Adjoint solve failed: `ImplicitAdjoint.recursiveCheckpoint` requires `useBacksolveFallback := true`."
    | none =>
        some "Adjoint solve failed: `ImplicitAdjoint` without backsolve fallback is not implemented yet."

private def diffeqsolveDirectAdjointWithReportCore
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
    (controller : ConstantStepSize := default)
    (unsupportedTag : String)
    (unsupportedReason : Option String) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) Args := by
  match unsupportedReason with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := ConstantStepSize))
          (Args := Args)
          t0 t1 unsupportedTag msg
  | none =>
      let (sol, adj) :=
        diffeqsolveDirectAdjoint
          term solver t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
      exact (sol, adj, none)

private def diffeqsolveBacksolveAdjointWithReportCore
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
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
    (controller : Controller := default)
    (unsupportedTag : String)
    (unsupportedReason : Option String) :
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  match unsupportedReason with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 unsupportedTag msg
  | none =>
      match backsolveAdjointUnsupportedReason solver t0 t1 saveat with
      | some msg =>
          exact
            mkAdjointFailure
              (Y := Y)
              (SolverState := solver.SolverState)
              (ControllerState := StepSizeController.State (C := Controller))
              (Args := Args)
              t0 t1 "unsupported_backsolve_adjoint" msg
      | none =>
          let (sol, adj) :=
            diffeqsolveBacksolveAdjoint
              (Controller := Controller)
              term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
          exact (sol, adj, none)

def diffeqsolveDirectAdjointMode
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointFnBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (mode : DirectAdjoint := {})
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
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) Args :=
  let (dt0Resolved, dt0UnsupportedReason) :=
    resolveDirectLikeDt0 "DirectAdjoint" mode.requireDt0 t0 t1 dt0 maxSteps
  let unsupportedReason :=
    match dt0UnsupportedReason with
    | some msg => some msg
    | none => directModeUnsupportedReason mode dt0Resolved
  diffeqsolveDirectAdjointWithReportCore
    term solver t0 t1 dt0Resolved y0 args adjY1 saveat maxSteps controller
    "unsupported_direct_adjoint_mode" unsupportedReason

def diffeqsolveForwardMode
    {Y Args : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqElem Y]
    [AdjointFnBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    (mode : ForwardMode := {})
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
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := ConstantStepSize)) Args :=
  let (dt0Resolved, dt0UnsupportedReason) :=
    resolveDirectLikeDt0 "ForwardMode" mode.requireDt0 t0 t1 dt0 maxSteps
  let unsupportedReason :=
    match dt0UnsupportedReason with
    | some msg => some msg
    | none => forwardModeUnsupportedReason mode dt0Resolved
  diffeqsolveDirectAdjointWithReportCore
    term solver t0 t1 dt0Resolved y0 args adjY1 saveat maxSteps controller
    "unsupported_forward_mode" unsupportedReason

def diffeqsolveRecursiveCheckpointAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (mode : RecursiveCheckpointAdjoint := {})
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
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  match recursiveCheckpointAdjointUnsupportedReason mode with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 "unsupported_recursive_checkpoint_adjoint" msg
  | none =>
      match backsolveAdjointUnsupportedReason solver t0 t1 saveat with
      | some msg =>
          exact
            mkAdjointFailure
              (Y := Y)
              (SolverState := solver.SolverState)
              (ControllerState := StepSizeController.State (C := Controller))
              (Args := Args)
              t0 t1 "unsupported_recursive_checkpoint_backsolve_contract" msg
      | none =>
          if !mode.recomputeSegments then
            let (sol, adjRes) :=
              diffeqsolveBacksolveAdjoint
                (Controller := Controller)
                term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
            exact (sol, adjRes, none)
          else
            let solPrimal :=
              diffeqsolve
                (Term := ODETerm Y Args)
                (Y := Y)
                (VF := Y)
                (Control := Time)
                (Args := Args)
                (Controller := Controller)
                term solver t0 t1 dt0 y0 args (saveat := saveat) (maxSteps := maxSteps)
                (controller := controller)
            if solPrimal.result != Result.successful then
              exact
                (solPrimal, none,
                  some
                    "Adjoint solve failed: primal solve did not finish successfully before recursive checkpoint backpropagation.")
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
                  term solver t0 t1 dt0 y0 args (saveat := saveatInternal)
                  (maxSteps := maxSteps)
                  (controller := controller)
              if solSteps.result != Result.successful then
                exact
                  (solPrimal, none,
                    some
                      "Adjoint solve failed: could not build recursive checkpoint trajectory because the internal step solve did not finish successfully.")
              else
                match solSteps.ts, solSteps.ys with
                | some ts, some ys =>
                    match recursiveCheckpointGrid ts ys (recursiveCheckpointChunkSize mode) with
                    | none =>
                        exact
                          (solPrimal, none,
                            some
                              "Adjoint solve failed: recursive checkpoint grid construction failed.")
                    | some (chkTs, chkYs) =>
                        let (adjRes, backpropFailure) :=
                          recursiveCheckpointBackprop
                            (Controller := Controller)
                            term solver adjoint.adjSolver chkTs chkYs dt0 args adjY1 maxSteps
                            controller
                        match backpropFailure with
                        | some msg => exact (solPrimal, none, some msg)
                        | none => exact (solPrimal, adjRes, none)
                | _, _ =>
                    exact
                      (solPrimal, none,
                        some
                          "Adjoint solve failed: internal recursive checkpoint step solve did not return `(ts, ys)`.")

def diffeqsolveImplicitAdjoint
    {Y Args Controller : Type}
    [DiffEqSpace Y] [DiffEqSpace Args]
    [DiffEqSeminorm Y] [DiffEqSeminorm Args]
    [DiffEqElem Y] [DiffEqElem Args]
    [AdjointBackend Y Args]
    [Inhabited Y] [Inhabited Args]
    [StepSizeController Controller] [StepSizeControllerValidation Controller]
    [Inhabited Controller]
    (mode : ImplicitAdjoint := {})
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
    AdjointSolveWithReport Y solver.SolverState
      (StepSizeController.State (C := Controller)) Args := by
  match implicitAdjointUnsupportedReason mode with
  | some msg =>
      exact
        mkAdjointFailure
          (Y := Y)
          (SolverState := solver.SolverState)
          (ControllerState := StepSizeController.State (C := Controller))
          (Args := Args)
          t0 t1 "unsupported_implicit_adjoint" msg
  | none =>
      match mode.recursiveCheckpoint with
      | some recursiveMode =>
          exact
            diffeqsolveRecursiveCheckpointAdjoint
              (Controller := Controller)
              recursiveMode term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps
              controller
      | none =>
          exact
            diffeqsolveBacksolveAdjointWithReportCore
              (Controller := Controller)
              term solver adjoint t0 t1 dt0 y0 args adjY1 saveat maxSteps controller
              "unsupported_implicit_adjoint"
              none

end DiffEq
end torch
