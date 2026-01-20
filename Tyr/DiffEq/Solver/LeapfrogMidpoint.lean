import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Leapfrog/Midpoint Solver -/

structure LeapfrogMidpoint where
  deriving Inhabited

def LeapfrogMidpoint.solver {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args]
    [DiffEqSpace Y] :
    AbstractSolver Term Y VF Control Args := {
  SolverState := Time × Y
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  order := fun _ => 2
  strongOrder := fun _ => 0.0
  init := fun _ t0 _t1 y0 _args => (t0, y0)
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    let (tm1, ym1) := state
    let control := inst.contr term tm1 t1
    let incr := inst.vf_prod term t0 y0 args control
    let y1 := DiffEqSpace.add ym1 incr
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := (t0, y0)
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver LeapfrogMidpoint := ⟨True.intro⟩

end DiffEq
end torch
