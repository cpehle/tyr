import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Midpoint Solver (Explicit RK2) -/

structure Midpoint where
  deriving Inhabited

def Midpoint.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  order := fun _ => 2
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    let dt := inst.contr term t0 t1
    let k1 := inst.vf_prod term t0 y0 args dt
    let yMid := DiffEqSpace.add y0 (DiffEqSpace.scale 0.5 k1)
    let tMid := t0 + dt / 2.0
    let k2 := inst.vf_prod term tMid yMid args dt
    let y1 := DiffEqSpace.add y0 k2
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver Midpoint := ⟨True.intro⟩
instance : StratonovichSolver Midpoint := ⟨True.intro⟩

end DiffEq
end torch
