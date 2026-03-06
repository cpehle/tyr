import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Heun Solver (Explicit RK2) -/

local instance (priority := 5) [DiffEqSpace α] : HAdd α α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
local instance (priority := 5) [DiffEqSpace α] : HSub α α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
local instance (priority := 5) [DiffEqSpace α] : HMul Scalar α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure Heun where
  deriving Inhabited

def Heun.solver {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Control Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  order := fun _ => 2
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    let dt := inst.contr term t0 t1
    let k1 := inst.vf_prod term t0 y0 args dt
    let yPred := y0 + k1
    let k2 := inst.vf_prod term t1 yPred args dt
    let y1 :=
      y0 + (0.5 * (k1 + k2))
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver Heun := ⟨True.intro⟩
instance : StratonovichSolver Heun := ⟨True.intro⟩

end DiffEq
end torch
