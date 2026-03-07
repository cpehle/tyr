import Tyr.DiffEq.Solver.Milstein

namespace torch
namespace DiffEq

/-! ## Stratonovich Milstein SDE Solver -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure StratonovichMilstein where
  deriving Inhabited

def StratonovichMilstein.solver {Drift Diffusion Y VFd VFg Control Args : Type}
    [TermLike Drift Y VFd Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [MilsteinJacobianControlLike Diffusion Y VFg Control Args]
    [MilsteinControl Control]
    [StratonovichMilsteinCorrectionLike Diffusion Y VFg Control Args]
    [DiffEqSpace Y] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (VFd × VFg) (Time × Control) Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.multi
  order := fun _ => 1
  strongOrder := fun _ => 1.0
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance : TermLike Drift Y VFd Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    let correctionInst := (inferInstance : StratonovichMilsteinCorrectionLike Diffusion Y VFg Control Args)
    let dt := driftInst.contr drift t0 t1
    let dControl := diffInst.contr diffusion t0 t1
    let f0 := driftInst.vf_prod drift t0 y0 args dt
    let g0 := diffInst.vf_prod diffusion t0 y0 args dControl
    let corr := correctionInst.correction diffusion t0 y0 args dControl
    let y1 := y0 + (f0 + (g0 + corr))
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun terms t y args =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance : TermLike Drift Y VFd Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    (driftInst.vf drift t y args, diffInst.vf diffusion t y args)
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver StratonovichMilstein := ⟨True.intro⟩
instance : StratonovichSolver StratonovichMilstein := ⟨True.intro⟩

end DiffEq
end torch
