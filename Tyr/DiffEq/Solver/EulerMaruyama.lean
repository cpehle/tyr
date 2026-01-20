import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Euler-Maruyama SDE Solver -/

structure EulerMaruyama where
  deriving Inhabited

def EulerMaruyama.solver {Drift Diffusion Y VFd VFg Control Args : Type}
    [TermLike Drift Y VFd Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [DiffEqSpace Y] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (VFd × VFg) (Time × Control) Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.multi
  order := fun _ => 1
  strongOrder := fun _ => 0.5
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance : TermLike Drift Y VFd Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    let dt := driftInst.contr drift t0 t1
    let dW := diffInst.contr diffusion t0 t1
    let f0 := driftInst.vf_prod drift t0 y0 args dt
    let g0 := diffInst.vf_prod diffusion t0 y0 args dW
    let y1 := DiffEqSpace.add y0 (DiffEqSpace.add f0 g0)
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

instance : ExplicitSolver EulerMaruyama := ⟨True.intro⟩
instance : ItoSolver EulerMaruyama := ⟨True.intro⟩

end DiffEq
end torch
