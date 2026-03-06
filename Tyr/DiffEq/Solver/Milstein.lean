import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Brownian

namespace torch
namespace DiffEq

/-! ## Milstein SDE Solver (Ito) -/

structure Milstein where
  deriving Inhabited

class MilsteinControl (Control : Type) where
  quadraticVariation : Control → Float

instance : MilsteinControl Float where
  quadraticVariation x := x * x

instance [BrownianIncrementLike Control Float] : MilsteinControl Control where
  quadraticVariation control :=
    let w := BrownianIncrementLike.W control
    w * w

instance (priority := 50) [DiffEqSeminorm Control] : MilsteinControl Control where
  quadraticVariation control :=
    let n := DiffEqSeminorm.rms control
    n * n

class MilsteinJacobianLike (τ : Type) (Y VF Control Args : Type) where
  jacobianProd : τ → Time → Y → Args → Y

instance [DiffusionTermLike τ Y VF Control Args] :
    MilsteinJacobianLike τ Y VF Control Args where
  jacobianProd term := (inferInstance : DiffusionTermLike τ Y VF Control Args).jacobian_prod term

structure FiniteDiffJacobianDiffusion (Diffusion : Type) where
  term : Diffusion
  epsilon : Float := 1.0e-4

def withFiniteDiffJacobian (term : Diffusion) (epsilon : Float := 1.0e-4) :
    FiniteDiffJacobianDiffusion Diffusion :=
  { term := term, epsilon := epsilon }

instance {Diffusion Y VF Control Args : Type} [TermLike Diffusion Y VF Control Args] :
    TermLike (FiniteDiffJacobianDiffusion Diffusion) Y VF Control Args where
  vf wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).vf wrapped.term
  contr wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).contr wrapped.term
  prod wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).prod wrapped.term
  vf_prod wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).vf_prod wrapped.term
  is_vf_expensive wrapped :=
    (inferInstance : TermLike Diffusion Y VF Control Args).is_vf_expensive wrapped.term

instance {Diffusion Y Control Args : Type} [TermLike Diffusion Y Y Control Args] [DiffEqSpace Y] :
    MilsteinJacobianLike (FiniteDiffJacobianDiffusion Diffusion) Y Y Control Args where
  jacobianProd wrapped t y args :=
    let inst := (inferInstance : TermLike Diffusion Y Y Control Args)
    let eps := if Float.abs wrapped.epsilon <= 1.0e-12 then 1.0e-4 else wrapped.epsilon
    let g0 := inst.vf wrapped.term t y args
    let yPert := DiffEqSpace.add y (DiffEqSpace.scale eps g0)
    let g1 := inst.vf wrapped.term t yPert args
    DiffEqSpace.scale (1.0 / eps) (DiffEqSpace.sub g1 g0)

def Milstein.solver {Drift Diffusion Y VFd VFg Control Args : Type}
    [TermLike Drift Y VFd Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [MilsteinJacobianLike Diffusion Y VFg Control Args]
    [MilsteinControl Control]
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
    let diffDerivInst := (inferInstance : MilsteinJacobianLike Diffusion Y VFg Control Args)
    let dt := driftInst.contr drift t0 t1
    let dControl := diffInst.contr diffusion t0 t1
    let f0 := driftInst.vf_prod drift t0 y0 args dt
    let g0 := diffInst.vf_prod diffusion t0 y0 args dControl
    let gg0 := diffDerivInst.jacobianProd diffusion t0 y0 args
    let qv := MilsteinControl.quadraticVariation dControl
    let corr := DiffEqSpace.scale (0.5 * (qv - dt)) gg0
    let y1 := DiffEqSpace.add y0 (DiffEqSpace.add f0 (DiffEqSpace.add g0 corr))
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

instance : ExplicitSolver Milstein := ⟨True.intro⟩
instance : ItoSolver Milstein := ⟨True.intro⟩

end DiffEq
end torch
