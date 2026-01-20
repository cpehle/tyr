import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Brownian

namespace torch
namespace DiffEq

/-! ## SRA1 Stochastic Runge-Kutta Solver (additive noise, Stratonovich) -/

structure SRA1 where
  deriving Inhabited

def SRA1.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.multi
  order := fun _ => 2
  strongOrder := fun _ => 1.5
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance : TermLike Drift Y Y Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    let ctrlInst := (inferInstance : SpaceTimeLevyAreaLike Control Float)
    let buildInst := (inferInstance : SpaceTimeLevyAreaBuild Control Float)
    let dt := driftInst.contr drift t0 t1
    let control := diffInst.contr diffusion t0 t1
    let w := ctrlInst.W control
    let h := ctrlInst.H control
    let dtControl := ctrlInst.dt control
    let zero := DiffEqSpace.sub w w
    let ctrlW := buildInst.build dtControl w zero
    let ctrlH := buildInst.build dtControl h zero
    let g0 := diffInst.vf diffusion t0 y0 args
    let g1 := diffInst.vf diffusion t1 y0 args
    let g_delta := DiffEqSpace.scale 0.5 (DiffEqSpace.sub g1 g0)
    let w_kg := diffInst.prod diffusion g0 ctrlW
    let h_kg := diffInst.prod diffusion g0 ctrlH
    let h_kf0 := driftInst.vf_prod drift t0 y0 args dt
    let drift1 := DiffEqSpace.scale 0.75 h_kf0
    let diff1 := DiffEqSpace.add (DiffEqSpace.scale 0.75 w_kg) (DiffEqSpace.scale 1.5 h_kg)
    let z1 := DiffEqSpace.add y0 (DiffEqSpace.add drift1 diff1)
    let h_kf1 := driftInst.vf_prod drift (t0 + 0.75 * dt) z1 args dt
    let drift_result :=
      DiffEqSpace.add (DiffEqSpace.scale (1.0 / 3.0) h_kf0) (DiffEqSpace.scale (2.0 / 3.0) h_kf1)
    let ctrlTime := buildInst.build dtControl (w - 2.0 * h) zero
    let time_var_term := diffInst.prod diffusion g_delta ctrlTime
    let diffusion_result := DiffEqSpace.add w_kg time_var_term
    let y1 := DiffEqSpace.add y0 (DiffEqSpace.add drift_result diffusion_result)
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
    let driftInst := (inferInstance : TermLike Drift Y Y Time Args)
    let diffInst := (inferInstance : TermLike Diffusion Y VFg Control Args)
    (driftInst.vf drift t y args, diffInst.vf diffusion t y args)
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver SRA1 := ⟨True.intro⟩
instance : StratonovichSolver SRA1 := ⟨True.intro⟩

end DiffEq
end torch
