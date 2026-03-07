import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Brownian

namespace torch
namespace DiffEq

/-! ## SlowRK Solver (commutative-noise Stratonovich SRK)

Faithful staged implementation based on Diffrax's `slowrk.py` tableau
structure, with explicit drift/diffusion stage separation.
-/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure SlowRK where
  deriving Inhabited

def SlowRK.solver {Drift Diffusion Y VFg Control Args : Type}
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
    let zero := w - w
    let ctrlW := buildInst.build dtControl w zero
    let ctrlH := buildInst.build dtControl h zero
    let tHalf := t0 + 0.5 * dt
    let tThreeQuarter := t0 + 0.75 * dt

    -- Stage 0: drift only.
    let h_kf0 := driftInst.vf_prod drift t0 y0 args dt

    -- Stages 1-4: diffusion-only evaluations.
    let z1 := y0 + 0.5 * h_kf0
    let w_kg1 := diffInst.vf_prod diffusion tHalf z1 args ctrlW

    let z2 := y0 + (0.5 * h_kf0 + 0.5 * w_kg1)
    let w_kg2 := diffInst.vf_prod diffusion tHalf z2 args ctrlW

    let z3 := y0 + (0.5 * h_kf0 + 0.5 * w_kg2)
    let w_kg3 := diffInst.vf_prod diffusion tHalf z3 args ctrlW
    let h_kg3 := diffInst.vf_prod diffusion tHalf z3 args ctrlH

    let z4 := y0 + (0.5 * h_kf0 + w_kg3)
    let w_kg4 := diffInst.vf_prod diffusion tHalf z4 args ctrlW

    -- Stage 5: drift-only evaluation.
    let z5 := y0 + (0.75 * h_kf0 + (0.75 * w_kg3 + 1.5 * h_kg3))
    let h_kf5 := driftInst.vf_prod drift tThreeQuarter z5 args dt

    -- Stage 6: diffusion-only evaluation for the H correction.
    let z6 := y0 + (h_kf0 + 0.5 * w_kg2)
    let h_kg6 := diffInst.vf_prod diffusion t1 z6 args ctrlH

    let driftResult := (1.0 / 3.0) * h_kf0 + (2.0 / 3.0) * h_kf5
    let wResult := (1.0 / 6.0) * w_kg1 + (1.0 / 3.0) * w_kg2 +
      (1.0 / 3.0) * w_kg3 + (1.0 / 6.0) * w_kg4
    let hResult := 2.0 * h_kg3 - 2.0 * h_kg6
    let y1 := y0 + (driftResult + (wResult + hResult))
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

instance : ExplicitSolver SlowRK := ⟨True.intro⟩
instance : StratonovichSolver SlowRK := ⟨True.intro⟩

end DiffEq
end torch
