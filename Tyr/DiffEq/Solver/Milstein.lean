import Tyr.DiffEq.Solver.Base
import Tyr.DiffEq.Brownian

namespace torch
namespace DiffEq

/-! ## Milstein SDE Solver (Ito) -/

local instance (priority := 5) [DiffEqSpace α] : HAdd α α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
local instance (priority := 5) [DiffEqSpace α] : HSub α α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
local instance (priority := 5) [DiffEqSpace α] : HMul Scalar α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

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

/-- Preferred Milstein Jacobian-product interface, parameterized by the current control
increment. This is convenient for autodiff/JVP plumbing and non-scalar controls. -/
class MilsteinJacobianControlLike (τ : Type) (Y VF Control Args : Type) where
  jacobianProd : τ → Time → Y → Args → Control → Y

instance (priority := 60) [MilsteinJacobianLike τ Y VF Control Args] :
    MilsteinJacobianControlLike τ Y VF Control Args where
  jacobianProd term t y args _control :=
    (inferInstance : MilsteinJacobianLike τ Y VF Control Args).jacobianProd term t y args

instance [DiffusionTermLike τ Y VF Control Args] :
    MilsteinJacobianLike τ Y VF Control Args where
  jacobianProd term := (inferInstance : DiffusionTermLike τ Y VF Control Args).jacobian_prod term

/-- Attach an explicit Milstein Jacobian-product callback to any diffusion term.
This is intended for ergonomic injection of user-provided autodiff/JVP rules. -/
structure JacobianProdDiffusion (Diffusion Y Control Args : Type) where
  term : Diffusion
  jacobianProd : Time → Y → Args → Control → Y

def withJacobianProd {Diffusion Y Control Args : Type}
    (term : Diffusion) (jacobianProd : Time → Y → Args → Control → Y) :
    JacobianProdDiffusion Diffusion Y Control Args :=
  { term := term, jacobianProd := jacobianProd }

def withAutodiffJacobianProd {Diffusion Y Control Args : Type}
    (term : Diffusion) (jacobianProd : Time → Y → Args → Control → Y) :
    JacobianProdDiffusion Diffusion Y Control Args :=
  withJacobianProd term jacobianProd

instance {Diffusion Y VF Control Args : Type} [TermLike Diffusion Y VF Control Args] :
    TermLike (JacobianProdDiffusion Diffusion Y Control Args) Y VF Control Args where
  vf wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).vf wrapped.term
  contr wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).contr wrapped.term
  prod wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).prod wrapped.term
  vf_prod wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).vf_prod wrapped.term
  is_vf_expensive wrapped :=
    (inferInstance : TermLike Diffusion Y VF Control Args).is_vf_expensive wrapped.term

instance {Diffusion Y VF Control Args : Type} :
    MilsteinJacobianControlLike (JacobianProdDiffusion Diffusion Y Control Args) Y VF Control Args where
  jacobianProd wrapped t y args control := wrapped.jacobianProd t y args control

structure FiniteDiffJacobianDiffusion (Diffusion : Type) where
  term : Diffusion
  epsilon : Float := 1.0e-4

def withFiniteDiffJacobian (term : Diffusion) (epsilon : Float := 1.0e-4) :
    FiniteDiffJacobianDiffusion Diffusion :=
  { term := term, epsilon := epsilon }

@[inline] private def sanitizeFiniteDiffEpsilon (epsilon : Float) : Float :=
  if Float.abs epsilon <= 1.0e-12 then 1.0e-4 else epsilon

@[inline] private def sanitizeQuadraticVariationFloor (qvFloor : Float) : Float :=
  let floor := Float.abs qvFloor
  if floor <= 1.0e-18 then 1.0e-12 else floor

@[inline] private def safeQuadraticVariationDenom (qv : Float) (qvFloor : Float := 1.0e-12) : Float :=
  let floor := sanitizeQuadraticVariationFloor qvFloor
  if Float.abs qv <= floor then
    if qv < 0.0 then -floor else floor
  else
    qv

private def finiteDiffJacobianFromVF {Diffusion Y Control Args : Type}
    [TermLike Diffusion Y Y Control Args] [DiffEqSpace Y]
    (term : Diffusion) (t : Time) (y : Y) (args : Args) (epsilon : Float := 1.0e-4) : Y :=
  let inst := (inferInstance : TermLike Diffusion Y Y Control Args)
  let eps := sanitizeFiniteDiffEpsilon epsilon
  let g0 := inst.vf term t y args
  let yPert := y + eps * g0
  let g1 := inst.vf term t yPert args
  (1.0 / eps) * (g1 - g0)

private def finiteDiffJacobianFromVfProd {Diffusion Y VF Control Args : Type}
    [TermLike Diffusion Y VF Control Args] [DiffEqSpace Y] [MilsteinControl Control]
    (term : Diffusion) (t : Time) (y : Y) (args : Args) (control : Control)
    (epsilon : Float := 1.0e-4) (qvFloor : Float := 1.0e-12) : Y :=
  let inst := (inferInstance : TermLike Diffusion Y VF Control Args)
  let eps := sanitizeFiniteDiffEpsilon epsilon
  let g0 := inst.vf_prod term t y args control
  let yPert := y + eps * g0
  let g1 := inst.vf_prod term t yPert args control
  let jvpScaled := (1.0 / eps) * (g1 - g0)
  let qv := MilsteinControl.quadraticVariation control
  let qvSafe := safeQuadraticVariationDenom qv qvFloor
  (1.0 / qvSafe) * jvpScaled

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
    finiteDiffJacobianFromVF
      (Diffusion := Diffusion) (Y := Y) (Control := Control) (Args := Args)
      wrapped.term t y args wrapped.epsilon

structure FiniteDiffProdJacobianDiffusion (Diffusion : Type) where
  term : Diffusion
  epsilon : Float := 1.0e-4
  qvFloor : Float := 1.0e-12

def withFiniteDiffJacobianProd (term : Diffusion) (epsilon : Float := 1.0e-4)
    (qvFloor : Float := 1.0e-12) :
    FiniteDiffProdJacobianDiffusion Diffusion :=
  { term := term, epsilon := epsilon, qvFloor := qvFloor }

instance {Diffusion Y VF Control Args : Type} [TermLike Diffusion Y VF Control Args] :
    TermLike (FiniteDiffProdJacobianDiffusion Diffusion) Y VF Control Args where
  vf wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).vf wrapped.term
  contr wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).contr wrapped.term
  prod wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).prod wrapped.term
  vf_prod wrapped := (inferInstance : TermLike Diffusion Y VF Control Args).vf_prod wrapped.term
  is_vf_expensive wrapped :=
    (inferInstance : TermLike Diffusion Y VF Control Args).is_vf_expensive wrapped.term

instance {Diffusion Y VF Control Args : Type}
    [TermLike Diffusion Y VF Control Args] [DiffEqSpace Y] [MilsteinControl Control] :
    MilsteinJacobianControlLike (FiniteDiffProdJacobianDiffusion Diffusion) Y VF Control Args where
  jacobianProd wrapped t y args control :=
    finiteDiffJacobianFromVfProd
      (Diffusion := Diffusion) (Y := Y) (VF := VF) (Control := Control) (Args := Args)
      wrapped.term t y args control wrapped.epsilon wrapped.qvFloor

instance (priority := 20) {Diffusion Y Control Args : Type}
    [TermLike Diffusion Y Y Control Args] [DiffEqSpace Y] [MilsteinControl Control] :
    MilsteinJacobianControlLike Diffusion Y Y Control Args where
  jacobianProd term t y args control :=
    let qv := MilsteinControl.quadraticVariation control
    if Float.abs qv <= 1.0e-12 then
      finiteDiffJacobianFromVF
        (Diffusion := Diffusion) (Y := Y) (Control := Control) (Args := Args)
        term t y args
    else
      finiteDiffJacobianFromVfProd
        (Diffusion := Diffusion) (Y := Y) (VF := Y) (Control := Control) (Args := Args)
        term t y args control

instance (priority := 5) {Diffusion Y VF Control Args : Type}
    [TermLike Diffusion Y VF Control Args] [DiffEqSpace Y] [MilsteinControl Control] :
    MilsteinJacobianControlLike Diffusion Y VF Control Args where
  jacobianProd term t y args control :=
    finiteDiffJacobianFromVfProd
      (Diffusion := Diffusion) (Y := Y) (VF := VF) (Control := Control) (Args := Args)
      term t y args control

def Milstein.solver {Drift Diffusion Y VFd VFg Control Args : Type}
    [TermLike Drift Y VFd Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [MilsteinJacobianControlLike Diffusion Y VFg Control Args]
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
    let diffDerivInst := (inferInstance : MilsteinJacobianControlLike Diffusion Y VFg Control Args)
    let dt := driftInst.contr drift t0 t1
    let dControl := diffInst.contr diffusion t0 t1
    let f0 := driftInst.vf_prod drift t0 y0 args dt
    let g0 := diffInst.vf_prod diffusion t0 y0 args dControl
    let gg0 := diffDerivInst.jacobianProd diffusion t0 y0 args dControl
    let qv := MilsteinControl.quadraticVariation dControl
    let corr := (0.5 * (qv - dt)) * gg0
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

instance : ExplicitSolver Milstein := ⟨True.intro⟩
instance : ItoSolver Milstein := ⟨True.intro⟩

end DiffEq
end torch
