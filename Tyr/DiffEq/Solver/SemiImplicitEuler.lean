import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Semi-Implicit Euler Solver -/

local instance (priority := 5) [DiffEqSpace α] : HAdd α α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
local instance (priority := 5) [DiffEqSpace α] : HSub α α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
local instance (priority := 5) [DiffEqSpace α] : HMul Scalar α α :=
  _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure SemiImplicitEuler where
  deriving Inhabited

def SemiImplicitEuler.solver {PositionTerm VelocityTerm Y VFx VFv Cx Cv Args : Type}
    [TermLike PositionTerm Y VFx Cx Args]
    [TermLike VelocityTerm Y VFv Cv Args]
    [DiffEqSpace Y]
    [DiffEqSpace (Y × Y)] :
    AbstractSolver (MultiTerm PositionTerm VelocityTerm) (Y × Y)
      (VFx × VFv) (Cx × Cv) Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo (Y × Y)
  termStructure := TermStructure.pair
  order := fun _ => 1
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let positionTerm := terms.term1
    let velocityTerm := terms.term2
    let posInst := (inferInstance : TermLike PositionTerm Y VFx Cx Args)
    let velInst := (inferInstance : TermLike VelocityTerm Y VFv Cv Args)
    let (x0, v0) := y0
    let controlX := posInst.contr positionTerm t0 t1
    let controlV := velInst.contr velocityTerm t0 t1
    let dx := posInst.vf_prod positionTerm t0 v0 args controlX
    let x1 := x0 + dx
    let dv := velInst.vf_prod velocityTerm t0 x1 args controlV
    let v1 := v0 + dv
    let y1 := (x1, v1)
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := none
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun terms t y args =>
    let positionTerm := terms.term1
    let velocityTerm := terms.term2
    let posInst := (inferInstance : TermLike PositionTerm Y VFx Cx Args)
    let velInst := (inferInstance : TermLike VelocityTerm Y VFv Cv Args)
    let (x, v) := y
    (posInst.vf positionTerm t v args, velInst.vf velocityTerm t x args)
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver SemiImplicitEuler := ⟨True.intro⟩
instance : SymplecticSolver SemiImplicitEuler := ⟨True.intro⟩

end DiffEq
end torch
