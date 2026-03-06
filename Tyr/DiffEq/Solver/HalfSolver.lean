import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Half Solver Wrapper

Wraps an existing solver and estimates local error by comparing:
1) one full step `t0 -> t1`
2) two half-steps `t0 -> tm -> t1`
-/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure HalfSolver where
  deriving Inhabited

namespace HalfSolver

def solver {Term Y VF Control Args : Type}
    [DiffEqSpace Y]
    (inner : AbstractSolver Term Y VF Control Args) :
    AbstractSolver Term Y VF Control Args := {
  SolverState := inner.SolverState
  DenseInfo := inner.DenseInfo
  termStructure := inner.termStructure
  termStructureMeta := inner.termStructureMeta
  order := inner.order
  strongOrder := inner.strongOrder
  errorOrder := inner.errorOrder
  init := inner.init
  step := fun term t0 t1 y0 args state madeJump =>
    let tm := 0.5 * (t0 + t1)
    let full := inner.step term t0 t1 y0 args state madeJump
    if full.result != Result.successful then
      {
        y1 := full.y1
        yError := full.yError
        denseInfo := full.denseInfo
        solverState := full.solverState
        result := full.result
      }
    else
      let half1 := inner.step term t0 tm y0 args state madeJump
      if half1.result != Result.successful then
        {
          y1 := half1.y1
          yError := half1.yError
          denseInfo := half1.denseInfo
          solverState := half1.solverState
          result := half1.result
        }
      else
        let half2 := inner.step term tm t1 half1.y1 args half1.solverState false
        if half2.result != Result.successful then
          {
            y1 := half2.y1
            yError := half2.yError
            denseInfo := half2.denseInfo
            solverState := half2.solverState
            result := half2.result
          }
        else
          let yErr := half2.y1 - full.y1
          {
            y1 := half2.y1
            yError := some yErr
            denseInfo := half2.denseInfo
            solverState := half2.solverState
            result := Result.successful
          }
  func := inner.func
  interpolation := inner.interpolation
}

end HalfSolver

instance : WrappedSolver HalfSolver := ⟨True.intro⟩
instance : AdaptiveSolver HalfSolver := ⟨True.intro⟩

end DiffEq
end torch
