import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Classic RK4 Solver -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure RK4 where
  deriving Inhabited

def RK4.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args :=
  ExplicitRK.solver { tableau := rk4Tableau }

instance : ExplicitSolver RK4 := ⟨True.intro⟩

end DiffEq
end torch
