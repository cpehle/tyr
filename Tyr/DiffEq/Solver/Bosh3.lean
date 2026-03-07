import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Bosh3 Solver -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure Bosh3 where
  deriving Inhabited

def Bosh3.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args :=
  ExplicitRK.solver { tableau := bosh3Tableau }

instance : ExplicitSolver Bosh3 := ⟨True.intro⟩
instance : StratonovichSolver Bosh3 := ⟨True.intro⟩

end DiffEq
end torch
