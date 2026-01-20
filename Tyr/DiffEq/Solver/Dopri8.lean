import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Dormand–Prince 8(7) Solver -/

structure Dopri8 where
  deriving Inhabited

def Dopri8.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args :=
  ExplicitRK.solver { tableau := dopri8Tableau }

instance : ExplicitSolver Dopri8 := ⟨True.intro⟩
instance : AdaptiveSolver Dopri8 := ⟨True.intro⟩

end DiffEq
end torch
