import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Ralston Solver -/

structure Ralston where
  deriving Inhabited

def Ralston.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args :=
  ExplicitRK.solver { tableau := ralstonTableau }

instance : ExplicitSolver Ralston := ⟨True.intro⟩
instance : StratonovichSolver Ralston := ⟨True.intro⟩

end DiffEq
end torch
