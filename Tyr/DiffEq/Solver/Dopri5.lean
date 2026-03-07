import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Dormand–Prince 5(4) Solver -/

structure Dopri5 where
  deriving Inhabited

def dopri5DenseKind : ExplicitRKDenseKind :=
  .dopri5Poly4

def Dopri5.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args :=
  ExplicitRK.solver { tableau := dopri5Tableau, denseKind := dopri5DenseKind }

instance : ExplicitSolver Dopri5 := ⟨True.intro⟩
instance : AdaptiveSolver Dopri5 := ⟨True.intro⟩

end DiffEq
end torch
