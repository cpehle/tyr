import Tyr.DiffEq.Solver.RungeKutta

namespace torch
namespace DiffEq

/-! ## Tsitouras 5(4) Solver -/

structure Tsit5 where
  deriving Inhabited

def tsit5DenseKind : ExplicitRKDenseKind :=
  .tsit5Poly4

def Tsit5.solver {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] : AbstractSolver Term Y VF Time Args :=
  ExplicitRK.solver { tableau := tsit5Tableau, denseKind := tsit5DenseKind }

instance : ExplicitSolver Tsit5 := ⟨True.intro⟩
instance : AdaptiveSolver Tsit5 := ⟨True.intro⟩

end DiffEq
end torch
