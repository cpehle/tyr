import Tyr.DiffEq.Solver.SEA

namespace torch
namespace DiffEq

/-! ## ALIGN Solver (compatibility wrapper)

Initial implementation mapped to SEA while specialized underdamped-Langevin
terms are being expanded.
-/

structure ALIGN where
  deriving Inhabited

def ALIGN.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  SEA.solver

instance : ExplicitSolver ALIGN := ⟨True.intro⟩
instance : StratonovichSolver ALIGN := ⟨True.intro⟩

end DiffEq
end torch
