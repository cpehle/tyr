import Tyr.DiffEq.Solver.ShARK

namespace torch
namespace DiffEq

/-! ## ShOULD Solver (compatibility wrapper)

Initial implementation mapped to ShARK while specialized underdamped-Langevin
terms are being expanded.
-/

structure ShOULD where
  deriving Inhabited

def ShOULD.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  ShARK.solver

instance : ExplicitSolver ShOULD := ⟨True.intro⟩
instance : StratonovichSolver ShOULD := ⟨True.intro⟩

end DiffEq
end torch
