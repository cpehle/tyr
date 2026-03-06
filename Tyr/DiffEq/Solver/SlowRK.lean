import Tyr.DiffEq.Solver.GeneralShARK

namespace torch
namespace DiffEq

/-! ## SlowRK Solver (compatibility wrapper)

This currently reuses the general ShARK tableau as a practical baseline.
-/

structure SlowRK where
  deriving Inhabited

def SlowRK.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  GeneralShARK.solver

instance : ExplicitSolver SlowRK := ⟨True.intro⟩
instance : StratonovichSolver SlowRK := ⟨True.intro⟩

end DiffEq
end torch
