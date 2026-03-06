import Tyr.DiffEq.Solver.SlowRK

namespace torch
namespace DiffEq

/-! ## QUICSORT Solver (compatibility wrapper)

Initial implementation mapped to SlowRK as a practical baseline while
specialized QUICSORT coefficients are being integrated.
-/

structure QUICSORT where
  deriving Inhabited

def QUICSORT.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  SlowRK.solver

instance : ExplicitSolver QUICSORT := ⟨True.intro⟩
instance : StratonovichSolver QUICSORT := ⟨True.intro⟩

end DiffEq
end torch
