import Tyr.DiffEq.Solver.GeneralShARK
import Tyr.DiffEq.Solver.HalfSolver

namespace torch
namespace DiffEq

/-! ## SPaRK Solver (compatibility wrapper)

Uses `HalfSolver` around `GeneralShARK` to expose an embedded error estimate.
-/

structure SPaRK where
  deriving Inhabited

def SPaRK.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  HalfSolver.solver
    (GeneralShARK.solver
      (Drift := Drift)
      (Diffusion := Diffusion)
      (Y := Y)
      (VFg := VFg)
      (Control := Control)
      (Args := Args))

instance : ExplicitSolver SPaRK := ⟨True.intro⟩
instance : StratonovichSolver SPaRK := ⟨True.intro⟩
instance : AdaptiveSolver SPaRK := ⟨True.intro⟩

end DiffEq
end torch
