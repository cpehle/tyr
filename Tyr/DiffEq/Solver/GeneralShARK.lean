import Tyr.DiffEq.Solver.SRK

namespace torch
namespace DiffEq

/-! ## General ShARK Stochastic Runge-Kutta Solver (Stratonovich) -/

structure GeneralShARK where
  deriving Inhabited

private def vec3 {α : Type} (a b c : α) : Vector 3 α := ⟨#[a, b, c], by simp⟩

private def generalSharkTableau : StochasticButcherTableau 3 := {
  a := vec3 #[] #[0.0] #[0.0, 5.0 / 6.0]
  bSol := vec3 0.0 0.4 0.6
  bErr := none
  c := vec3 0.0 0.0 (5.0 / 6.0)
  coeffsW := StochasticCoeffs.general {
    a := vec3 #[] #[0.0] #[0.0, 5.0 / 6.0]
    bSol := vec3 0.0 0.4 0.6
  }
  coeffsH := some (StochasticCoeffs.general {
    a := vec3 #[] #[1.0] #[1.0, 0.0]
    bSol := vec3 0.0 1.2 (-1.2)
  })
  order := 2
  strongOrder := 0.5
}

def GeneralShARK.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  StochasticRK.solver { tableau := generalSharkTableau }

instance : ExplicitSolver GeneralShARK := ⟨True.intro⟩
instance : StratonovichSolver GeneralShARK := ⟨True.intro⟩

end DiffEq
end torch
