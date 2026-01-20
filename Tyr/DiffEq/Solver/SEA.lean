import Tyr.DiffEq.Solver.SRK

namespace torch
namespace DiffEq

/-! ## SEA Stochastic Runge-Kutta Solver (additive noise, Stratonovich) -/

structure SEA where
  deriving Inhabited

private def seaTableau : StochasticButcherTableau 1 := {
  a := vec1 #[]
  bSol := vec1 1.0
  bErr := none
  c := vec1 0.0
  coeffsW := StochasticCoeffs.additive { a := vec1 0.5, bSol := 1.0 }
  coeffsH := some (StochasticCoeffs.additive { a := vec1 1.0, bSol := 0.0 })
  order := 1
  strongOrder := 1.0
}

def SEA.solver {Drift Diffusion Y VFg Control Args : Type}
    [TermLike Drift Y Y Time Args]
    [TermLike Diffusion Y VFg Control Args]
    [SpaceTimeLevyAreaLike Control Float]
    [SpaceTimeLevyAreaBuild Control Float]
    [DiffEqSpace Y]
    [DiffEqSpace VFg] :
    AbstractSolver (MultiTerm Drift Diffusion) Y (Y × VFg) (Time × Control) Args :=
  StochasticRK.solver { tableau := seaTableau }

instance : ExplicitSolver SEA := ⟨True.intro⟩
instance : StratonovichSolver SEA := ⟨True.intro⟩

end DiffEq
end torch
