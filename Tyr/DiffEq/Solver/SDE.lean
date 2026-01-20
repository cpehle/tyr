import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## SDE Type Aliases and Helpers -/

abbrev SDETerm (Drift Diffusion : Type) : Type :=
  MultiTerm Drift Diffusion

abbrev SDESolver (Drift Diffusion Y VFd VFg Control Args : Type) : Type 1 :=
  AbstractSolver (MultiTerm Drift Diffusion) Y (VFd × VFg) (Time × Control) Args

namespace SDETerm

def mk {Drift Diffusion : Type} (drift : Drift) (diffusion : Diffusion) : SDETerm Drift Diffusion :=
  { term1 := drift, term2 := diffusion }

end SDETerm

end DiffEq
end torch
