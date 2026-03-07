import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## SDE Type Aliases and Helpers -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

abbrev SDETerm (Drift Diffusion : Type) : Type :=
  MultiTerm Drift Diffusion

abbrev SDETerm3 (Drift Diffusion1 Diffusion2 : Type) : Type :=
  MultiTerm3 Drift Diffusion1 Diffusion2

abbrev SDETerm4 (Drift Diffusion1 Diffusion2 Diffusion3 : Type) : Type :=
  MultiTerm4 Drift Diffusion1 Diffusion2 Diffusion3

abbrev SDETermArray (Drift Diffusion : Type) : Type :=
  MultiTerm Drift (MultiTermArray Diffusion)

abbrev SDESolver (Drift Diffusion Y VFd VFg Control Args : Type) : Type 1 :=
  AbstractSolver (MultiTerm Drift Diffusion) Y (VFd × VFg) (Time × Control) Args

abbrev SDESolverArray (Drift Diffusion Y VFd VFg Control Args : Type) : Type 1 :=
  AbstractSolver (SDETermArray Drift Diffusion) Y (VFd × Array VFg) (Time × Array Control) Args

namespace SDETerm

def mk {Drift Diffusion : Type} (drift : Drift) (diffusion : Diffusion) : SDETerm Drift Diffusion :=
  { term1 := drift, term2 := diffusion }

def mk3 {Drift Diffusion1 Diffusion2 : Type}
    (drift : Drift) (diffusion1 : Diffusion1) (diffusion2 : Diffusion2) :
    SDETerm3 Drift Diffusion1 Diffusion2 :=
  MultiTerm.of3 drift diffusion1 diffusion2

def mk4 {Drift Diffusion1 Diffusion2 Diffusion3 : Type}
    (drift : Drift)
    (diffusion1 : Diffusion1)
    (diffusion2 : Diffusion2)
    (diffusion3 : Diffusion3) :
    SDETerm4 Drift Diffusion1 Diffusion2 Diffusion3 :=
  MultiTerm.of4 drift diffusion1 diffusion2 diffusion3

def mkArray {Drift Diffusion : Type}
    (drift : Drift) (diffusions : MultiTermArray Diffusion) : SDETermArray Drift Diffusion :=
  { term1 := drift, term2 := diffusions }

def mkArray? {Drift Diffusion : Type}
    (drift : Drift) (diffusions : Array Diffusion) : Option (SDETermArray Drift Diffusion) :=
  match MultiTermArray.ofArray? diffusions with
  | some ds => some (mkArray drift ds)
  | none => none

end SDETerm

end DiffEq
end torch
