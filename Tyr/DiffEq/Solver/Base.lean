import Tyr.DiffEq.Term
import Tyr.DiffEq.Solution

namespace torch
namespace DiffEq

/-! ## Solver Base Types -/

inductive TermStructure where
  | single
  | pair
  | multi
  deriving Repr, BEq

namespace TermStructure

def arity? : TermStructure → Option Nat
  | .single => some 1
  | .pair => some 2
  | .multi => none

def ofArity (arity : Nat) : TermStructure :=
  if arity <= 1 then .single else if arity == 2 then .pair else .multi

def ofArity? (arity? : Option Nat) : TermStructure :=
  match arity? with
  | some arity => ofArity arity
  | none => .multi

end TermStructure

structure TermStructureMeta where
  arity? : Option Nat := none
  layoutTag? : Option String := none
  partitionArities? : Option (Array Nat) := none
  deriving Repr, BEq

namespace TermStructureMeta

def single : TermStructureMeta :=
  {
    arity? := some 1
    layoutTag? := some "single"
  }

def pair : TermStructureMeta :=
  {
    arity? := some 2
    layoutTag? := some "pair"
    partitionArities? := some #[1, 1]
  }

def array (arity : Nat) : TermStructureMeta :=
  {
    arity? := some arity
    layoutTag? := some "array"
  }

def ofTerm {Term : Type} [torch.DiffEq.TermShape Term] (term : Term) : TermStructureMeta :=
  {
    arity? := (inferInstance : torch.DiffEq.TermShape Term).arity? term
    layoutTag? := (inferInstance : torch.DiffEq.TermShape Term).layoutTag? term
  }

end TermStructureMeta

structure StepOutput (Y DenseInfo SolverState : Type) where
  y1 : Y
  yError : Option Y
  denseInfo : DenseInfo
  solverState : SolverState
  result : Result

structure AbstractSolver (Term Y VF Control Args : Type) where
  SolverState : Type
  DenseInfo : Type
  termStructure : TermStructure := TermStructure.single
  termStructureMeta : Option TermStructureMeta := none
  order : Term → Nat := fun _ => 1
  strongOrder : Term → Float := fun _ => 0.0
  errorOrder : Term → Float := fun term =>
    let strong := strongOrder term
    if strong > 0.0 then strong + 0.5 else Float.ofNat (order term)
  init : Term → Time → Time → Y → Args → SolverState
  step : Term → Time → Time → Y → Args → SolverState → Bool → StepOutput Y DenseInfo SolverState
  func : Term → Time → Y → Args → VF
  interpolation : DenseInfo → DenseInterpolation Y

namespace AbstractSolver

def termArity? (solver : AbstractSolver Term Y VF Control Args) : Option Nat :=
  match solver.termStructureMeta with
  | some structureMeta =>
      match structureMeta.arity? with
      | some arity => some arity
      | none => TermStructure.arity? solver.termStructure
  | none => TermStructure.arity? solver.termStructure

def termLayoutTag? (solver : AbstractSolver Term Y VF Control Args) : Option String :=
  match solver.termStructureMeta with
  | some structureMeta => structureMeta.layoutTag?
  | none => none

def termStructureKind (solver : AbstractSolver Term Y VF Control Args) : TermStructure :=
  solver.termStructure

def withTermStructureMeta (solver : AbstractSolver Term Y VF Control Args)
    (structureMeta : TermStructureMeta) : AbstractSolver Term Y VF Control Args :=
  {
    solver with
    termStructure := TermStructure.ofArity? structureMeta.arity?
    termStructureMeta := some structureMeta
  }

def withInferredTermStructure {Term Y VF Control Args : Type}
    [torch.DiffEq.TermShape Term]
    (solver : AbstractSolver Term Y VF Control Args)
    (term : Term) : AbstractSolver Term Y VF Control Args :=
  withTermStructureMeta solver (TermStructureMeta.ofTerm term)

end AbstractSolver

/-! ## Marker Traits -/

class ExplicitSolver (S : Type) : Prop where
  isTrue : True
class ImplicitSolver (S : Type) : Prop where
  isTrue : True
class AdaptiveSolver (S : Type) : Prop where
  isTrue : True
class ItoSolver (S : Type) : Prop where
  isTrue : True
class StratonovichSolver (S : Type) : Prop where
  isTrue : True
class SymplecticSolver (S : Type) : Prop where
  isTrue : True
class ReversibleSolver (S : Type) : Prop where
  isTrue : True
class WrappedSolver (S : Type) : Prop where
  isTrue : True

end DiffEq
end torch
