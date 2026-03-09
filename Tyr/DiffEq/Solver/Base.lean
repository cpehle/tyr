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
  tree? : Option torch.DiffEq.TermTree := none
  treeDepth? : Option Nat := none
  leafArities? : Option (Array Nat) := none
  deriving Repr, BEq

namespace TermStructureMeta

def single : TermStructureMeta :=
  {
    arity? := some 1
    layoutTag? := some "single"
    tree? := some TermTree.single
    treeDepth? := some 1
    leafArities? := some #[1]
  }

def pair : TermStructureMeta :=
  {
    arity? := some 2
    layoutTag? := some "pair"
    partitionArities? := some #[1, 1]
    tree? := some (.pair TermTree.single TermTree.single)
    treeDepth? := some 2
    leafArities? := some #[1, 1]
  }

def array (arity : Nat) : TermStructureMeta :=
  {
    arity? := some arity
    layoutTag? := some "array"
    partitionArities? := some (Array.replicate arity 1)
    tree? := some (.array arity TermTree.single)
    treeDepth? := some 2
    leafArities? := some (Array.replicate arity 1)
  }

def effectiveArity? (structureMeta : TermStructureMeta) : Option Nat :=
  match structureMeta.arity? with
  | some arity => some arity
  | none =>
      match structureMeta.tree? with
      | some tree => TermTree.arity? tree
      | none => none

def effectiveLayoutTag? (structureMeta : TermStructureMeta) : Option String :=
  match structureMeta.layoutTag? with
  | some tag => some tag
  | none =>
      match structureMeta.tree? with
      | some tree => some (TermTree.layoutTag tree)
      | none => none

def effectivePartitionArities? (structureMeta : TermStructureMeta) : Option (Array Nat) :=
  match structureMeta.partitionArities? with
  | some partitionArities => some partitionArities
  | none =>
      match structureMeta.tree? with
      | some tree => TermTree.partitionArities? tree
      | none => none

def effectiveTreeDepth? (structureMeta : TermStructureMeta) : Option Nat :=
  match structureMeta.treeDepth? with
  | some depth => some depth
  | none =>
      match structureMeta.tree? with
      | some tree => some (TermTree.depth tree)
      | none => none

def effectiveLeafArities? (structureMeta : TermStructureMeta) : Option (Array Nat) :=
  match structureMeta.leafArities? with
  | some leafArities => some leafArities
  | none =>
      match structureMeta.tree? with
      | some tree => TermTree.leafArities? tree
      | none => none

def ofTerm {Term : Type} [torch.DiffEq.TermShape Term] (term : Term) : TermStructureMeta :=
  let shape := (inferInstance : torch.DiffEq.TermShape Term)
  let arity? := shape.arity? term
  let tree? :=
    match shape.tree? term with
    | some tree => some tree
    | none =>
        match arity? with
        | some _ => some (.leaf arity?)
        | none => none
  let inferredArity? :=
    match arity? with
    | some arity => some arity
    | none =>
        match tree? with
        | some tree => TermTree.arity? tree
        | none => none
  let inferredLayoutTag? :=
    match shape.layoutTag? term with
    | some layoutTag => some layoutTag
    | none =>
        match tree? with
        | some tree => some (TermTree.layoutTag tree)
        | none => none
  let inferredPartitionArities? :=
    match shape.partitionArities? term with
    | some partitionArities => some partitionArities
    | none =>
        match tree? with
        | some tree => TermTree.partitionArities? tree
        | none => none
  {
    arity? := inferredArity?
    layoutTag? := inferredLayoutTag?
    partitionArities? := inferredPartitionArities?
    tree? := tree?
    treeDepth? := tree?.map TermTree.depth
    leafArities? := tree?.bind TermTree.leafArities?
  }

end TermStructureMeta

structure StepOutput (Y DenseInfo SolverState : Type) where
  y1 : Y
  yError : Option Y
  denseInfo : DenseInfo
  solverState : SolverState
  result : Result

structure ExplicitRKAdjointTableau where
  a : Array (Array Time)
  b : Array Time
  c : Array Time
  deriving Inhabited

inductive ODEStepAdjoint where
  | explicitRK (tableau : ExplicitRKAdjointTableau)
  deriving Inhabited

structure AbstractSolver (Term Y VF Control Args : Type) where
  SolverState : Type
  DenseInfo : Type
  termStructure : TermStructure := TermStructure.single
  termStructureMeta : Option TermStructureMeta := none
  odeStepAdjoint? : Option ODEStepAdjoint := none
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
      match structureMeta.effectiveArity? with
      | some arity => some arity
      | none => TermStructure.arity? solver.termStructure
  | none => TermStructure.arity? solver.termStructure

def termLayoutTag? (solver : AbstractSolver Term Y VF Control Args) : Option String :=
  match solver.termStructureMeta with
  | some structureMeta => structureMeta.effectiveLayoutTag?
  | none => none

def termPartitionArities? (solver : AbstractSolver Term Y VF Control Args) : Option (Array Nat) :=
  match solver.termStructureMeta with
  | some structureMeta => structureMeta.effectivePartitionArities?
  | none => none

def termTree? (solver : AbstractSolver Term Y VF Control Args) : Option TermTree :=
  match solver.termStructureMeta with
  | some structureMeta => structureMeta.tree?
  | none => none

def termTreeDepth? (solver : AbstractSolver Term Y VF Control Args) : Option Nat :=
  match solver.termStructureMeta with
  | some structureMeta => structureMeta.effectiveTreeDepth?
  | none => none

def termLeafArities? (solver : AbstractSolver Term Y VF Control Args) : Option (Array Nat) :=
  match solver.termStructureMeta with
  | some structureMeta => structureMeta.effectiveLeafArities?
  | none => none

def termStructureKind (solver : AbstractSolver Term Y VF Control Args) : TermStructure :=
  solver.termStructure

def withTermStructureMeta (solver : AbstractSolver Term Y VF Control Args)
    (structureMeta : TermStructureMeta) : AbstractSolver Term Y VF Control Args :=
  let arity? := structureMeta.effectiveArity?
  {
    solver with
    termStructure := TermStructure.ofArity? arity?
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
