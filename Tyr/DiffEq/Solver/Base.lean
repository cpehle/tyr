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
  order : Term → Nat := fun _ => 1
  strongOrder : Term → Float := fun _ => 0.0
  errorOrder : Term → Float := fun term =>
    let strong := strongOrder term
    if strong > 0.0 then strong + 0.5 else Float.ofNat (order term)
  init : Term → Time → Time → Y → Args → SolverState
  step : Term → Time → Time → Y → Args → SolverState → Bool → StepOutput Y DenseInfo SolverState
  func : Term → Time → Y → Args → VF
  interpolation : DenseInfo → DenseInterpolation Y

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
