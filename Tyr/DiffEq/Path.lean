import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## Path Abstractions

Paths provide control increments over time intervals.
-/

structure AbstractPath (Control : Type) where
  t0 : Time
  t1 : Time
  evaluate : Time → Option Time → Bool → Control
  derivativeFn? : Option (Time → Bool → Control) := none

namespace AbstractPath

def derivative (path : AbstractPath Control) (t : Time) (left : Bool := true) : Option Control :=
  match path.derivativeFn? with
  | some derivativeFn => some (derivativeFn t left)
  | none => none

def withDerivative (path : AbstractPath Control) (derivativeFn : Time → Bool → Control) :
    AbstractPath Control :=
  { path with derivativeFn? := some derivativeFn }

def clearDerivative (path : AbstractPath Control) : AbstractPath Control :=
  { path with derivativeFn? := none }

end AbstractPath

end DiffEq
end torch
