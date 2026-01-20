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

namespace AbstractPath

def derivative (_path : AbstractPath Control) (_t : Time) (_left : Bool := true) : Option Control :=
  none

end AbstractPath

end DiffEq
end torch
