import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## SaveAt

Configuration for solution saving.
-/

structure SaveAt where
  t0 : Bool := false
  t1 : Bool := true
  ts : Option (Array Time) := none
  steps : Bool := false
  dense : Bool := false
  solverState : Bool := false
  controllerState : Bool := false
  madeJump : Bool := false

structure SubSaveAt where
  saveat : SaveAt

end DiffEq
end torch
