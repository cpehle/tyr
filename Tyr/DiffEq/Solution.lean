import Tyr.DiffEq.Path
import Tyr.DiffEq.Interpolation

namespace torch
namespace DiffEq

/-! ## Solution and Result Types -/

inductive Result where
  | successful
  | maxStepsReached
  | dtMinReached
  | eventOccurred
  | maxStepsRejected
  | internalError
  deriving Repr, BEq

namespace Result

def isOkay (r : Result) : Bool :=
  r == Result.successful || r == Result.eventOccurred

def isSuccessful (r : Result) : Bool :=
  r == Result.successful

def isEvent (r : Result) : Bool :=
  r == Result.eventOccurred

def message (r : Result) : String :=
  match r with
  | successful => ""
  | maxStepsReached => "The maximum number of solver steps was reached."
  | dtMinReached => "The minimum step size was reached in the solver."
  | eventOccurred => "Terminating differential equation solve because an event occurred."
  | maxStepsRejected => "Maximum number of rejected steps was reached."
  | internalError => "An internal error occurred in the solver."

end Result

structure Solution (Y SolverState ControllerState : Type) where
  t0 : Time
  t1 : Time
  ts : Option (Array Time)
  ys : Option (Array Y)
  interpolation : Option (DenseInterpolation Y)
  stats : List (String × Nat)
  result : Result
  solverState : Option SolverState
  controllerState : Option ControllerState
  madeJump : Option Bool
  eventMask : Option (Array Bool)

namespace Solution

def evaluate [Inhabited Y] (sol : Solution Y SolverState ControllerState) (t0 : Time)
    (t1 : Option Time := none) (left : Bool := true) : Y :=
  match sol.interpolation with
  | none => panic! "Dense solution has not been saved; pass SaveAt(dense=True)."
  | some interp => interp.evaluate t0 t1 left

def derivative [Inhabited Y] (sol : Solution Y SolverState ControllerState) (t : Time)
    (left : Bool := true) : Y :=
  match sol.interpolation with
  | none => panic! "Dense solution has not been saved; pass SaveAt(dense=True)."
  | some interp => interp.derivative t left

end Solution

end DiffEq
end torch
