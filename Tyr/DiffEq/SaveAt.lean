import Tyr.DiffEq.Types

namespace torch
namespace DiffEq

/-! ## SaveAt

Configuration for solution saving.
-/

structure StepCadence where
  value : Nat := 0
  deriving Inhabited, Repr, DecidableEq

instance : Coe Bool StepCadence where
  coe b := { value := if b then 1 else 0 }

instance : Coe Nat StepCadence where
  coe n := { value := n }

instance : Coe StepCadence Nat where
  coe c := c.value

instance : Coe StepCadence Bool where
  coe c := c.value != 0

namespace StepCadence

def enabled (c : StepCadence) : Bool :=
  c.value != 0

def shouldSave (c : StepCadence) (acceptedSteps : Nat) : Bool :=
  c.enabled && acceptedSteps % c.value == 0

end StepCadence

structure SubSaveAt where
  ts : Option (Array Time) := none
  steps : StepCadence := (0 : Nat)
  dense : Bool := false

structure SaveAt where
  t0 : Bool := false
  t1 : Bool := true
  ts : Option (Array Time) := none
  steps : StepCadence := (0 : Nat)
  dense : Bool := false
  solverState : Bool := false
  controllerState : Bool := false
  madeJump : Bool := false
  subs : Array SubSaveAt := #[]

namespace SaveAt

def stepCadence (saveat : SaveAt) : Nat :=
  let base : Nat := saveat.steps
  if base != 0 then
    base
  else
    match saveat.subs.find? (fun sub => (sub.steps : Nat) != 0) with
    | some sub => sub.steps
    | none => 0

def stepsEnabled (saveat : SaveAt) : Bool :=
  saveat.stepCadence != 0

def shouldSaveAcceptedStep (saveat : SaveAt) (acceptedSteps : Nat) : Bool :=
  let cadence := saveat.stepCadence
  cadence != 0 && acceptedSteps % cadence == 0

def effectiveTs (saveat : SaveAt) : Option (Array Time) :=
  match saveat.ts with
  | some ts => some ts
  | none =>
      match saveat.subs.find? (fun sub => sub.ts.isSome) with
      | some sub => sub.ts
      | none => none

def effectiveDense (saveat : SaveAt) : Bool :=
  saveat.dense || saveat.subs.any (fun sub => sub.dense)

end SaveAt

end DiffEq
end torch
