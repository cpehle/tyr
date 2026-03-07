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
  t0 : Bool := false
  t1 : Bool := false
  ts : Option (Array Time) := none
  steps : StepCadence := (0 : Nat)
  dense : Bool := false
  solverState : Bool := false
  controllerState : Bool := false
  madeJump : Bool := false
  subs : Array SubSaveAt := #[]
  deriving Inhabited

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
  deriving Inhabited

namespace SaveAt

namespace SubSaveAt

def flatten (root : SubSaveAt) : Array SubSaveAt := Id.run do
  let mut out : Array SubSaveAt := #[]
  let mut pending : Array SubSaveAt := #[root]
  while pending.size > 0 do
    let i := pending.size - 1
    let node := pending[i]!
    pending := pending.pop
    out := out.push node
    for offset in [:node.subs.size] do
      let childIdx := node.subs.size - 1 - offset
      pending := pending.push (node.subs[childIdx]!)
  return out

end SubSaveAt

private def rootSubSaveAt (saveat : SaveAt) : SubSaveAt := {
  t0 := saveat.t0
  t1 := saveat.t1
  ts := saveat.ts
  steps := saveat.steps
  dense := saveat.dense
  solverState := saveat.solverState
  controllerState := saveat.controllerState
  madeJump := saveat.madeJump
  subs := saveat.subs
}

private def allSubs (saveat : SaveAt) : Array SubSaveAt :=
  SaveAt.SubSaveAt.flatten (rootSubSaveAt saveat)

private def payloadRoots (saveat : SaveAt) : Array SubSaveAt :=
  if saveat.subs.size == 0 then
    #[rootSubSaveAt saveat]
  else
    saveat.subs

private def payloadTreeSubs (saveat : SaveAt) : Array SubSaveAt := Id.run do
  let mut out : Array SubSaveAt := #[]
  for root in payloadRoots saveat do
    out := out ++ SaveAt.SubSaveAt.flatten root
  return out

private def payloadLeaves (saveat : SaveAt) : Array SubSaveAt :=
  (payloadTreeSubs saveat).filter (fun sub => sub.subs.size == 0)

private def hasPayloadOutput (sub : SubSaveAt) : Bool :=
  sub.t0 || sub.t1 || sub.ts.isSome || sub.steps.enabled

def payloadSubs (saveat : SaveAt) : Array SubSaveAt :=
  (payloadLeaves saveat).filter hasPayloadOutput

private def appendTs (acc : Array Time) (ts : Option (Array Time)) : Array Time :=
  match ts with
  | some xs => acc ++ xs
  | none => acc

def effectiveT0 (saveat : SaveAt) : Bool :=
  (payloadLeaves saveat).any (fun sub => sub.t0)

def effectiveT1 (saveat : SaveAt) : Bool :=
  (payloadLeaves saveat).any (fun sub => sub.t1)

def stepCadences (saveat : SaveAt) : Array Nat := Id.run do
  let mut cadences : Array Nat := #[]
  for sub in payloadLeaves saveat do
    let cadence : Nat := sub.steps
    if cadence != 0 then
      cadences := cadences.push cadence
  return cadences

def stepCadence (saveat : SaveAt) : Nat :=
  saveat.stepCadences.foldl
    (init := 0)
    (fun acc cadence =>
      if acc == 0 || cadence < acc then cadence else acc)

def stepsEnabled (saveat : SaveAt) : Bool :=
  saveat.stepCadences.size != 0

def shouldSaveAcceptedStep (saveat : SaveAt) (acceptedSteps : Nat) : Bool :=
  saveat.stepCadences.any (fun cadence => acceptedSteps % cadence == 0)

def effectiveTs (saveat : SaveAt) : Option (Array Time) :=
  let ts := Id.run do
    let mut out : Array Time := #[]
    for sub in payloadLeaves saveat do
      out := appendTs out sub.ts
    return out
  if ts.size == 0 then none else some ts

def effectiveDense (saveat : SaveAt) : Bool :=
  (allSubs saveat).any (fun sub => sub.dense)

def effectiveSolverState (saveat : SaveAt) : Bool :=
  (allSubs saveat).any (fun sub => sub.solverState)

def effectiveControllerState (saveat : SaveAt) : Bool :=
  (allSubs saveat).any (fun sub => sub.controllerState)

def effectiveMadeJump (saveat : SaveAt) : Bool :=
  (allSubs saveat).any (fun sub => sub.madeJump)

end SaveAt

end DiffEq
end torch
