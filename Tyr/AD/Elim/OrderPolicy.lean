import Tyr.AD.JaxprLike.Core

/-!
# Tyr.AD.Elim.OrderPolicy

Order-policy representation shared by Tyr elimination and AlphaGrad integration.
The intent is to keep ID-space conversions explicit and validated.
-/

namespace Tyr.AD.Elim

/-- AlphaGrad action IDs live in 0-based action space. -/
abbrev ActionId0 := Nat

/-- Graphax/Tyr elimination vertices live in 1-based space. -/
abbrev VertexId1 := Nat

inductive CommPattern where
  | allReduce
  | allGather
  | reduceScatter
  | pointToPoint
  deriving Repr, BEq, Inhabited, DecidableEq

structure CommHint where
  pattern : CommPattern
  bytes : Nat := 0
  collectiveCount : Nat := 0
  deriving Repr, Inhabited

structure ConstraintSpec where
  /-- Hard ordering edges `(u, v)` meaning `u` must be eliminated before `v`. -/
  hardPrecedence : Array (VertexId1 × VertexId1) := #[]
  /-- Soft ordering edges with penalty weight. -/
  softPrecedence : Array (VertexId1 × VertexId1 × Float) := #[]
  /-- Optional grouped hints (kept abstract for now). -/
  groups : Array (Array VertexId1) := #[]
  /-- Optional communication hints consumed by comm-aware cost models. -/
  commHints : Array CommHint := #[]
  deriving Repr, Inhabited

inductive OrderPolicy where
  | explicitVertex (order1 : Array VertexId1)
  | constrainedVertex (base : Option (Array VertexId1)) (constraints : ConstraintSpec := {})
  | alphaGradAction (actions0 : Array ActionId0) (constraints : Option ConstraintSpec := none)
  | heuristic (name : String)
  deriving Repr, Inhabited

structure NormalizedOrderPolicy where
  /-- Optional total order in Graphax/Tyr vertex space. -/
  baseOrder1? : Option (Array VertexId1) := none
  /-- Constraint set to enforce during scheduling/execution. -/
  constraints : ConstraintSpec := {}
  /-- Source tag to preserve policy provenance in diagnostics. -/
  source : String := "unspecified"
  deriving Repr, Inhabited

def isValidActionId (numVertices : Nat) (action : ActionId0) : Bool :=
  action < numVertices

def isValidVertexId (numVertices : Nat) (vertex : VertexId1) : Bool :=
  (0 < vertex) && (vertex <= numVertices)

/-- True when `xs` contains no duplicates. -/
def hasNoDuplicates (xs : Array Nat) : Bool := Id.run do
  let mut seen : Std.HashSet Nat := {}
  for x in xs do
    if seen.contains x then
      return false
    seen := seen.insert x
  return true

def validateActionIds (numVertices : Nat) (actions0 : Array ActionId0) : Except String Unit :=
  match actions0.find? (fun action => !(isValidActionId numVertices action)) with
  | some bad =>
    .error s!"Invalid ActionId0 {bad}. Expected action ID in [0, {numVertices - 1}]."
  | none => .ok ()

def validateVertexIds (numVertices : Nat) (order1 : Array VertexId1) : Except String Unit :=
  match order1.find? (fun vertex => !(isValidVertexId numVertices vertex)) with
  | some bad =>
    .error s!"Invalid VertexId1 {bad}. Expected vertex ID in [1, {numVertices}]."
  | none => .ok ()

def validateExplicitVertexOrder
    (expectedEliminable : Nat)
    (numVertices : Nat)
    (order1 : Array VertexId1) :
    Except String Unit :=
  if order1.size != expectedEliminable then
    .error s!"Custom order length {order1.size} does not match expected eliminable count {expectedEliminable}."
  else
    match validateVertexIds numVertices order1 with
    | .error msg => .error msg
    | .ok () =>
      if hasNoDuplicates order1 then
        .ok ()
      else
        .error "Custom order contains duplicate vertex IDs."

def validateAlphaGradActionOrder
    (expectedEliminable : Nat)
    (numVertices : Nat)
    (actions0 : Array ActionId0) :
    Except String Unit :=
  if actions0.size != expectedEliminable then
    .error s!"Action sequence length {actions0.size} does not match expected eliminable count {expectedEliminable}."
  else
    match validateActionIds numVertices actions0 with
    | .error msg => .error msg
    | .ok () =>
      if hasNoDuplicates actions0 then
        .ok ()
      else
        .error "Action sequence contains duplicate action IDs."

def normalizeOrderPolicyShape
    (expectedEliminable : Nat)
    (numVertices : Nat)
    (policy : OrderPolicy) :
    Except String NormalizedOrderPolicy :=
  match policy with
  | .explicitVertex order1 =>
    match validateExplicitVertexOrder expectedEliminable numVertices order1 with
    | .error msg => .error msg
    | .ok () => .ok { baseOrder1? := some order1, source := "explicit-vertex" }
  | .constrainedVertex base constraints =>
    match base with
    | some order1 =>
      match validateExplicitVertexOrder expectedEliminable numVertices order1 with
      | .error msg => .error msg
      | .ok () =>
        .ok { baseOrder1? := some order1, constraints := constraints, source := "constrained-vertex" }
    | none =>
      .ok { constraints := constraints, source := "constrained-vertex" }
  | .alphaGradAction actions0 constraints? =>
    match validateAlphaGradActionOrder expectedEliminable numVertices actions0 with
    | .error msg => .error msg
    | .ok () =>
      .ok { constraints := constraints?.getD {}, source := "alphagrad-action" }
  | .heuristic name =>
    .ok { source := s!"heuristic:{name}" }

end Tyr.AD.Elim
