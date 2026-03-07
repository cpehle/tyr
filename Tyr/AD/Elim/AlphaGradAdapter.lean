import Tyr.AD.Elim.OrderPolicy

/-!
# Tyr.AD.Elim.AlphaGradAdapter

Adapters between AlphaGrad action space (0-based) and elimination vertex space (1-based).
-/

namespace Tyr.AD.Elim

/-- Canonical full-vertex AlphaGrad action space (`action = vertex - 1`). -/
def defaultActionVertices (numVertices : Nat) : Array VertexId1 :=
  ((List.range numVertices).map (fun i => i + 1)).toArray

/-- Lossless map from AlphaGrad action ID to Graphax/Tyr vertex ID. -/
def actionToVertex (action : ActionId0) : VertexId1 :=
  action + 1

/-- Domain-checked map from action to vertex. -/
def actionToVertex? (numVertices : Nat) (action : ActionId0) : Except String VertexId1 :=
  if isValidActionId numVertices action then
    .ok (actionToVertex action)
  else
    .error s!"Invalid ActionId0 {action}. Expected action ID in [0, {numVertices - 1}]."

/-- Domain-checked inverse map from vertex ID to action ID. -/
def vertexToAction? (numVertices : Nat) (vertex : VertexId1) : Except String ActionId0 :=
  if isValidVertexId numVertices vertex then
    .ok (vertex - 1)
  else
    .error s!"Invalid VertexId1 {vertex}. Expected vertex ID in [1, {numVertices}]."

/-- Domain-checked map from action index into an explicit action-space vertex table. -/
def actionToVertexInSpace?
    (actionVertices : Array VertexId1)
    (action : ActionId0) :
    Except String VertexId1 :=
  if action < actionVertices.size then
    match actionVertices[action]? with
    | some vertex => .ok vertex
    | none => .error s!"Action-space lookup failed at action {action}."
  else
    .error s!"Invalid ActionId0 {action}. Expected action ID in [0, {actionVertices.size - 1}] for action-space size {actionVertices.size}."

/-- Domain-checked inverse map from vertex into an explicit action-space table. -/
def vertexToActionInSpace?
    (actionVertices : Array VertexId1)
    (vertex : VertexId1) :
    Except String ActionId0 :=
  match actionVertices.findIdx? (fun v => v = vertex) with
  | some action => .ok action
  | none =>
    .error s!"VertexId1 {vertex} is not present in the configured action-space vertex set."

def actionsToVertices? (numVertices : Nat) (actions0 : Array ActionId0) : Except String (Array VertexId1) :=
  Id.run do
    let mut out : Array VertexId1 := #[]
    for action in actions0 do
      match actionToVertex? numVertices action with
      | .ok vertex => out := out.push vertex
      | .error err => return .error err
    return .ok out

def verticesToActions? (numVertices : Nat) (order1 : Array VertexId1) : Except String (Array ActionId0) :=
  Id.run do
    let mut out : Array ActionId0 := #[]
    for vertex in order1 do
      match vertexToAction? numVertices vertex with
      | .ok action => out := out.push action
      | .error err => return .error err
    return .ok out

/-- Action-level feasibility that composes elimination mask with constraint feasibility. -/
def actionFeasible
    (numVertices : Nat)
    (isEliminated : VertexId1 → Bool)
    (constraintFeasible : VertexId1 → Bool)
    (action : ActionId0) :
    Bool :=
  match actionToVertexInSpace? (defaultActionVertices numVertices) action with
  | .ok vertex => !(isEliminated vertex) && constraintFeasible vertex
  | .error _ => false

/-- Action-level feasibility against an explicit action-space vertex table. -/
def actionFeasibleInSpace
    (actionVertices : Array VertexId1)
    (isEliminated : VertexId1 → Bool)
    (constraintFeasible : VertexId1 → Bool)
    (action : ActionId0) :
    Bool :=
  match actionToVertexInSpace? actionVertices action with
  | .ok vertex => !(isEliminated vertex) && constraintFeasible vertex
  | .error _ => false

/--
Normalize an AlphaGrad action sequence into a strict elimination order in vertex space.
This is the compatibility boundary used before execution in Tyr/Graphax-style eliminators.
-/
def normalizeAlphaGradOrder
    (expectedEliminable : Nat)
    (numVertices : Nat)
    (actions0 : Array ActionId0)
    (constraints? : Option ConstraintSpec := none) :
    Except String (Array VertexId1 × ConstraintSpec) := do
  validateAlphaGradActionOrder expectedEliminable numVertices actions0
  let order1 ← actionsToVertices? numVertices actions0
  pure (order1, constraints?.getD {})

end Tyr.AD.Elim
