import Tyr.AD.Elim.OrderPolicy

/-!
# Tyr.AD.Elim.ActionSpace

Checked conversions between the 0-based action space and the 1-based
elimination-vertex space.
-/

namespace Tyr.AD.Elim

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

def actionsToVerticesInSpace?
    (actionVertices : Array VertexId1)
    (actions0 : Array ActionId0) :
    Except String (Array VertexId1) :=
  Id.run do
    let mut out : Array VertexId1 := #[]
    for action in actions0 do
      match actionToVertexInSpace? actionVertices action with
      | .ok vertex => out := out.push vertex
      | .error err => return .error err
    return .ok out

def verticesToActionsInSpace?
    (actionVertices : Array VertexId1)
    (order1 : Array VertexId1) :
    Except String (Array ActionId0) :=
  Id.run do
    let mut out : Array ActionId0 := #[]
    for vertex in order1 do
      match vertexToActionInSpace? actionVertices vertex with
      | .ok action => out := out.push action
      | .error err => return .error err
    return .ok out

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

end Tyr.AD.Elim
