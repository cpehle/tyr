import Tyr.AD.Elim.Graph

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
  | forward
  | reverse
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

private def graphVertexUpperBound (g : ElimGraph) : Nat :=
  (vertices g).foldl (init := 0) max

private def normalizeHeuristicName (name : String) : String :=
  name.trimAscii.toString.toLower

private def validateExplicitVertexOrderAgainstEliminable
    (eliminable : Array VertexId1)
    (order1 : Array VertexId1) :
    Except String Unit := do
  let mut allowed : Std.HashSet VertexId1 := {}
  for v in eliminable do
    allowed := allowed.insert v
  match order1.find? (fun v => !allowed.contains v) with
  | some bad =>
    .error s!"Custom order references non-eliminable vertex {bad}."
  | none =>
    .ok ()

/--
Validate an explicit vertex order against the graph's actual eliminable set,
not only the ambient `[1, n]` ID domain.
-/
def validateExplicitVertexOrderAgainstGraph
    (g : ElimGraph)
    (order1 : Array VertexId1) :
    Except String Unit := do
  let numVertices := graphVertexUpperBound g
  validateExplicitVertexOrder g.eliminable.size numVertices order1
  validateExplicitVertexOrderAgainstEliminable g.eliminable order1

/--
Validate an AlphaGrad action sequence against the graph's explicit eliminable set.
The action domain remains `[0, maxVertexId)` with `vertex = action + 1`.
-/
def validateAlphaGradActionOrderAgainstGraph
    (g : ElimGraph)
    (actions0 : Array ActionId0) :
    Except String Unit := do
  let numVertices := graphVertexUpperBound g
  validateAlphaGradActionOrder g.eliminable.size numVertices actions0
  let order1 := actions0.map (fun action => action + 1)
  validateExplicitVertexOrderAgainstEliminable g.eliminable order1

def normalizeOrderPolicyShape
    (expectedEliminable : Nat)
    (numVertices : Nat)
    (policy : OrderPolicy) :
    Except String NormalizedOrderPolicy :=
  match policy with
  | .forward =>
    .ok { source := "forward" }
  | .reverse =>
    .ok { source := "reverse" }
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

/--
Resolve an order policy against a concrete elimination graph, using the graph's
explicit eliminable partition for Graphax-style `forward` / `reverse` / custom
order semantics.
-/
def normalizeOrderPolicyAgainstGraph
    (g : ElimGraph)
    (policy : OrderPolicy) :
    Except String NormalizedOrderPolicy :=
  match policy with
  | .forward =>
    .ok {
      baseOrder1? := some (forwardEliminationOrder g)
      source := "forward"
    }
  | .reverse =>
    .ok {
      baseOrder1? := some (reverseEliminationOrder g)
      source := "reverse"
    }
  | .explicitVertex order1 =>
    match validateExplicitVertexOrderAgainstGraph g order1 with
    | .error msg => .error msg
    | .ok () =>
      .ok {
        baseOrder1? := some order1
        source := "explicit-vertex"
      }
  | .constrainedVertex base constraints =>
    match base with
    | some order1 =>
      match validateExplicitVertexOrderAgainstGraph g order1 with
      | .error msg => .error msg
      | .ok () =>
        .ok {
          baseOrder1? := some order1
          constraints := constraints
          source := "constrained-vertex"
        }
    | none =>
      .ok {
        constraints := constraints
        source := "constrained-vertex"
      }
  | .alphaGradAction actions0 constraints? =>
    match validateAlphaGradActionOrderAgainstGraph g actions0 with
    | .error msg => .error msg
    | .ok () =>
      .ok {
        baseOrder1? := some (actions0.map (fun action => action + 1))
        constraints := constraints?.getD {}
        source := "alphagrad-action"
      }
  | .heuristic name =>
    let name' := normalizeHeuristicName name
    if name' = "fwd" || name' = "forward" then
      .ok {
        baseOrder1? := some (forwardEliminationOrder g)
        source := s!"heuristic:{name}"
      }
    else if name' = "rev" || name' = "reverse" then
      .ok {
        baseOrder1? := some (reverseEliminationOrder g)
        source := s!"heuristic:{name}"
      }
    else
      .ok { source := s!"heuristic:{name}" }

end Tyr.AD.Elim
