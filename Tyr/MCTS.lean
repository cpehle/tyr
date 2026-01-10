/-
  Monte Carlo Tree Search

  MCTS implementation for theorem proving, inspired by AlphaProof/nanoproof.
  Supports AND-OR game trees for proof search.
-/
import Std.Data.HashMap

namespace mcts

-- Float utilities (not in stdlib)
def floatMax (a b : Float) : Float := if a >= b then a else b
def floatMin (a b : Float) : Float := if a <= b then a else b
def floatInf : Float := 1e308
def floatNegInf : Float := -1e308

/-- Player type: OR selects any tactic, AND requires all goals -/
inductive Player
  | OR   -- Selects any child (existential: need one tactic to work)
  | AND  -- Requires all children (universal: need all subgoals)
  deriving Repr, BEq, Inhabited

instance : ToString Player where
  toString := fun
    | .OR => "OR"
    | .AND => "AND"

/-- Action in the proof search -/
inductive Action
  | tactic (s : String)  -- Apply a tactic
  | focus (idx : Nat)    -- Focus on a specific goal
  deriving Repr, BEq, Inhabited

instance : ToString Action where
  toString := fun
    | .tactic s => s!"tactic {s}"
    | .focus idx => s!"focus {idx}"

/-- Proof state representation -/
structure ProofState where
  goals : Array String      -- List of current goals (as strings)
  history : Array Action    -- Actions taken to reach this state
  isSolved : Bool := false
  deriving Repr, Inhabited

/-- Check if proof is complete -/
def ProofState.isComplete (s : ProofState) : Bool :=
  s.isSolved || s.goals.isEmpty

/-- MCTS Node in the search tree -/
structure Node where
  action : Option Action        -- Action that led to this node
  state : ProofState            -- Current proof state
  player : Player               -- Whose turn it is
  prior : Float                 -- Policy network prior P(a|s)
  value : Float := 0.0          -- Running value estimate
  visitCount : Nat := 0         -- N(s,a)
  children : Array Node := #[]  -- Child nodes
  expanded : Bool := false      -- Has this node been expanded?
  deriving Repr, Inhabited

/-- MCTS configuration -/
structure MCTSConfig where
  numSimulations : Nat := 800       -- Number of MCTS simulations per move
  pbCBase : Float := 19652.0        -- Exploration constant base
  pbCInit : Float := 1.25           -- Initial exploration constant
  valueDiscount : Float := 0.99     -- Discount factor for value backup
  explorationWeight : Float := 1.0  -- Weight for exploration bonus
  deriving Repr, Inhabited

/-- Compute PUCT score for child selection -/
def puctScore (parent : Node) (child : Node) (config : MCTSConfig) : Float :=
  let parentVisits := floatMax 1.0 parent.visitCount.toFloat
  let childVisits := floatMax 1.0 child.visitCount.toFloat

  -- Exploitation: Q(s,a)
  let q := if child.visitCount > 0 then child.value / childVisits else 0.0

  -- Exploration: c(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
  let pbC := Float.log ((parentVisits + config.pbCBase + 1.0) / config.pbCBase) + config.pbCInit
  let u := config.explorationWeight * pbC * child.prior * Float.sqrt parentVisits / (1.0 + childVisits)

  q + u

/-- Select best child by PUCT score -/
def selectChild (node : Node) (config : MCTSConfig) : Option (Nat × Node) := Id.run do
  if node.children.isEmpty then return none

  let mut bestIdx := 0
  let mut bestScore := floatNegInf
  for i in [:node.children.size] do
    if h : i < node.children.size then
      let child := node.children[i]
      let score := puctScore node child config
      if score > bestScore then
        bestScore := score
        bestIdx := i

  if h : bestIdx < node.children.size then
    return some (bestIdx, node.children[bestIdx])
  return none

/-- Calculate minimum visits needed to continue (progressive sampling) -/
def progressiveThreshold (node : Node) : Nat :=
  let totalChildren := node.children.size
  if totalChildren == 0 then 0
  else
    -- After N visits, can expand up to log(N) children
    let threshold := (Float.log (node.visitCount.toFloat + 1.0)).toUInt64.toNat
    Nat.min threshold totalChildren

/-- Should we expand another child? -/
def shouldExpand (node : Node) : Bool :=
  let expandedCount := node.children.filter (fun c => c.visitCount > 0) |>.size
  expandedCount < progressiveThreshold node

/-- Compute value for AND nodes (minimum of children - all must succeed) -/
def andNodeValue (children : Array Node) : Float :=
  if children.isEmpty then 0.0
  else
    children.foldl (init := floatInf) fun minVal child =>
      floatMin minVal (if child.visitCount > 0 then child.value / child.visitCount.toFloat else 0.0)

/-- Compute value for OR nodes (maximum of children - need one to succeed) -/
def orNodeValue (children : Array Node) : Float :=
  if children.isEmpty then 0.0
  else
    children.foldl (init := floatNegInf) fun maxVal child =>
      floatMax maxVal (if child.visitCount > 0 then child.value / child.visitCount.toFloat else 0.0)

/-- Update node with backpropagated value -/
def updateNode (node : Node) (value : Float) : Node :=
  { node with
    value := node.value + value
    visitCount := node.visitCount + 1
  }

/-- Create a child node from an action and policy prior -/
def createChild (action : Action) (state : ProofState) (prior : Float) : Node :=
  let player := if state.goals.size > 1 then .AND else .OR
  { action := some action
  , state := state
  , player := player
  , prior := prior
  }

/-- Result of a single MCTS simulation -/
structure SimulationResult where
  value : Float      -- Value to backpropagate
  path : Array Nat   -- Indices of selected children
  terminal : Bool    -- Was a terminal state reached?
  deriving Repr, Inhabited

/-- MCTS search result -/
structure SearchResult where
  bestAction : Option Action
  rootValue : Float
  visitCounts : Array (Action × Nat)
  deriving Repr, Inhabited

/-- Get the most visited action from root -/
def getMostVisitedAction (root : Node) : Option Action := Id.run do
  if root.children.isEmpty then return none

  let mut bestIdx := 0
  let mut bestVisits := 0
  for i in [:root.children.size] do
    if h : i < root.children.size then
      let child := root.children[i]
      if child.visitCount > bestVisits then
        bestVisits := child.visitCount
        bestIdx := i

  if h : bestIdx < root.children.size then
    return root.children[bestIdx].action
  return none

/-- Get action probabilities (normalized visit counts) -/
def getActionProbs (root : Node) : Array (Action × Float) := Id.run do
  let totalVisits := root.children.foldl (init := 0) fun acc c => acc + c.visitCount
  if totalVisits == 0 then return #[]

  let mut probs := Array.mkEmpty root.children.size
  for child in root.children do
    if let some action := child.action then
      let prob := child.visitCount.toFloat / totalVisits.toFloat
      probs := probs.push (action, prob)
  return probs

/-- Create root node from initial proof state -/
def createRoot (state : ProofState) : Node :=
  { action := none
  , state := state
  , player := .OR
  , prior := 1.0
  }

/-- Empty search result -/
def SearchResult.empty : SearchResult :=
  { bestAction := none
  , rootValue := 0.0
  , visitCounts := #[]
  }

end mcts
