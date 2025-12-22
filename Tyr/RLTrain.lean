/-
  RL Training

  Reinforcement learning training loop for NanoProof.
  Combines MCTS self-play with supervised fine-tuning.
-/
import Tyr.MCTS
import Tyr.Prover
import Tyr.Train

namespace rltrain

open mcts prover

/-- Training transition: (state, action, value) -/
structure Transition where
  state : ProofState
  action : Action
  targetPolicy : Array Float  -- MCTS visit distribution
  targetValue : Float         -- Outcome (1 = solved, 0 = failed)
  deriving Repr, Inhabited

/-- Replay buffer for RL training -/
structure ReplayBuffer where
  transitions : Array Transition
  maxSize : Nat := 100000
  deriving Repr, Inhabited

/-- Create empty replay buffer -/
def ReplayBuffer.empty (maxSize : Nat := 100000) : ReplayBuffer :=
  { transitions := #[], maxSize := maxSize }

/-- Add transitions to buffer -/
def ReplayBuffer.add (buffer : ReplayBuffer) (ts : Array Transition) : ReplayBuffer :=
  let newTransitions := buffer.transitions ++ ts
  -- Keep only the most recent maxSize transitions
  let trimmed := if newTransitions.size > buffer.maxSize then
    newTransitions.extract (newTransitions.size - buffer.maxSize) newTransitions.size
  else
    newTransitions
  { buffer with transitions := trimmed }

/-- Sample a batch from buffer (sequential for now, TODO: add proper random sampling) -/
def ReplayBuffer.sample (buffer : ReplayBuffer) (batchSize : Nat) (offset : Nat := 0)
    : Array Transition := Id.run do
  if buffer.transitions.isEmpty then return #[]

  let mut batch := Array.mkEmpty batchSize
  for i in [:batchSize] do
    let idx := (offset + i) % buffer.transitions.size
    if h : idx < buffer.transitions.size then
      batch := batch.push buffer.transitions[idx]

  return batch

/-- RL training configuration -/
structure RLConfig where
  -- Base training config
  batchSize : UInt64 := 64
  learningRate : Float := 0.0001
  weightDecay : Float := 0.01

  -- RL-specific
  fractionSFT : Float := 0.1       -- Fraction of SFT data to mix in
  numSelfPlayGames : Nat := 100    -- Games per training iteration
  mctsConfig : MCTSConfig := {}
  searchConfig : SearchConfig := {}

  -- Buffer
  bufferSize : Nat := 100000
  minBufferSize : Nat := 1000      -- Min transitions before training
  deriving Repr, Inhabited

/-- RL training state -/
structure RLTrainState where
  buffer : ReplayBuffer
  step : Nat := 0
  gamesPlayed : Nat := 0
  gamesSolved : Nat := 0
  totalLoss : Float := 0.0
  deriving Repr, Inhabited

/-- Create initial RL training state -/
def RLTrainState.init (config : RLConfig) : RLTrainState :=
  { buffer := ReplayBuffer.empty config.bufferSize
  , step := 0
  , gamesPlayed := 0
  , gamesSolved := 0
  , totalLoss := 0.0
  }

/-- Extract transitions from a solved proof tree -/
def extractTransitions (root : Node) (solved : Bool) : Array Transition := Id.run do
  let outcome : Float := if solved then 1.0 else 0.0
  let mut result : Array Transition := #[]

  -- DFS through tree, collecting visited nodes
  let mut stack := #[(root, outcome)]

  while !stack.isEmpty do
    match stack.back? with
    | none => break
    | some (node, value) =>
      stack := stack.pop

      -- Only include nodes that were actually visited
      if node.visitCount > 0 && node.action.isSome then
        -- Compute target policy from visit counts
        let totalVisits : Nat := node.children.foldl (init := 0) fun acc c => acc + c.visitCount
        let targetPolicy := if totalVisits > 0 then
          node.children.map fun c => c.visitCount.toFloat / totalVisits.toFloat
        else
          node.children.map fun _ => 1.0 / node.children.size.toFloat

        let transition : Transition := {
          state := node.state
          action := node.action.get!
          targetPolicy := targetPolicy
          targetValue := value
        }
        result := result.push transition

      -- Add children to stack with discounted value
      for child in node.children do
        if child.visitCount > 0 then
          stack := stack.push (child, value * 0.99)

  return result

/-- Training result for one iteration -/
structure IterationResult where
  gamesPlayed : Nat
  gamesSolved : Nat
  transitionsCollected : Nat
  trainingLoss : Float
  deriving Repr, Inhabited

/-- Single game result -/
structure GameResult where
  solved : Bool
  transitions : Array Transition
  tactics : Array String
  deriving Repr, Inhabited

end rltrain
