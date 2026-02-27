import Tyr.Mctx.Base
import Tyr.Mctx.Math

namespace torch.mctx

/-- Search tree state (unbatched Lean port of mctx.Tree). -/
structure Tree (S E : Type) where
  nodeVisits : Array Nat
  rawValues : Array Float
  nodeValues : Array Float
  parents : Array Int
  actionFromParent : Array Int
  childrenIndex : Array (Array Int)
  childrenPriorLogits : Array (Array Float)
  childrenVisits : Array (Array Nat)
  childrenRewards : Array (Array Float)
  childrenDiscounts : Array (Array Float)
  childrenValues : Array (Array Float)
  embeddings : Array S
  rootInvalidActions : Array Bool
  extraData : E
  deriving Repr

instance [Inhabited S] [Inhabited E] : Inhabited (Tree S E) where
  default := {
    nodeVisits := #[]
    rawValues := #[]
    nodeValues := #[]
    parents := #[]
    actionFromParent := #[]
    childrenIndex := #[]
    childrenPriorLogits := #[]
    childrenVisits := #[]
    childrenRewards := #[]
    childrenDiscounts := #[]
    childrenValues := #[]
    embeddings := #[]
    rootInvalidActions := #[]
    extraData := default
  }

/-- Root node index. -/
def ROOT_INDEX : Nat := 0

/-- Sentinel for missing parent. -/
def NO_PARENT : Int := -1

/-- Sentinel for unexpanded edge. -/
def UNVISITED : Int := -1

/-- Number of action branches per node. -/
def Tree.numActions (tree : Tree S E) : Nat :=
  (tree.childrenIndex.getD ROOT_INDEX #[]).size

/-- Number of simulations represented in this tree capacity. -/
def Tree.numSimulations (tree : Tree S E) : Nat :=
  tree.nodeVisits.size - 1

/-- Q-values `r + gamma * v` for all actions from `nodeIndex`. -/
def Tree.qvalues (tree : Tree S E) (nodeIndex : Nat) : Array Float :=
  let rewards := tree.childrenRewards.getD nodeIndex #[]
  let discounts := tree.childrenDiscounts.getD nodeIndex #[]
  let values := tree.childrenValues.getD nodeIndex #[]
  (List.range rewards.size).toArray.map fun a =>
    rewards.getD a 0.0 + discounts.getD a 0.0 * values.getD a 0.0

private def visitProbsFromCounts (counts : Array Nat) (numActions : Nat) : Array Float :=
  let total : Nat := counts.foldl (init := 0) (· + ·)
  if total = 0 then
    if numActions = 0 then #[] else Array.replicate numActions (1.0 / Float.ofNat numActions)
  else
    counts.map (fun c => Float.ofNat c / Float.ofNat total)

/-- Root summary statistics used by policies. -/
def Tree.summary (tree : Tree S E) : SearchSummary :=
  let value := tree.nodeValues.getD ROOT_INDEX 0.0
  let visitCounts := tree.childrenVisits.getD ROOT_INDEX #[]
  let qvalues := tree.qvalues ROOT_INDEX
  let visitProbs := visitProbsFromCounts visitCounts tree.numActions
  { visitCounts := visitCounts
  , visitProbs := visitProbs
  , value := value
  , qvalues := qvalues
  }

end torch.mctx
