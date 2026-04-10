import Tyr.Mctx.Batched

namespace torch.mctx

private def updateAt (xs : Array α) (i : Nat) (v : α) : Array α :=
  if i < xs.size then xs.set! i v else xs

private def updateAt2D (xs : Array (Array α)) (i j : Nat) (v : α) : Array (Array α) :=
  if i < xs.size then
    let row := xs.getD i #[]
    if j < row.size then
      let row' := row.set! j v
      xs.set! i row'
    else
      xs
  else
    xs

private def intToNatNonneg (x : Int) : Nat :=
  if x < 0 then 0 else Int.toNat x

private def treeAt! [Inhabited S] [Inhabited E] (trees : Array (Tree S E)) (i : Nat) : Tree S E :=
  match trees[i]? with
  | some t => t
  | none => panic! s!"batched tree index out of bounds: {i}"

private def invalidRow
    (invalidActions : Option (Array (Array Bool)))
    (batchIdx : Nat)
    (numActions : Nat)
    : Array Bool :=
  match invalidActions with
  | none => Array.replicate numActions false
  | some rows => rows.getD batchIdx (Array.replicate numActions false)

private def invalidRowOpt
    (invalidActions : Option (Array (Array Bool)))
    (batchIdx : Nat)
    : Option (Array Bool) :=
  match invalidActions with
  | none => none
  | some rows => some (rows.getD batchIdx #[])

private def batchedRootAt [Inhabited S] (root : BatchedRootFnOutput S) (batchIdx : Nat) : RootFnOutput S := {
  priorLogits := root.priorLogits.getD batchIdx #[]
  value := root.value.getD batchIdx 0.0
  embedding := root.embedding.getD batchIdx default
}

private def expandWithStep [Inhabited S]
    (tree : Tree S E)
    (parentIndex : NodeIndex)
    (action : Action)
    (nextNodeIndex : NodeIndex)
    (step : RecurrentFnOutput)
    (nextEmbedding : S)
    : Tree S E :=
  let inBounds := nextNodeIndex < tree.nodeVisits.size
  let tree :=
    if inBounds then updateTreeNode tree nextNodeIndex step.priorLogits step.value nextEmbedding
    else tree
  let childIdx : Int := if inBounds then Int.ofNat nextNodeIndex else UNVISITED
  { tree with
    childrenIndex := updateAt2D tree.childrenIndex parentIndex action childIdx
    childrenRewards := updateAt2D tree.childrenRewards parentIndex action step.reward
    childrenDiscounts := updateAt2D tree.childrenDiscounts parentIndex action step.discount
    parents := updateAt tree.parents nextNodeIndex (Int.ofNat parentIndex)
    actionFromParent := updateAt tree.actionFromParent nextNodeIndex (Int.ofNat action)
  }

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

private def clamp01 (x : Float) : Float :=
  if x < 1e-6 then 1e-6 else if x > 1.0 - 1e-6 then 1.0 - 1e-6 else x

private def pseudoUniform01 (key : UInt64) (i : Nat) : Float :=
  let x := (key + UInt64.ofNat (i + 1) * 0x9e3779b97f4a7c15)
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  clamp01 (Float.ofNat mant / denom)

private def pseudoProbs (key : UInt64) (n : Nat) : Array Float :=
  let vals := (List.range n).toArray.map (fun i => pseudoUniform01 key i)
  let z := vals.foldl (init := 0.0) (· + ·)
  if z <= 0.0 then
    Array.replicate n (1.0 / Float.ofNat (Nat.max 1 n))
  else
    vals.map (fun v => v / z)

private def getLogitsFromProbs (probs : Array Float) : Array Float :=
  probs.map logSafe

private def addDirichletNoise
    (key : UInt64)
    (probs : Array Float)
    (dirichletFraction : Float)
    : Array Float :=
  let noise := pseudoProbs key probs.size
  (List.range probs.size).toArray.map fun i =>
    (1.0 - dirichletFraction) * probs.getD i 0.0 + dirichletFraction * noise.getD i 0.0

private def sampleGumbel (key : UInt64) (n : Nat) (scale : Float) : Array Float :=
  (List.range n).toArray.map fun i =>
    let u := pseudoUniform01 key i
    scale * (-(Float.log (-Float.log u)))

/-- Batched monadic search continuation from a pre-initialized tree array. -/
def searchBatchedWithTreesM
    [Monad m]
    [Inhabited S]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (trees : Array (Tree S E))
    (recurrentFn : BatchedRecurrentFnM m P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    : m (BatchedTree S E) := do
  let batchSize := trees.size
  let actionSelectionFn :=
    switchingActionSelectionWrapper rootActionSelectionFn interiorActionSelectionFn
  let depthCutoff :=
    match trees[0]? with
    | some t => maxDepth.getD t.numSimulations
    | none => maxDepth.getD numSimulations

  let mut trees := trees
  let mut sim := 0
  while sim < numSimulations do
    let mut parentIndices : Array Nat := Array.mkEmpty batchSize
    let mut actions : Array Action := Array.mkEmpty batchSize
    let mut nextNodeIndices : Array Nat := Array.mkEmpty batchSize
    let mut backupLeafIndices : Array Nat := Array.mkEmpty batchSize
    let mut parentEmbeddings : Array S := Array.mkEmpty batchSize

    for bi in [:batchSize] do
      let tree := treeAt! trees bi
      let simKey := rngKey + UInt64.ofNat ((sim + 1) * 1315423911 + bi)
      let (parentIndex, action) := simulate simKey tree actionSelectionFn depthCutoff
      let existing := (tree.childrenIndex.getD parentIndex #[]).getD action UNVISITED
      let nextNodeIndex :=
        if existing = UNVISITED then tree.nextNodeIndex else intToNatNonneg existing
      let inBounds := nextNodeIndex < tree.nodeVisits.size
      parentIndices := parentIndices.push parentIndex
      actions := actions.push action
      nextNodeIndices := nextNodeIndices.push nextNodeIndex
      backupLeafIndices := backupLeafIndices.push (if inBounds then nextNodeIndex else parentIndex)
      parentEmbeddings := parentEmbeddings.push (tree.embeddings.getD parentIndex default)

    let recKey := rngKey + UInt64.ofNat (sim + 1)
    let (stepBatch, nextEmbeddings) ← recurrentFn params recKey actions parentEmbeddings

    for bi in [:batchSize] do
      let tree := treeAt! trees bi
      let parentIndex := parentIndices.getD bi ROOT_INDEX
      let action := actions.getD bi 0
      let nextNodeIndex := nextNodeIndices.getD bi tree.nodeVisits.size
      let backupLeaf := backupLeafIndices.getD bi parentIndex
      let step : RecurrentFnOutput := {
        reward := stepBatch.reward.getD bi 0.0
        discount := stepBatch.discount.getD bi 0.0
        priorLogits := stepBatch.priorLogits.getD bi #[]
        value := stepBatch.value.getD bi 0.0
      }
      let nextEmbedding := nextEmbeddings.getD bi default
      let tree := expandWithStep tree parentIndex action nextNodeIndex step nextEmbedding
      let tree := backward tree backupLeaf
      trees := trees.set! bi tree

    sim := sim + 1

  pure { trees := trees }

/-- Batched monadic MCTS search. -/
def searchBatchedM
    [Monad m]
    [Inhabited S]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFnM m P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    (invalidActions : Option (Array (Array Bool)) := none)
    (extraData : Option (Array E) := none)
    : m (BatchedTree S E) := do
  let batchSize := root.value.size
  let depthCutoff := maxDepth.getD numSimulations

  let mut trees : Array (Tree S E) := Array.mkEmpty batchSize
  for bi in [:batchSize] do
    let rootRow := batchedRootAt root bi
    let invalid := invalidRow invalidActions bi rootRow.priorLogits.size
    let extra :=
      match extraData with
      | some rows => rows.getD bi default
      | none => default
    let tree := instantiateTreeFromRoot rootRow numSimulations invalid extra
    trees := trees.push tree

  searchBatchedWithTreesM
    params rngKey trees recurrentFn rootActionSelectionFn interiorActionSelectionFn
    numSimulations (some depthCutoff)

/-- Batched monadic MuZero policy. -/
def muzeroPolicyBatchedM
    [Monad m]
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFnM m P S)
    (numSimulations : Nat)
    (invalidActions : Option (Array (Array Bool)) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (_dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : m (BatchedPolicyOutput (BatchedTree S Unit)) := do
  let batchSize := root.value.size
  let noisyPrior := (List.range batchSize).toArray.map fun bi =>
    let row := root.priorLogits.getD bi #[]
    let probs := softmax row
    let noisyProbs := addDirichletNoise (rngKey + UInt64.ofNat bi) probs dirichletFraction
    let noisyLogits := getLogitsFromProbs noisyProbs
    maskInvalidActions noisyLogits (invalidRowOpt invalidActions bi)

  let root : BatchedRootFnOutput S := {
    priorLogits := noisyPrior
    value := root.value
    embedding := root.embedding
  }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let searchTree ← searchBatchedM
    params rngKey root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions

  let summary := searchTree.summary
  let actionWeights := summary.visitProbs
  let actions := (List.range batchSize).toArray.map fun bi =>
    let logits := applyTemperature (getLogitsFromProbs (actionWeights.getD bi #[])) temperature
    argmax logits

  pure {
    action := actions
    actionWeights := actionWeights
    searchTree := searchTree
  }

/-- Batched monadic AlphaZero-style policy with optional tree continuation. -/
def alphazeroPolicyBatchedM
    [Monad m]
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFnM m P S)
    (numSimulations : Nat)
    (searchTree : Option (BatchedTree S Unit) := none)
    (maxNodes : Option Nat := none)
    (invalidActions : Option (Array (Array Bool)) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (_dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : m (BatchedPolicyOutput (BatchedTree S Unit)) := do
  let batchSize := root.value.size
  let noisyPrior := (List.range batchSize).toArray.map fun bi =>
    let row := root.priorLogits.getD bi #[]
    let probs := softmax row
    let noisyProbs := addDirichletNoise (rngKey + UInt64.ofNat bi) probs dirichletFraction
    let noisyLogits := getLogitsFromProbs noisyProbs
    maskInvalidActions noisyLogits (invalidRowOpt invalidActions bi)

  let root : BatchedRootFnOutput S := {
    priorLogits := noisyPrior
    value := root.value
    embedding := root.embedding
  }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let capacity := maxNodes.getD (numSimulations + 1)
  let initialTrees : Array (Tree S Unit) :=
    match searchTree with
    | none =>
      (List.range batchSize).toArray.map fun bi =>
        let rootRow := batchedRootAt root bi
        let invalid := invalidRow invalidActions bi rootRow.priorLogits.size
        instantiateTreeFromRootWithCapacity rootRow capacity invalid ()
    | some bt =>
      (List.range batchSize).toArray.map fun bi =>
        let rootRow := batchedRootAt root bi
        let invalid := invalidRow invalidActions bi rootRow.priorLogits.size
        let fallback := instantiateTreeFromRootWithCapacity rootRow capacity invalid ()
        let existing := bt.trees.getD bi fallback
        updateTreeWithRoot existing rootRow invalid ()

  let searchTree ← searchBatchedWithTreesM
    params rngKey initialTrees recurrentFn rootFn interiorFn numSimulations maxDepth

  let summary := searchTree.summary
  let actionWeights := summary.visitProbs
  let actions := (List.range batchSize).toArray.map fun bi =>
    let logits := applyTemperature (getLogitsFromProbs (actionWeights.getD bi #[])) temperature
    argmax logits

  pure {
    action := actions
    actionWeights := actionWeights
    searchTree := searchTree
  }

/-- Batched monadic Gumbel MuZero policy. -/
def gumbelMuZeroPolicyBatchedM
    [Monad m]
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : BatchedRootFnOutput S)
    (recurrentFn : BatchedRecurrentFnM m P S)
    (numSimulations : Nat)
    (invalidActions : Option (Array (Array Bool)) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S GumbelMuZeroExtraData := qtransformCompletedByMixValue)
    (maxNumConsideredActions : Nat := 16)
    (gumbelScale : Float := 1.0)
    : m (BatchedPolicyOutput (BatchedTree S GumbelMuZeroExtraData)) := do
  let batchSize := root.value.size

  let maskedPrior := (List.range batchSize).toArray.map fun bi =>
    let row := root.priorLogits.getD bi #[]
    maskInvalidActions row (invalidRowOpt invalidActions bi)

  let root : BatchedRootFnOutput S := {
    priorLogits := maskedPrior
    value := root.value
    embedding := root.embedding
  }

  let gumbels := (List.range batchSize).toArray.map fun bi =>
    sampleGumbel (rngKey + UInt64.ofNat (bi + 17)) (maskedPrior.getD bi #[]).size gumbelScale
  let extras : Array GumbelMuZeroExtraData :=
    gumbels.map (fun g => { rootGumbel := g })

  let rootFn : RootActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex =>
    gumbelMuZeroRootActionSelection tree nodeIndex numSimulations maxNumConsideredActions qtransform
  let interiorFn : InteriorActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex _depth =>
    gumbelMuZeroInteriorActionSelection tree nodeIndex qtransform

  let searchTree ← searchBatchedM
    params rngKey root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions (extraData := some extras)

  let summary := searchTree.summary

  let actions := (List.range batchSize).toArray.map fun bi =>
    let tree := treeAt! searchTree.trees bi
    let visitCounts := summary.visitCounts.getD bi #[]
    let consideredVisit := visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc
    let completedQvalues := qtransform tree ROOT_INDEX
    let gumbel := gumbels.getD bi #[]
    let prior := maskedPrior.getD bi #[]
    let toArgmax := scoreConsidered consideredVisit gumbel prior completedQvalues visitCounts
    maskedArgmax toArgmax (invalidRowOpt invalidActions bi)

  let actionWeights := (List.range batchSize).toArray.map fun bi =>
    let tree := treeAt! searchTree.trees bi
    let completedQvalues := qtransform tree ROOT_INDEX
    let logits := addArrays (maskedPrior.getD bi #[]) completedQvalues
    softmax (maskInvalidActions logits (invalidRowOpt invalidActions bi))

  pure {
    action := actions
    actionWeights := actionWeights
    searchTree := searchTree
  }

end torch.mctx
