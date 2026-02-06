import Std
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Model.Flowfusion
import Tyr.Manifolds.Basic

/-!
  BranchingFlows-style abstractions (Lean port, minimal core).

  This module focuses on the combinatorial/structural pieces:
  - coalescence policies
  - coalescent forest sampling
  - branching state bookkeeping
  - simple RNG utilities

  It intentionally leaves base-process specifics (bridge/step/loss) as
  user-supplied functions. This keeps the API usable without a full
  Flowfusion port.
-/

namespace torch.branching

open Tyr.AD

/-! ## RNG utilities (deterministic LCG) -/

structure Rng where
  state : UInt64
  deriving Repr

private def lcgNext (s : UInt64) : UInt64 :=
  s * 6364136223846793005 + 1442695040888963407

def Rng.next (r : Rng) : Rng := { state := lcgNext r.state }

def randUInt64 (r : Rng) : UInt64 × Rng :=
  let s := lcgNext r.state
  (s, { state := s })

private def u64Denom : Float :=
  Float.ofNat (Nat.pow 2 64)

def randFloat (r : Rng) : Float × Rng :=
  let (u, r') := randUInt64 r
  let f := (Float.ofNat u.toNat) / u64Denom
  (f, r')

def randNat (r : Rng) (n : Nat) : Nat × Rng :=
  if n = 0 then
    (0, r)
  else
    let (u, r') := randUInt64 r
    let n64 := UInt64.ofNat n
    let v := (u % n64).toNat
    (v, r')

def randBool (r : Rng) : Bool × Rng :=
  let (u, r') := randUInt64 r
  (u % 2 = 0, r')

def randBernoulli (r : Rng) (p : Float) : Bool × Rng :=
  let (u, r') := randFloat r
  (u < p, r')

def randBinomial (r : Rng) (n : Nat) (p : Float) : Nat × Rng := Id.run do
  let mut count := 0
  let mut rng := r
  for _ in [:n] do
    let (b, rng') := randBernoulli rng p
    rng := rng'
    if b then
      count := count + 1
  return (count, rng)

def randPoisson (r : Rng) (lambda : Float) : Nat × Rng :=
  if lambda <= 0 then
    (0, r)
  else
    Id.run do
    let L := Float.exp (-lambda)
    let mut k := 0
    let mut p := 1.0
    let mut rng := r
    while p > L do
      k := k + 1
      let (u, rng') := randFloat rng
      rng := rng'
      p := p * u
    return (Nat.pred k, rng)

def randExponential (r : Rng) : Float × Rng :=
  let (u, r') := randFloat r
  let u := if u <= 1e-12 then 1e-12 else u
  (-Float.log u, r')

/-! ## Time distributions -/

structure TimeDist where
  cdf : Float → Float
  pdf : Float → Float
  quantile : Float → Float

/-! ## Flow tree node -/

structure FlowNode (α : Type) where
  time : Float
  data : α
  weight : Nat
  group : Int
  branchable : Bool
  del : Bool
  id : Int
  flowable : Bool
  children : Array (FlowNode α) := #[]
  deriving Repr, Inhabited

namespace FlowNode

def leaf (time : Float) (data : α) (group : Int) (branchable del flowable : Bool) (id : Int) : FlowNode α :=
  { time, data, weight := 1, group, branchable, del, id, flowable, children := #[] }

def merge (time : Float) (data : α) (left right : FlowNode α) : FlowNode α :=
  { time,
    data,
    weight := left.weight + right.weight,
    group := left.group,
    branchable := true,
    del := false,
    id := 0,
    flowable := true,
    children := #[left, right] }

end FlowNode

/-! ## Branching state (single sequence) -/

structure BranchingState (α : Type) where
  state : Array α
  groupings : Array Int
  del : Array Bool
  ids : Array Int
  branchmask : Array Bool
  flowmask : Array Bool
  padmask : Array Bool
  deriving Repr, Inhabited

namespace BranchingState

def length (x : BranchingState α) : Nat := x.state.size

def mkDefault (state : Array α) (groupings : Array Int) : BranchingState α :=
  let n := state.size
  { state,
    groupings,
    del := Array.replicate n false,
    ids := (Array.range n).map (fun i => Int.ofNat (i + 1)),
    branchmask := Array.replicate n true,
    flowmask := Array.replicate n true,
    padmask := Array.replicate n true }

end BranchingState

/-! ## Anchor merging -/

class AnchorMerge (α : Type) where
  merge : α → α → Nat → Nat → α

def canonicalAnchorMerge [AnchorMerge α] (a b : α) (w1 w2 : Nat) : α :=
  AnchorMerge.merge a b w1 w2

def selectAnchorMerge [AnchorMerge α] (a b : α) (w1 w2 : Nat) (rng : Rng) : α × Rng :=
  let total := w1 + w2
  if total = 0 then
    (canonicalAnchorMerge a b w1 w2, rng)
  else
    let (u, rng') := randFloat rng
    let p := w1.toFloat / total.toFloat
    if u < p then
      (canonicalAnchorMerge a b 1 0, rng')
    else
      (canonicalAnchorMerge b a 0 1, rng')

instance : AnchorMerge Float where
  merge a b w1 w2 :=
    let total := (w1 + w2).toFloat
    if total == 0.0 then a else (a * w1.toFloat + b * w2.toFloat) / total

instance {s : Shape} : AnchorMerge (T s) where
  merge a b w1 w2 :=
    let total := (w1 + w2).toFloat
    if total == 0.0 then a
    else
      let wa := torch.mul_scalar a w1.toFloat
      let wb := torch.mul_scalar b w2.toFloat
      torch.div_scalar (torch.add wa wb) total

instance [AnchorMerge α] [AnchorMerge β] : AnchorMerge (α × β) where
  merge a b w1 w2 :=
    (AnchorMerge.merge a.1 b.1 w1 w2, AnchorMerge.merge a.2 b.2 w1 w2)

instance [AnchorMerge α] : AnchorMerge (torch.flowfusion.MaskedState α) where
  merge a b w1 w2 :=
    torch.flowfusion.maskLike (AnchorMerge.merge a.state b.state w1 w2) a

/-! ## Coalescence policies -/

abbrev GroupMins := Std.HashMap Int Nat

/-! ## Group-minimum helpers -/

inductive GroupMinsSpec where
  | none
  | uniform (min : Nat)
  | perGroup (mins : GroupMins)
  deriving Repr, Inhabited

private def defaultGroupMins (groupings : Array Int) (min : Nat) : GroupMins := Id.run do
  let mut mins : GroupMins := {}
  for g in groupings do
    match mins.get? g with
    | some _ => continue
    | none => mins := mins.insert g min
  mins

def resolveGroupMins (spec : GroupMinsSpec) (groupings : Array Int) : GroupMins :=
  match spec with
  | .none => defaultGroupMins groupings 1
  | .uniform min => defaultGroupMins groupings min
  | .perGroup mins => mins

def resolveGroupMinsBatch (default : GroupMinsSpec) (perItem : Array GroupMinsSpec)
    (groupings : Array (Array Int)) : Array GroupMins := Id.run do
  if perItem.isEmpty then
    return groupings.map (resolveGroupMins default)
  if perItem.size != groupings.size then
    return groupings.map (resolveGroupMins default)
  let mut out : Array GroupMins := #[]
  for i in [:groupings.size] do
    out := out.push (resolveGroupMins perItem[i]! groupings[i]!)
  out

def groupwiseMaxCoalescences (nodes : Array (FlowNode α)) : Nat := Id.run do
  let mut counts : Std.HashMap Int Nat := {}
  for n in nodes do
    if n.branchable then
      let c := counts.getD n.group 0
      counts := counts.insert n.group (c + 1)
  return counts.fold (init := 0) (fun acc _ c => acc + (Nat.pred c))

structure CoalescencePolicy (α : Type) where
  select : Array (FlowNode α) → Option GroupMins → Rng → Option (Nat × Nat) × Rng
  maxCoalescences : Array (FlowNode α) → Nat
  init : Array (FlowNode α) → Rng → Rng := fun _ r => r
  update : Array (FlowNode α) → Nat → Nat → Nat → Rng → Rng := fun _ _ _ _ r => r
  reorder : Array (FlowNode α) → Array (FlowNode α) := id
  shouldAppendOnSplit : Bool := false

section InhabitedOps

variable [Inhabited α]

def sequentialPairs (nodes : Array (FlowNode α)) : Array Nat := Id.run do
  let mut idx : Array Nat := #[]
  let n := nodes.size
  if n <= 1 then
    return idx
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      idx := idx.push i
  return idx

def sequentialUniformSelect (nodes : Array (FlowNode α)) (groupMins : Option GroupMins) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let n := nodes.size
  if n <= 1 then
    return (none, rng)
  let mut eligible : Array Nat := #[]
  let mut groupSizes : Std.HashMap Int Nat := {}
  match groupMins with
  | none =>
      -- Just collect eligible adjacent pairs
      for i in [:n-1] do
        let a := nodes[i]!
        let b := nodes[i+1]!
        if a.branchable && b.branchable && a.group == b.group then
          eligible := eligible.push i
  | some mins =>
      -- Count branchable sizes per group
      for node in nodes do
        if node.branchable then
          let c := groupSizes.getD node.group 0
          groupSizes := groupSizes.insert node.group (c + 1)
      for i in [:n-1] do
        let a := nodes[i]!
        let b := nodes[i+1]!
        if a.branchable && b.branchable && a.group == b.group then
          let minSize := mins.getD a.group 0
          let size := groupSizes.getD a.group 0
          if size > minSize then
            eligible := eligible.push i
  if eligible.isEmpty then
    return (none, rng)
  else
    let (k, rng') := randNat rng eligible.size
    let i := eligible[k]!
    return (some (i, i+1), rng')

def sequentialUniformPolicy (α : Type) [Inhabited α] : CoalescencePolicy α :=
  { select := sequentialUniformSelect,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

def sequentialUniformBlockMinSelect (blockMin : Nat) (nodes : Array (FlowNode α)) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let n := nodes.size
  if n <= 1 then
    return (none, rng)
  let mut blockSizes : Std.HashMap Nat Nat := {}
  let mut block := 0
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      let c := blockSizes.getD block 0
      blockSizes := blockSizes.insert block (c + 1)
    else
      block := block + 1
  let mut eligible : Array Nat := #[]
  let mut block2 := 0
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      let size := blockSizes.getD block2 0
      if size > (blockMin - 1) then
        eligible := eligible.push i
    else
      block2 := block2 + 1
  if eligible.isEmpty then
    return (none, rng)
  else
    let (k, rng') := randNat rng eligible.size
    let i := eligible[k]!
    return (some (i, i+1), rng')

def sequentialUniformBlockMinPolicy (α : Type) [Inhabited α] (blockMin : Nat) : CoalescencePolicy α :=
  { select := fun nodes _ rng => sequentialUniformBlockMinSelect blockMin nodes rng,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

private def weightedIndex (weights : Array Float) (rng : Rng) : Option Nat × Rng := Id.run do
  if weights.isEmpty then
    return (none, rng)
  let total := weights.foldl (init := 0.0) (fun acc w => acc + w)
  if total <= 0.0 then
    return (none, rng)
  let (u, rng') := randFloat rng
  let target := u * total
  let mut acc := 0.0
  for i in [:weights.size] do
    acc := acc + weights[i]!
    if acc >= target then
      return (some i, rng')
  return (some (weights.size - 1), rng')

def balancedSequentialSelect (alpha : Float) (nodes : Array (FlowNode α))
    (_groupMins : Option GroupMins) (rng : Rng)
    : Option (Nat × Nat) × Rng := Id.run do
  let alpha := if alpha < 0.0 then 0.0 else alpha
  let n := nodes.size
  if n <= 1 then
    return (none, rng)
  let mut eligible : Array Nat := #[]
  let mut weights : Array Float := #[]
  for i in [:n-1] do
    let a := nodes[i]!
    let b := nodes[i+1]!
    if a.branchable && b.branchable && a.group == b.group then
      eligible := eligible.push i
      let w := (a.weight + b.weight).toFloat
      weights := weights.push (Float.pow w (-alpha))
  let (k, rng') := weightedIndex weights rng
  match k with
  | none => return (none, rng')
  | some k =>
      let i := eligible[k]!
      return (some (i, i+1), rng')

def balancedSequentialPolicy (α : Type) [Inhabited α] (alpha : Float := 1.0) : CoalescencePolicy α :=
  { select := balancedSequentialSelect alpha,
    maxCoalescences := fun nodes => (sequentialPairs nodes).size }

/-! ## Forest sampling -/

def nextSplitTime (dist : TimeDist) (W : Nat) (t0 : Float) (rng : Rng) : Float × Rng :=
  let m := (W - 1).toFloat
  let S0 := 1.0 - dist.cdf t0
  if S0 <= 0.0 then
    (t0, rng)
  else
    let (e, rng') := randExponential rng
    let sStar := S0 * Float.exp (-(e / m))
    let p := 1.0 - sStar
    let t := dist.quantile p
    (max t0 (min t 1.0), rng')

partial def sampleSplitTimes (dist : TimeDist) (node : FlowNode α) (t0 : Float) (rng : Rng)
    : FlowNode α × Array Float × Rng := Id.run do
  if node.weight <= 1 then
    return (node, #[], rng)
  else
    let (t, rng') := nextSplitTime dist node.weight t0 rng
    let mut times : Array Float := #[t]
    let mut children : Array (FlowNode α) := #[]
    let mut rng'' := rng'
    for child in node.children do
      let (child', ctimes, rng''') := sampleSplitTimes dist child t rng''
      rng'' := rng'''
      children := children.push child'
      times := times ++ ctimes
    return ({ node with time := t, children := children }, times, rng'')

private def eraseIdx (arr : Array α) (idx : Nat) : Array α := Id.run do
  let mut out : Array α := #[]
  for i in [:arr.size] do
    if i != idx then
      out := out.push arr[i]!
  out

def sampleForest
    (elements : Array α)
    (groupings : Array Int)
    (branchable : Array Bool)
    (flowable : Array Bool)
    (deleted : Array Bool)
    (ids : Array Int)
    (branchTimeDist : TimeDist)
    (policy : CoalescencePolicy α := sequentialUniformPolicy α)
    (coalescenceFactor : Float := 1.0)
    (merger : α → α → Nat → Nat → α)
    (groupMins : Option GroupMins := none)
    (rng : Rng := { state := 0 })
    : Array (FlowNode α) × Array Float × Rng := Id.run do
  let n := elements.size
  let mut nodes : Array (FlowNode α) := #[]
  for i in [:n] do
    nodes := nodes.push (FlowNode.leaf 1.0 elements[i]! groupings[i]! branchable[i]! deleted[i]! flowable[i]! ids[i]!)
  let rng := policy.init nodes rng
  let maxMerges := policy.maxCoalescences nodes
  let (sampledMerges, rng) := randBinomial rng maxMerges coalescenceFactor
  let mut rng := rng
  let mut nodesAcc := nodes
  for _ in [:sampledMerges] do
    let (sel, rng') := policy.select nodesAcc groupMins rng
    rng := rng'
    match sel with
    | none => break
    | some (i, j) =>
        let i := if i < j then i else j
        let j := if i < j then j else i
        let left := nodesAcc[i]!
        let right := nodesAcc[j]!
        let mergedData := merger left.data right.data left.weight right.weight
        let merged := FlowNode.merge 0.0 mergedData left right
        -- replace i, remove j
        nodesAcc := nodesAcc.set! i merged
        nodesAcc := eraseIdx nodesAcc j
        rng := policy.update nodesAcc i j i rng
  nodesAcc := policy.reorder nodesAcc
  -- sample split times for each root
  let mut allTimes : Array Float := #[]
  let mut roots : Array (FlowNode α) := #[]
  for node in nodesAcc do
    let (node', times, rng') := sampleSplitTimes branchTimeDist node 0.0 rng
    rng := rng'
    roots := roots.push node'
    allTimes := allTimes ++ times
  (roots, allTimes, rng)

/-! ## Deletion insertion utilities -/

def groupCounts (groupings : Array Int) : Std.HashMap Int Nat := Id.run do
  let mut counts : Std.HashMap Int Nat := {}
  for g in groupings do
    let c := counts.getD g 0
    counts := counts.insert g (c + 1)
  counts

def uniformDelInsertions (x : BranchingState α) (delP : Float) (rng : Rng)
    : BranchingState α × Rng := Id.run do
  let n := x.state.size
  let mut rng := rng
  let mut newIndices : Array Nat := #[]
  let mut delFlags : Array Bool := #[]
  for i in [:n] do
    let eligible := x.flowmask[i]! && x.branchmask[i]!
    let (doDup, rng') := if eligible then randBernoulli rng delP else (false, rng)
    rng := rng'
    if doDup then
      newIndices := newIndices.push i
      newIndices := newIndices.push i
      let (chooseDup, rng'') := randBool rng
      rng := rng''
      -- exactly one of the two gets deletion
      delFlags := delFlags.push (!chooseDup)
      delFlags := delFlags.push chooseDup
    else
      newIndices := newIndices.push i
      delFlags := delFlags.push false
  let mut newState : Array α := #[]
  let mut groupings : Array Int := #[]
  let mut ids : Array Int := #[]
  let mut branchmask : Array Bool := #[]
  let mut flowmask : Array Bool := #[]
  let mut padmask : Array Bool := #[]
  for idx in newIndices do
    newState := newState.push x.state[idx]!
    groupings := groupings.push x.groupings[idx]!
    ids := ids.push x.ids[idx]!
    branchmask := branchmask.push x.branchmask[idx]!
    flowmask := flowmask.push x.flowmask[idx]!
    padmask := padmask.push x.padmask[idx]!
  ({ state := newState, groupings, del := delFlags, ids, branchmask, flowmask, padmask }, rng)

def fixedcountDelInsertions (x : BranchingState α) (numEvents : Nat) (rng : Rng)
    : BranchingState α × Rng := Id.run do
  let n := x.state.size
  if numEvents = 0 then
    return (x, rng)
  let eligible : Array Nat :=
    (Array.range n)
      |>.filter (fun i => x.flowmask[i]! && x.branchmask[i]!)
  if eligible.isEmpty then
    return (x, rng)
  let mut rng := rng
  let mut beforeFlags : Array (Array Bool) := Array.replicate n #[]
  let mut afterFlags : Array (Array Bool) := Array.replicate n #[]
  let mut origDel : Array Bool := Array.replicate n false
  for _ in [:numEvents] do
    let (k, rng') := randNat rng eligible.size
    rng := rng'
    let i := eligible[k]!
    let (before, rng'') := randBool rng
    rng := rng''
    let (useOrig, rng''') := randBool rng
    rng := rng'''
    if before then
      if useOrig && !(origDel[i]!) then
        origDel := origDel.set! i true
        beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push false)
      else
        beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push true)
    else
      if useOrig && !(origDel[i]!) then
        origDel := origDel.set! i true
        afterFlags := afterFlags.set! i (afterFlags[i]! |>.push false)
      else
        afterFlags := afterFlags.set! i (afterFlags[i]! |>.push true)
  let mut newState : Array α := #[]
  let mut groupings : Array Int := #[]
  let mut ids : Array Int := #[]
  let mut branchmask : Array Bool := #[]
  let mut flowmask : Array Bool := #[]
  let mut padmask : Array Bool := #[]
  let mut delFlags : Array Bool := #[]
  for i in [:n] do
    for flag in beforeFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
    newState := newState.push x.state[i]!
    groupings := groupings.push x.groupings[i]!
    ids := ids.push x.ids[i]!
    branchmask := branchmask.push x.branchmask[i]!
    flowmask := flowmask.push x.flowmask[i]!
    padmask := padmask.push x.padmask[i]!
    delFlags := delFlags.push origDel[i]!
    for flag in afterFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
  ({ state := newState, groupings, del := delFlags, ids, branchmask, flowmask, padmask }, rng)

def groupFixedcountDelInsertions (x : BranchingState α) (groupNumEvents : Std.HashMap Int Nat) (rng : Rng)
    : BranchingState α × Rng := Id.run do
  let n := x.state.size
  if groupNumEvents.isEmpty then
    return (x, rng)
  let mut rng := rng
  let mut beforeFlags : Array (Array Bool) := Array.replicate n #[]
  let mut afterFlags : Array (Array Bool) := Array.replicate n #[]
  let mut origDel : Array Bool := Array.replicate n false
  let mut actualEvents := 0
  for (g, numEvents) in groupNumEvents.toList do
    if numEvents = 0 then
      continue
    let eligible : Array Nat :=
      (Array.range n)
        |>.filter (fun i => x.flowmask[i]! && x.branchmask[i]! && x.groupings[i]! == g)
    if eligible.isEmpty then
      continue
    for _ in [:numEvents] do
      let (k, rng') := randNat rng eligible.size
      rng := rng'
      let i := eligible[k]!
      let (before, rng'') := randBool rng
      rng := rng''
      let (useOrig, rng''') := randBool rng
      rng := rng'''
      if before then
        if useOrig && !(origDel[i]!) then
          origDel := origDel.set! i true
          beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push false)
        else
          beforeFlags := beforeFlags.set! i (beforeFlags[i]! |>.push true)
      else
        if useOrig && !(origDel[i]!) then
          origDel := origDel.set! i true
          afterFlags := afterFlags.set! i (afterFlags[i]! |>.push false)
        else
          afterFlags := afterFlags.set! i (afterFlags[i]! |>.push true)
      actualEvents := actualEvents + 1
  if actualEvents = 0 then
    return (x, rng)
  let mut newState : Array α := #[]
  let mut groupings : Array Int := #[]
  let mut ids : Array Int := #[]
  let mut branchmask : Array Bool := #[]
  let mut flowmask : Array Bool := #[]
  let mut padmask : Array Bool := #[]
  let mut delFlags : Array Bool := #[]
  for i in [:n] do
    for flag in beforeFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
    newState := newState.push x.state[i]!
    groupings := groupings.push x.groupings[i]!
    ids := ids.push x.ids[i]!
    branchmask := branchmask.push x.branchmask[i]!
    flowmask := flowmask.push x.flowmask[i]!
    padmask := padmask.push x.padmask[i]!
    delFlags := delFlags.push origDel[i]!
    for flag in afterFlags[i]! do
      newState := newState.push x.state[i]!
      groupings := groupings.push x.groupings[i]!
      ids := ids.push x.ids[i]!
      branchmask := branchmask.push x.branchmask[i]!
      flowmask := flowmask.push x.flowmask[i]!
      padmask := padmask.push x.padmask[i]!
      delFlags := delFlags.push flag
  ({ state := newState, groupings, del := delFlags, ids, branchmask, flowmask, padmask }, rng)

/-! ## Bridge helpers (generic) -/

structure Segment (α : Type) where
  Xt : α
  t : Float
  anchor : α
  descendants : Nat
  del : Bool
  branchable : Bool
  flowable : Bool
  group : Int
  lastCoalescence : Float
  id : Int
  deriving Repr

structure CoalescentFlow (P α : Type) where
  base : P
  branchTime : TimeDist
  splitTransform : Float → Float
  policy : CoalescencePolicy α
  deletionTime : TimeDist

partial def treeBridge
    (bridge : P → α → α → Float → Float → α)
    (P : P)
    (node : FlowNode α)
    (x0 : α)
    (targetT currentT : Float)
    (deletionDist : TimeDist)
    (rng : Rng)
    : Array (Segment α) × Rng := Id.run do
  if !node.flowable then
    let seg : Segment α :=
      { Xt := node.data, t := targetT, anchor := node.data, descendants := node.weight,
        del := node.del, branchable := false, flowable := false, group := node.group,
        lastCoalescence := currentT, id := node.id }
    return (#[seg], rng)
  if node.time > targetT then
    -- deletion hazard
    let mut rng := rng
    let mut survive := true
    if node.del then
      let sCurr := max (1.0 - deletionDist.cdf currentT) 0.0
      let sTgt := max (1.0 - deletionDist.cdf targetT) 0.0
      let survRatio := if sCurr > 0.0 then sTgt / sCurr else 0.0
      let (u, rng') := randFloat rng
      rng := rng'
      survive := u >= (1.0 - survRatio)
    if survive then
      let Xt := bridge P x0 node.data currentT targetT
      let seg : Segment α :=
        { Xt, t := targetT, anchor := node.data, descendants := node.weight,
          del := node.del, branchable := node.branchable, flowable := true, group := node.group,
          lastCoalescence := currentT, id := node.id }
      return (#[seg], rng)
    else
      return (#[], rng)
  else
    let nextX := bridge P x0 node.data currentT node.time
    let mut out : Array (Segment α) := #[]
    let mut rng := rng
    for child in node.children do
      let (segs, rng') := treeBridge bridge P child nextX targetT node.time deletionDist rng
      rng := rng'
      out := out ++ segs
    return (out, rng)

def forestBridge
    (bridge : P → α → α → Float → Float → α)
    (P : P)
    (x0Sampler : FlowNode α → α)
    (x1 : Array α)
    (t : Float)
    (groups : Array Int)
    (branchable : Array Bool)
    (flowable : Array Bool)
    (deleted : Array Bool)
    (branchTime : TimeDist)
    (deletionTime : TimeDist)
    (policy : CoalescencePolicy α)
    (merger : α → α → Nat → Nat → α)
    (groupMins : Option GroupMins := none)
    (coalescenceFactor : Float := 1.0)
    (useBranchingTimeProb : Float := 0.0)
    (maxLen : Option Nat := none)
    (maxResamples : Nat := 8)
    (rng : Rng := { state := 0 })
    : Array (Segment α) × Float × Rng := Id.run do
  let mut rng := rng
  let mut forest : Array (FlowNode α) := #[]
  let mut coalTimes : Array Float := #[]
  let mut tUsed := t
  let mut accept := false
  let mut attempts := 0
  while !accept && attempts <= maxResamples do
    let (forest', coalTimes', rng') :=
      sampleForest x1 groups branchable flowable deleted ((Array.range x1.size).map (fun i => Int.ofNat (i + 1)))
        branchTime policy coalescenceFactor merger groupMins rng
    rng := rng'
    let mut t' := t
    if coalTimes'.size > 0 then
      let (u, rng'') := randFloat rng
      rng := rng''
      if u < useBranchingTimeProb then
        let (k, rng''') := randNat rng coalTimes'.size
        rng := rng'''
        t' := coalTimes'[k]!
    let segCount := forest'.size + (coalTimes'.filter (fun τ => τ <= t')).size
    let ok :=
      match maxLen with
      | none => true
      | some m => segCount <= m
    forest := forest'
    coalTimes := coalTimes'
    tUsed := t'
    if ok then
      accept := true
    else
      attempts := attempts + 1
  let mut out : Array (Segment α) := #[]
  for root in forest do
    let x0 := x0Sampler root
    let (segs, rng') := treeBridge bridge P root x0 tUsed 0.0 deletionTime rng
    rng := rng'
    out := out ++ segs
  (out, tUsed, rng)

structure BranchingBridgeResult (α : Type) where
  t : Array Float
  segments : Array (Array (Segment α))
  Xt : Array (BranchingState α)
  X1anchor : Array (Array α)
  descendants : Array (Array Nat)
  del : Array (Array Bool)
  splitsTarget : Array (Array Nat)
  prevCoalescence : Array (Array Float)
  deriving Repr

def branchingBridge
    (bridge : P → α → α → Float → Float → α)
    (P : P)
    (x0Sampler : FlowNode α → α)
    (x1s : Array (BranchingState α))
    (times : Array Float)
    (branchTime : TimeDist)
    (deletionTime : TimeDist)
    (policy : CoalescencePolicy α)
    (merger : α → α → Nat → Nat → α)
    (groupMins : Option GroupMins := none)
    (coalescenceFactor : Float := 1.0)
    (useBranchingTimeProb : Float := 0.0)
    (maxLen : Option Nat := none)
    (maxResamples : Nat := 8)
    (lengthMins : GroupMinsSpec := .none)
    (lengthMinsPerItem : Array GroupMinsSpec := #[])
    (deletionPad : Float := 0.0)
    (x1Modifier : BranchingState α → BranchingState α := id)
    (rng : Rng := { state := 0 })
    : BranchingBridgeResult α × Rng := Id.run do
  let mut rng := rng
  let groupings := x1s.map (fun x => x.groupings)
  let resolvedMins : Array GroupMins :=
    match groupMins with
    | some mins => Array.replicate x1s.size mins
    | none => resolveGroupMinsBatch lengthMins lengthMinsPerItem groupings
  let mut x1s := x1s
  if deletionPad > 0.0 then
    let mut padded : Array (BranchingState α) := #[]
    for i in [:x1s.size] do
      let x1 := x1s[i]!
      let counts := groupCounts x1.groupings
      let mins := resolvedMins[i]!
      let mut groupNumEvents : Std.HashMap Int Nat := {}
      for (g, count) in counts.toList do
        let minLen := mins.getD g 1
        let target := deletionPad * (Nat.max count minLen).toFloat
        let lam := max (target - count.toFloat) 0.0
        let (k, rng') := randPoisson rng lam
        rng := rng'
        if k > 0 then
          groupNumEvents := groupNumEvents.insert g k
      let (x1', rng') := groupFixedcountDelInsertions x1 groupNumEvents rng
      rng := rng'
      padded := padded.push (x1Modifier x1')
    x1s := padded
  let mut out : Array (Array (Segment α)) := #[]
  let mut usedTimes : Array Float := #[]
  let mut XtStates : Array (BranchingState α) := #[]
  let mut anchors : Array (Array α) := #[]
  let mut descendants : Array (Array Nat) := #[]
  let mut delFlags : Array (Array Bool) := #[]
  let mut splitsTargets : Array (Array Nat) := #[]
  let mut prevCoals : Array (Array Float) := #[]
  for i in [:x1s.size] do
    let x1 := x1s[i]!
    let t := times[i]!
    let (segs, tUsed, rng') :=
      forestBridge bridge P x0Sampler x1.state t x1.groupings x1.branchmask x1.flowmask x1.del
        branchTime deletionTime policy merger (some (resolvedMins[i]!)) coalescenceFactor
        useBranchingTimeProb maxLen maxResamples rng
    rng := rng'
    out := out.push segs
    usedTimes := usedTimes.push tUsed
    let stateArray := segs.map (fun s => s.Xt)
    let groupArray := segs.map (fun s => s.group)
    let delArray := segs.map (fun s => s.del)
    let idArray := segs.map (fun s => s.id)
    let branchArray := segs.map (fun s => s.branchable)
    let flowArray := segs.map (fun s => s.flowable)
    let padArray := Array.replicate segs.size true
    XtStates := XtStates.push
      { state := stateArray, groupings := groupArray, del := delArray, ids := idArray,
        branchmask := branchArray, flowmask := flowArray, padmask := padArray }
    anchors := anchors.push (segs.map (fun s => s.anchor))
    descendants := descendants.push (segs.map (fun s => s.descendants))
    delFlags := delFlags.push delArray
    splitsTargets := splitsTargets.push (segs.map (fun s => Nat.pred s.descendants))
    prevCoals := prevCoals.push (segs.map (fun s => s.lastCoalescence))
  ({ t := usedTimes, segments := out, Xt := XtStates, X1anchor := anchors,
     descendants := descendants, del := delFlags, splitsTarget := splitsTargets,
     prevCoalescence := prevCoals }, rng)

end InhabitedOps

end torch.branching
