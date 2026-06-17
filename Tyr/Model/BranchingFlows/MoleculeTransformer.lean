import Tyr.Model.BranchingFlows.MoleculeTrain

/-!
  A small trainable molecule transformer for BranchingFlows batches.

  The paper-scale QM9 model stacks many transformer layers with richer spatial
  features.  This module keeps the same training boundary: coordinates, labels,
  time, padding mask, and heads for coordinate endpoints, label logits, split
  logits, and deletion logits.
-/

namespace torch.branching

open torch
open torch.nn

structure MoleculeTransformerParams (vocab heads headDim mlp : UInt64) where
  coordW : T #[heads * headDim, 3]
  coordB : T #[heads * headDim]
  labelEmbed : T #[vocab, heads * headDim]
  timeW : T #[heads * headDim, 1]
  timeB : T #[heads * headDim]
  attnLnWeight : T #[heads * headDim]
  attnLnBias : T #[heads * headDim]
  qW : T #[heads * headDim, heads * headDim]
  qB : T #[heads * headDim]
  kW : T #[heads * headDim, heads * headDim]
  kB : T #[heads * headDim]
  vW : T #[heads * headDim, heads * headDim]
  vB : T #[heads * headDim]
  oW : T #[heads * headDim, heads * headDim]
  oB : T #[heads * headDim]
  pairDistSlope : T #[heads]
  mlpLnWeight : T #[heads * headDim]
  mlpLnBias : T #[heads * headDim]
  fc1W : T #[mlp, heads * headDim]
  fc1B : T #[mlp]
  fc2W : T #[heads * headDim, mlp]
  fc2B : T #[heads * headDim]
  coordHeadW : T #[3, heads * headDim]
  coordHeadB : T #[3]
  labelHeadW : T #[vocab, heads * headDim]
  labelHeadB : T #[vocab]
  splitHeadW : T #[1, heads * headDim]
  splitHeadB : T #[1]
  delHeadW : T #[1, heads * headDim]
  delHeadB : T #[1]
  deriving TensorStruct

namespace MoleculeTransformerParams

private def randnScaled (shape : Shape) (scale : Float) : IO (T shape) := do
  let x ← torch.randn shape
  pure (x * scale)

def init (vocab heads headDim mlp : UInt64) : IO (MoleculeTransformerParams vocab heads headDim mlp) := do
  let initScale := 0.05
  let coordW ← randnScaled #[heads * headDim, 3] initScale
  let labelEmbed ← randnScaled #[vocab, heads * headDim] initScale
  let timeW ← randnScaled #[heads * headDim, 1] initScale
  let qW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let kW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let vW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let oW ← randnScaled #[heads * headDim, heads * headDim] initScale
  let fc1W ← randnScaled #[mlp, heads * headDim] initScale
  let fc2W ← randnScaled #[heads * headDim, mlp] initScale
  let coordHeadW ← randnScaled #[3, heads * headDim] initScale
  let labelHeadW ← randnScaled #[vocab, heads * headDim] initScale
  let splitHeadW ← randnScaled #[1, heads * headDim] initScale
  let delHeadW ← randnScaled #[1, heads * headDim] initScale
  pure {
    coordW,
    coordB := torch.zeros #[heads * headDim],
    labelEmbed,
    timeW,
    timeB := torch.zeros #[heads * headDim],
    attnLnWeight := torch.ones #[heads * headDim],
    attnLnBias := torch.zeros #[heads * headDim],
    qW,
    qB := torch.zeros #[heads * headDim],
    kW,
    kB := torch.zeros #[heads * headDim],
    vW,
    vB := torch.zeros #[heads * headDim],
    oW,
    oB := torch.zeros #[heads * headDim],
    pairDistSlope := torch.full #[heads] (-4.0),
    mlpLnWeight := torch.ones #[heads * headDim],
    mlpLnBias := torch.zeros #[heads * headDim],
    fc1W,
    fc1B := torch.zeros #[mlp],
    fc2W,
    fc2B := torch.zeros #[heads * headDim],
    coordHeadW,
    coordHeadB := torch.zeros #[3],
    labelHeadW,
    labelHeadB := torch.zeros #[vocab],
    splitHeadW,
    splitHeadB := torch.zeros #[1],
    delHeadW,
    delHeadB := torch.zeros #[1]
  }

private def pairwiseDistanceSq {batch maxLen : UInt64}
    (coord : T #[batch, maxLen, 3]) : T #[batch, maxLen, maxLen] :=
  let ci0 : T #[batch, maxLen, 1, 3] := nn.unsqueeze coord 2
  let cj0 : T #[batch, 1, maxLen, 3] := nn.unsqueeze coord 1
  let ci : T #[batch, maxLen, maxLen, 3] := nn.expand ci0 #[batch, maxLen, maxLen, 3]
  let cj : T #[batch, maxLen, maxLen, 3] := nn.expand cj0 #[batch, maxLen, maxLen, 3]
  let diff := ci - cj
  nn.sumDim (diff * diff) 3 false

private def spatialAttention {batch maxLen vocab heads headDim mlp : UInt64}
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (coord : T #[batch, maxLen, 3])
    (padmask : T #[batch, maxLen])
    (h : T #[batch, maxLen, heads * headDim]) : T #[batch, maxLen, heads * headDim] :=
  let q0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h params.qW params.qB
  let k0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h params.kW params.kB
  let v0 : T #[batch, maxLen, heads * headDim] := torch.affine3d h params.vW params.vB
  let q : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape q0 #[batch, maxLen, heads, headDim])
  let k : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape k0 #[batch, maxLen, heads, headDim])
  let v : T #[batch, heads, maxLen, headDim] :=
    nn.transpose_for_attention (reshape v0 #[batch, maxLen, heads, headDim])
  let kt : T #[batch, heads, headDim, maxLen] := nn.transpose k 2 3
  let scores0 : T #[batch, heads, maxLen, maxLen] :=
    (nn.bmm4d q kt) / (Float.sqrt headDim.toFloat)
  let dist : T #[batch, maxLen, maxLen] := pairwiseDistanceSq coord
  let dist4 : T #[batch, heads, maxLen, maxLen] :=
    nn.expand (reshape dist #[batch, 1, maxLen, maxLen]) #[batch, heads, maxLen, maxLen]
  let slope : T #[heads] := nn.softplus params.pairDistSlope
  let slope4 : T #[batch, heads, maxLen, maxLen] :=
    nn.expand (reshape slope #[1, heads, 1, 1]) #[batch, heads, maxLen, maxLen]
  let scores1 := scores0 - (slope4 * dist4)
  let keyMask0 : T #[batch, 1, 1, maxLen] := nn.unsqueeze (nn.unsqueeze padmask 1) 1
  let keyMask : T #[batch, heads, maxLen, maxLen] :=
    nn.expand keyMask0 #[batch, heads, maxLen, maxLen]
  let invalid : T #[batch, heads, maxLen, maxLen] := torch.lt_scalar keyMask 0.5
  let scores := nn.masked_fill scores1 invalid (-1.0e9)
  let attn : T #[batch, heads, maxLen, maxLen] := nn.softmax_dim scores 3
  let ctx : T #[batch, heads, maxLen, headDim] := nn.bmm4d attn v
  let out4 : T #[batch, maxLen, heads, headDim] := nn.transpose_from_attention ctx
  let out3 : T #[batch, maxLen, heads * headDim] := reshape out4 #[batch, maxLen, heads * headDim]
  torch.affine3d out3 params.oW params.oB

def forward {batch maxLen vocab heads headDim mlp : UInt64}
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (coord : T #[batch, maxLen, 3])
    (label : T #[batch, maxLen])
    (t : T #[batch])
    (padmask : T #[batch, maxLen]) :
    IO (T #[batch, maxLen, 3] × T #[batch, maxLen, vocab] × T #[batch, maxLen] × T #[batch, maxLen]) := do
  let coordEmb : T #[batch, maxLen, heads * headDim] := torch.affine3d coord params.coordW params.coordB
  let labelEmb : T #[batch, maxLen, heads * headDim] :=
    nn.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := heads * headDim)
      label params.labelEmbed
  let time0 : T #[batch, 1, 1] := nn.unsqueeze (nn.unsqueeze t 1) 1
  let timeIn : T #[batch, maxLen, 1] := nn.expand time0 #[batch, maxLen, 1]
  let timeEmb : T #[batch, maxLen, heads * headDim] := torch.affine3d timeIn params.timeW params.timeB
  let h0 := coordEmb + labelEmb + timeEmb
  let h1n := nn.layer_norm h0 params.attnLnWeight params.attnLnBias 1.0e-5
  let attnOut := spatialAttention params coord padmask h1n
  let h1 := h0 + attnOut
  let h2n := nn.layer_norm h1 params.mlpLnWeight params.mlpLnBias 1.0e-5
  let mlp1 := nn.gelu (torch.affine3d h2n params.fc1W params.fc1B)
  let mlp2 := torch.affine3d mlp1 params.fc2W params.fc2B
  let h := h1 + mlp2
  let coordPred : T #[batch, maxLen, 3] := torch.affine3d h params.coordHeadW params.coordHeadB
  let labelLogits : T #[batch, maxLen, vocab] := torch.affine3d h params.labelHeadW params.labelHeadB
  let split3 : T #[batch, maxLen, 1] := torch.affine3d h params.splitHeadW params.splitHeadB
  let del3 : T #[batch, maxLen, 1] := torch.affine3d h params.delHeadW params.delHeadB
  let splitLogits : T #[batch, maxLen] := reshape split3 #[batch, maxLen]
  let delLogits : T #[batch, maxLen] := reshape del3 #[batch, maxLen]
  pure (coordPred, labelLogits, splitLogits, delLogits)

end MoleculeTransformerParams

def moleculeTransformerModel {maxLen vocab heads headDim mlp : UInt64} :
    BranchingMoleculeModel maxLen vocab (MoleculeTransformerParams vocab heads headDim mlp) :=
  { forward := fun {batch} params coord label t padmask =>
      MoleculeTransformerParams.forward (batch := batch) params coord label t padmask }

private def clampUpper? (cap? : Option Float) (x : Float) : Float :=
  match cap? with
  | some cap => min x cap
  | none => x

private def scalar3d {maxLen channels : UInt64}
    (x : T #[1, maxLen, channels]) (j k : Nat) : Float :=
  let row : T #[1, 1, channels] := data.slice x 1 j.toUInt64 1
  let cell : T #[1, 1, 1] := data.slice row 2 k.toUInt64 1
  nn.item (reshape cell #[])

private def scalar2d {maxLen : UInt64}
    (x : T #[1, maxLen]) (j : Nat) : Float :=
  let cell : T #[1, 1] := data.slice x 1 j.toUInt64 1
  nn.item (reshape cell #[])

private def packMoleculeStateForTransformer {maxLen vocab : UInt64}
    (padToken : Int64)
    (t : Float)
    (state : BranchingState MoleculeAtom) :
    IO (T #[1, maxLen, 3] × T #[1, maxLen] × T #[1] × T #[1, maxLen]) := do
  if state.state.size > maxLen.toNat then
    throw (IO.userError s!"molecule state length {state.state.size} exceeds transformer maxLen {maxLen}")
  let maxLenNat := maxLen.toNat
  let mut coordArr : Array Float := Array.replicate (maxLenNat * 3) 0.0
  let mut labelArr : Array Int64 := Array.replicate maxLenNat padToken
  let mut padArr : Array Int64 := Array.replicate maxLenNat 0
  for j in [:state.state.size] do
    let atom := state.state[j]!
    if atom.label >= vocab.toNat then
      throw (IO.userError s!"molecule label {atom.label} is outside transformer vocab {vocab}")
    let offset := j * 3
    coordArr := coordArr.set! offset atom.coord.x
    coordArr := coordArr.set! (offset + 1) atom.coord.y
    coordArr := coordArr.set! (offset + 2) atom.coord.z
    labelArr := labelArr.set! j (Int64.ofNat atom.label)
    padArr := padArr.set! j 1
  let coord := reshape (data.fromFloatArray coordArr) #[1, maxLen, 3]
  let label := reshape (data.fromInt64Array labelArr) #[1, maxLen]
  let time := reshape (data.fromFloatArray #[t]) #[1]
  let padmask := toFloat' (reshape (data.fromInt64Array padArr) #[1, maxLen])
  pure (coord, label, time, padmask)

private def tensorVec3 {maxLen : UInt64}
    (coordPred : T #[1, maxLen, 3]) (j : Nat) : Vec3 :=
  { x := scalar3d coordPred j 0,
    y := scalar3d coordPred j 1,
    z := scalar3d coordPred j 2 }

private def tensorLabelLogits {maxLen vocab : UInt64}
    (labelLogits : T #[1, maxLen, vocab]) (j : Nat) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for k in [:vocab.toNat] do
    out := out.push (scalar3d labelLogits j k)
  out

def moleculeTransformerPrediction {maxLen vocab heads headDim mlp : UInt64}
    (padToken : Int64)
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (t : Float)
    (state : BranchingState MoleculeAtom)
    (splitLogitCap? : Option Float := none) :
    IO MoleculeModelPrediction := do
  torch.autograd.no_grad do
    let (coord, label, time, padmask) ←
      packMoleculeStateForTransformer (maxLen := maxLen) (vocab := vocab) padToken t state
    let (coordPred, labelLogits, splitLogits, delLogits) ←
      MoleculeTransformerParams.forward (batch := 1) params coord label time padmask
    let n := state.state.size
    let canSplit := n < maxLen.toNat
    let coordTargets := (Array.range n).map (fun j => tensorVec3 coordPred j)
    let labelTargets := (Array.range n).map (fun j => tensorLabelLogits labelLogits j)
    let splitTargets := (Array.range n).map (fun j =>
      if canSplit then clampUpper? splitLogitCap? (scalar2d splitLogits j) else -100.0)
    let delTargets := (Array.range n).map (fun j => scalar2d delLogits j)
    pure {
      coordTargets
      labelLogits := labelTargets
      splitLogits := splitTargets
      delLogits := delTargets
    }

def moleculeTransformerIOModel {maxLen vocab heads headDim mlp : UInt64}
    (padToken : Int64)
    (params : MoleculeTransformerParams vocab heads headDim mlp)
    (splitLogitCap? : Option Float := none) :
    Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction :=
  fun t state =>
    moleculeTransformerPrediction (maxLen := maxLen) (vocab := vocab)
      (heads := heads) (headDim := headDim) (mlp := mlp)
      padToken params t state (splitLogitCap? := splitLogitCap?)

end torch.branching
