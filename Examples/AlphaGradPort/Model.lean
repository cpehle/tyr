import Examples.AlphaGradPort.Tasks
import Tyr.Module
import Tyr.Mctx

namespace Examples.AlphaGradPort

open torch
open torch.mctx
open Tyr.AD
open Tyr.AD.Elim
open Tyr.AD.Sparse
open Tyr.AD.JaxprLike

/-!
  AlphaGrad-style graph observation export and transformer policy/value model.

  This is intentionally closer to the upstream AlphaGrad stack than the older
  handcrafted feature MLP:
  - observations are exported as a 5-channel graph tensor-like state
  - each vertex column becomes a token
  - policy/value are produced by a shared transformer encoder

  The search code still uses Tyr's `mctx` port; the missing piece here is
  making the state/model shape more faithful to AlphaGrad.
-/

abbrev GraphChannels : Nat := 5

private abbrev MODEL_DIM : UInt64 := 64
private abbrev FF_DIM : UInt64 := 128
private abbrev NUM_HEADS : UInt64 := 4
private abbrev HEAD_DIM : UInt64 := 16

private def modelDimNat : Nat := MODEL_DIM.toNat
private def ffDimNat : Nat := FF_DIM.toNat
private def numHeadsNat : Nat := NUM_HEADS.toNat
private def headDimNat : Nat := HEAD_DIM.toNat

private def channelOffset (rowDim : Nat) (channel : Nat) : Nat :=
  channel * rowDim

private def setTokenValue
    (token : Array Float)
    (rowDim : Nat)
    (channel row : Nat)
    (value : Float) :
    Array Float :=
  let idx := channelOffset rowDim channel + row
  if idx < token.size then
    token.set! idx value
  else
    token

private def sumAbsWeights (m : Tyr.AD.Sparse.SparseLinearMap) : Float :=
  m.entries.foldl (init := 0.0) fun acc e => acc + Float.abs e.weight

private def producerFamilyCode (producer? : Option VertexProducer) : Float :=
  match producer? with
  | none => 0.0
  | some producer =>
    match producer.typed.schema with
    | .dotGeneral
    | .mma
    | .outer => 4.0
    | .reduce
    | .reduceAccum
    | .cumsum
    | .cumprod => 3.0
    | .controlFlow => 5.0
    | .binary
    | .unary
    | .broadcast
    | .binaryBroadcast => 2.0
    | _ => 1.0

private def sparseEdgeCode (m : Tyr.AD.Sparse.SparseLinearMap) : Float :=
  match m.repr with
  | SparseMapTag.zero => 0.0
  | .identityLike
  | .identity _ => 1.0
  | .semantic tag =>
    match tag with
    | .dotGeneral _ => 4.0
    | .outer _ _ => 4.0
    | .reduce _ _ _ _
    | .reduceAccum _ _ _ _
    | .cumsum _ _ _
    | .cumprod _ _ _ => 3.0
    | .broadcast _ _ _
    | .binaryBroadcast _ _ _ _ => 2.0
    | _ => 1.5
  | .named _
  | .add _ _
  | .compose _ _
  | .placeholder => 1.0

private def graphRowDim (graph : ElimGraph) (numVertices : Nat) : Nat :=
  1 + graph.inputs.size + numVertices

def observationTokenDim (graph : ElimGraph) (numVertices : Nat) : Nat :=
  GraphChannels * graphRowDim graph numVertices

private def rowIndexOfVertex?
    (graph : ElimGraph)
    (numVertices : Nat)
    (vertex : VertexId1) :
    Option Nat :=
  match graph.inputs.findIdx? (· = vertex) with
  | some i => some (i + 1)
  | none =>
    if vertex > 0 && vertex <= numVertices then
      some (1 + graph.inputs.size + (vertex - 1))
    else
      none

private def eliminationOrderMap (trace : Array VertexId1) : Std.HashMap VertexId1 Nat := Id.run do
  let mut out : Std.HashMap VertexId1 Nat := {}
  for i in [:trace.size] do
    let vertex := trace.getD i 0
    out := out.insert vertex (i + 1)
  return out

private def observationRowDimFromTokenDim (tokenDim : Nat) : Nat :=
  tokenDim / GraphChannels

private def validTokenFromMetadata
    (isAction isUnavailable isOutput : Float) :
    Float :=
  if isOutput > 0.5 then
    1.0
  else if isAction > 0.5 then
    if isUnavailable > 0.5 then 0.0 else 1.0
  else
    1.0

def attentionMaskFromObservationFlat
    (flat : Array Float)
    (vertexDim tokenDim : Nat) :
    Array Float := Id.run do
  let rowDim := observationRowDimFromTokenDim tokenDim
  let mut mask : Array Float := Array.replicate vertexDim 1.0
  for col in [:vertexDim] do
    let tokenBase := col * tokenDim
    let isAction := flat.getD (tokenBase + channelOffset rowDim 0) 0.0
    let isUnavailable := flat.getD (tokenBase + channelOffset rowDim 1) 0.0
    let isOutput := flat.getD (tokenBase + channelOffset rowDim 2) 0.0
    mask := mask.set! col (validTokenFromMetadata isAction isUnavailable isOutput)
  return mask

/-- Export a state into a 5-channel AlphaGrad-style token observation. -/
def exportObservationFlat
    (envCfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Float := Id.run do
  let vertexDim := s.numVertices
  let rowDim := graphRowDim s.graph vertexDim
  let tokenDim := observationTokenDim s.graph vertexDim
  let invalid := invalidActionMask envCfg s
  let orderMap := eliminationOrderMap s.vertexTrace
  let mut flat : Array Float := Array.replicate (vertexDim * tokenDim) 0.0

  let isActionVertex : Std.HashSet VertexId1 :=
    s.actionVertices.foldl (init := ({} : Std.HashSet VertexId1)) fun acc v => acc.insert v
  let isOutputVertex : Std.HashSet VertexId1 :=
    s.graph.outputs.foldl (init := ({} : Std.HashSet VertexId1)) fun acc v => acc.insert v

  for col in [:vertexDim] do
    let vertex : VertexId1 := col + 1
    let tokenBase := col * tokenDim
    let token := flat.extract tokenBase (tokenBase + tokenDim)
    let mut token := token

    let isAction := isActionVertex.contains vertex
    let action? :=
      if isAction then
        match vertexToActionInSpace? s.actionVertices vertex with
        | .ok action => some action
        | .error _ => none
      else
        none
    let isUnavailable :=
      match action? with
      | some action =>
        s.isActionEliminated action || invalid.getD action true
      | none =>
        true

    token := setTokenValue token rowDim 0 0 (if isAction then 1.0 else 0.0)
    token := setTokenValue token rowDim 1 0 (if isUnavailable then 1.0 else 0.0)
    token := setTokenValue token rowDim 2 0 (if isOutputVertex.contains vertex then 1.0 else 0.0)
    token := setTokenValue token rowDim 3 0 (Float.ofNat (orderMap.getD vertex 0))
    token := setTokenValue token rowDim 4 0 (producerFamilyCode (producerInfo? s.graph vertex))

    match rowIndexOfVertex? s.graph vertexDim vertex with
    | some diagRow =>
      token := setTokenValue token rowDim 4 diagRow (producerFamilyCode (producerInfo? s.graph vertex))
    | none => pure ()

    for src in s.graph.inputs do
      match rowIndexOfVertex? s.graph vertexDim src, findEdge? s.graph src vertex with
      | some row, some edge =>
        token := setTokenValue token rowDim 0 row (sparseEdgeCode edge)
        token := setTokenValue token rowDim 1 row (Float.ofNat (edge.inDim?.getD 0))
        token := setTokenValue token rowDim 2 row (Float.ofNat (edge.outDim?.getD 0))
        token := setTokenValue token rowDim 3 row (Float.ofNat edge.entries.size)
        token := setTokenValue token rowDim 4 row (sumAbsWeights edge)
      | _, _ => pure ()

    for src in [1:vertexDim + 1] do
      match rowIndexOfVertex? s.graph vertexDim src, findEdge? s.graph src vertex with
      | some row, some edge =>
        token := setTokenValue token rowDim 0 row (sparseEdgeCode edge)
        token := setTokenValue token rowDim 1 row (Float.ofNat (edge.inDim?.getD 0))
        token := setTokenValue token rowDim 2 row (Float.ofNat (edge.outDim?.getD 0))
        token := setTokenValue token rowDim 3 row (Float.ofNat edge.entries.size)
        token := setTokenValue token rowDim 4 row (sumAbsWeights edge)
      | _, _ => pure ()

    for i in [:token.size] do
      let idx := tokenBase + i
      if idx < flat.size then
        flat := flat.set! idx (token.getD i 0.0)

  return flat

private def geluApprox (x : Float) : Float :=
  let piApprox : Float := 3.141592653589793
  let c : Float := Float.sqrt (2.0 / piApprox)
  let x3 := x * x * x
  0.5 * x * (1.0 + Float.tanh (c * (x + 0.044715 * x3)))

private def dot (xs ys : Array Float) : Float := Id.run do
  let n := Nat.min xs.size ys.size
  let mut acc := 0.0
  for i in [:n] do
    acc := acc + xs.getD i 0.0 * ys.getD i 0.0
  return acc

private def softmaxArray (xs : Array Float) : Array Float :=
  if xs.isEmpty then
    #[]
  else
    let m := maxD xs (xs.getD 0 0.0)
    let shifted := xs.map (fun x => Float.exp (x - m))
    let z := shifted.foldl (init := 0.0) (· + ·)
    if z <= 0.0 then
      Array.replicate xs.size (1.0 / Float.ofNat (Nat.max xs.size 1))
    else
      shifted.map (fun x => x / z)

private structure EvalLinear where
  inDim : Nat
  outDim : Nat
  weight : Array Float
  bias : Array Float
  deriving Repr

private def evalLinear (lin : EvalLinear) (x : Array Float) : Array Float := Id.run do
  let mut out : Array Float := Array.replicate lin.outDim 0.0
  for o in [:lin.outDim] do
    let mut acc := lin.bias.getD o 0.0
    for i in [:lin.inDim] do
      let w := lin.weight.getD (o * lin.inDim + i) 0.0
      acc := acc + w * x.getD i 0.0
    out := out.set! o acc
  return out

private def layerNormArray (weight bias x : Array Float) (eps : Float := 1e-5) : Array Float :=
  if x.isEmpty then
    #[]
  else
    let n := Float.ofNat x.size
    let mean := x.foldl (init := 0.0) (· + ·) / n
    let var :=
      x.foldl (init := 0.0) fun acc xi =>
        let d := xi - mean
        acc + d * d
    let invStd := 1.0 / Float.sqrt (var / n + eps)
    (Array.range x.size).map fun i =>
      let norm := (x.getD i 0.0 - mean) * invStd
      norm * weight.getD i 1.0 + bias.getD i 0.0

private structure EvalAttention where
  modelDim : Nat
  numHeads : Nat
  headDim : Nat
  qProj : EvalLinear
  kProj : EvalLinear
  vProj : EvalLinear
  oProj : EvalLinear
  deriving Repr

private def sliceHead (xs : Array Float) (head headDim : Nat) : Array Float :=
  xs.extract (head * headDim) ((head + 1) * headDim)

private def evalAttention
    (attn : EvalAttention)
    (tokens : Array (Array Float))
    (attnMask : Option (Array Float) := none) :
    Array (Array Float) := Id.run do
  let seq := tokens.size
  let mut qAll : Array (Array Float) := #[]
  let mut kAll : Array (Array Float) := #[]
  let mut vAll : Array (Array Float) := #[]
  for tok in tokens do
    qAll := qAll.push (evalLinear attn.qProj tok)
    kAll := kAll.push (evalLinear attn.kProj tok)
    vAll := vAll.push (evalLinear attn.vProj tok)

  let scale := Float.sqrt (Float.ofNat attn.headDim)
  let mut out : Array (Array Float) := #[]
  for i in [:seq] do
    let queryValid : Bool :=
      match attnMask with
      | some mask => mask.getD i 1.0 > 0.5
      | none => true
    let mut merged : Array Float := #[]
    for h in [:attn.numHeads] do
      let qi := sliceHead (qAll.getD i #[]) h attn.headDim
      let mut scores : Array Float := #[]
      for j in [:seq] do
        let kj := sliceHead (kAll.getD j #[]) h attn.headDim
        let keyValid : Bool :=
          match attnMask with
          | some mask => mask.getD j 1.0 > 0.5
          | none => true
        scores := scores.push (if keyValid then dot qi kj / scale else -1.0e30)
      let probs := softmaxArray scores
      let mut ctx : Array Float := Array.replicate attn.headDim 0.0
      for j in [:seq] do
        let vj := sliceHead (vAll.getD j #[]) h attn.headDim
        let p := probs.getD j 0.0
        for d in [:attn.headDim] do
          ctx := ctx.set! d (ctx.getD d 0.0 + p * vj.getD d 0.0)
      merged := merged ++ ctx
    let outTok :=
      if queryValid then
        evalLinear attn.oProj merged
      else
        Array.replicate attn.modelDim 0.0
    out := out.push outTok
  return out

private structure EvalMLP where
  fc1 : EvalLinear
  fc2 : EvalLinear
  deriving Repr

private def evalMLP (mlp : EvalMLP) (x : Array Float) : Array Float :=
  let h1 := evalLinear mlp.fc1 x
  let h1 := h1.map geluApprox
  evalLinear mlp.fc2 h1

private structure EvalBlock where
  ln1Weight : Array Float
  ln1Bias : Array Float
  attn : EvalAttention
  ln2Weight : Array Float
  ln2Bias : Array Float
  mlp : EvalMLP
  deriving Repr

private def addVec (a b : Array Float) : Array Float :=
  (Array.range (Nat.max a.size b.size)).map fun i => a.getD i 0.0 + b.getD i 0.0

private def evalBlock
    (blk : EvalBlock)
    (tokens : Array (Array Float))
    (attnMask : Option (Array Float) := none) :
    Array (Array Float) :=
  let h1 := tokens.map (layerNormArray blk.ln1Weight blk.ln1Bias ·)
  let attnOut := evalAttention blk.attn h1 attnMask
  let resid1 := (Array.range tokens.size).map fun i => addVec (tokens.getD i #[]) (attnOut.getD i #[])
  let h2 := resid1.map (layerNormArray blk.ln2Weight blk.ln2Bias ·)
  let mlpOut := h2.map (evalMLP blk.mlp ·)
  (Array.range resid1.size).map fun i => addVec (resid1.getD i #[]) (mlpOut.getD i #[])

structure EvalNet (vertexDim tokenDim : UInt64) where
  tokenProj : EvalLinear
  posEmb : Array Float
  block0 : EvalBlock
  block1 : EvalBlock
  policyHead : EvalLinear
  valueHead : EvalLinear
  deriving Repr

private def posSlice (posEmb : Array Float) (vertex modelDim : Nat) : Array Float :=
  posEmb.extract (vertex * modelDim) ((vertex + 1) * modelDim)

private def extractBiasOrZeros
    {n : UInt64}
    (b? : Option (T #[n])) :
    IO (Array Float) := do
  match b? with
  | some b => data.tensorToFloatArray' (nn.eraseShape b)
  | none => pure (Array.replicate n.toNat 0.0)

private def buildEvalLinear {inDim outDim : UInt64} (lin : Linear inDim outDim) : IO EvalLinear := do
  let weight ← data.tensorToFloatArray' (nn.eraseShape lin.weight)
  let bias ← extractBiasOrZeros lin.bias
  pure {
    inDim := inDim.toNat
    outDim := outDim.toNat
    weight := weight
    bias := bias
  }

private def buildEvalLayerNorm {dim : UInt64} (ln : LayerNorm dim) : IO (Array Float × Array Float) := do
  let weight ← data.tensorToFloatArray' (nn.eraseShape ln.weight)
  let bias ← data.tensorToFloatArray' (nn.eraseShape ln.bias)
  pure (weight, bias)

structure AlphaGradSelfAttention (modelDim headDim numHeads : UInt64) where
  qProj : Linear modelDim (numHeads * headDim)
  kProj : Linear modelDim (numHeads * headDim)
  vProj : Linear modelDim (numHeads * headDim)
  oProj : Linear (numHeads * headDim) modelDim
  deriving TensorStruct

namespace AlphaGradSelfAttention

def init (modelDim headDim numHeads : UInt64) : IO (AlphaGradSelfAttention modelDim headDim numHeads) := do
  pure {
    qProj := ← Linear.init modelDim (numHeads * headDim) true
    kProj := ← Linear.init modelDim (numHeads * headDim) true
    vProj := ← Linear.init modelDim (numHeads * headDim) true
    oProj := ← Linear.init (numHeads * headDim) modelDim true
  }

def forward {batch seq modelDim headDim numHeads : UInt64}
    (attn : AlphaGradSelfAttention modelDim headDim numHeads)
    (x : T #[batch, seq, modelDim])
    (attnMask : Option (T #[batch, seq]) := none) :
    T #[batch, seq, modelDim] :=
  let q0 : T #[batch, seq, numHeads * headDim] := Linear.forward3d attn.qProj x
  let k0 : T #[batch, seq, numHeads * headDim] := Linear.forward3d attn.kProj x
  let v0 : T #[batch, seq, numHeads * headDim] := Linear.forward3d attn.vProj x
  let q : T #[batch, seq, numHeads, headDim] := reshape q0 #[batch, seq, numHeads, headDim]
  let k : T #[batch, seq, numHeads, headDim] := reshape k0 #[batch, seq, numHeads, headDim]
  let v : T #[batch, seq, numHeads, headDim] := reshape v0 #[batch, seq, numHeads, headDim]
  let qh : T #[batch, numHeads, seq, headDim] := nn.transpose_for_attention q
  let kh : T #[batch, numHeads, seq, headDim] := nn.transpose_for_attention k
  let vh : T #[batch, numHeads, seq, headDim] := nn.transpose_for_attention v
  let attnOut : T #[batch, numHeads, seq, headDim] :=
    match attnMask with
    | some mask =>
      nn.scaledDotProductAttentionGQAMask qh kh vh mask 0.0 false true
    | none =>
      nn.scaledDotProductAttentionGQA qh kh vh 0.0 false true
  let out3 : T #[batch, seq, numHeads, headDim] := nn.transpose_from_attention attnOut
  let out2 : T #[batch, seq, numHeads * headDim] := reshape out3 #[batch, seq, numHeads * headDim]
  Linear.forward3d attn.oProj out2

end AlphaGradSelfAttention

structure AlphaGradFeedForward (modelDim ffDim : UInt64) where
  fc1 : Linear modelDim ffDim
  fc2 : Linear ffDim modelDim
  deriving TensorStruct

namespace AlphaGradFeedForward

def init (modelDim ffDim : UInt64) : IO (AlphaGradFeedForward modelDim ffDim) := do
  pure {
    fc1 := ← Linear.init modelDim ffDim true
    fc2 := ← Linear.init ffDim modelDim true
  }

def forward {batch seq modelDim ffDim : UInt64}
    (mlp : AlphaGradFeedForward modelDim ffDim)
    (x : T #[batch, seq, modelDim]) :
    T #[batch, seq, modelDim] :=
  let h1 : T #[batch, seq, ffDim] := Linear.forward3d mlp.fc1 x
  let h2 := nn.gelu h1
  Linear.forward3d mlp.fc2 h2

end AlphaGradFeedForward

structure AlphaGradBlock (modelDim ffDim headDim numHeads : UInt64) where
  ln1 : LayerNorm modelDim
  attn : AlphaGradSelfAttention modelDim headDim numHeads
  ln2 : LayerNorm modelDim
  mlp : AlphaGradFeedForward modelDim ffDim
  deriving TensorStruct

namespace AlphaGradBlock

def init (modelDim ffDim headDim numHeads : UInt64) : IO (AlphaGradBlock modelDim ffDim headDim numHeads) := do
  pure {
    ln1 := LayerNorm.init modelDim
    attn := ← AlphaGradSelfAttention.init modelDim headDim numHeads
    ln2 := LayerNorm.init modelDim
    mlp := ← AlphaGradFeedForward.init modelDim ffDim
  }

def forward {batch seq modelDim ffDim headDim numHeads : UInt64}
    (blk : AlphaGradBlock modelDim ffDim headDim numHeads)
    (x : T #[batch, seq, modelDim])
    (attnMask : Option (T #[batch, seq]) := none) :
    T #[batch, seq, modelDim] :=
  let h1 := LayerNorm.forward3d blk.ln1 x
  let x := x + AlphaGradSelfAttention.forward blk.attn h1 attnMask
  let h2 := LayerNorm.forward3d blk.ln2 x
  x + AlphaGradFeedForward.forward blk.mlp h2

end AlphaGradBlock

structure AlphaGradNet (vertexDim tokenDim : UInt64) where
  tokenProj : Linear tokenDim MODEL_DIM
  posEmb : T #[vertexDim, MODEL_DIM]
  block0 : AlphaGradBlock MODEL_DIM FF_DIM HEAD_DIM NUM_HEADS
  block1 : AlphaGradBlock MODEL_DIM FF_DIM HEAD_DIM NUM_HEADS
  policyHead : Linear MODEL_DIM 1
  valueHead : Linear MODEL_DIM 1
  deriving TensorStruct

namespace AlphaGradNet

def init (vertexDim tokenDim : UInt64) : IO (AlphaGradNet vertexDim tokenDim) := do
  let tokenProj ← Linear.init tokenDim MODEL_DIM true
  let posRaw ← randn #[vertexDim, MODEL_DIM]
  let posEmb := autograd.set_requires_grad (mul_scalar posRaw 0.02) true
  pure {
    tokenProj := tokenProj
    posEmb := posEmb
    block0 := ← AlphaGradBlock.init MODEL_DIM FF_DIM HEAD_DIM NUM_HEADS
    block1 := ← AlphaGradBlock.init MODEL_DIM FF_DIM HEAD_DIM NUM_HEADS
    policyHead := ← Linear.init MODEL_DIM 1 true
    valueHead := ← Linear.init MODEL_DIM 1 true
  }

def forward {batch vertexDim tokenDim : UInt64}
    (net : AlphaGradNet vertexDim tokenDim)
    (x : T #[batch, vertexDim, tokenDim])
    (attnMask : Option (T #[batch, vertexDim]) := none) :
    T #[batch, vertexDim] × T #[batch, 1] :=
  let h0 : T #[batch, vertexDim, MODEL_DIM] := Linear.forward3d net.tokenProj x
  let pos : T #[batch, vertexDim, MODEL_DIM] :=
    nn.expand (reshape net.posEmb #[1, vertexDim, MODEL_DIM]) #[batch, vertexDim, MODEL_DIM]
  let h := h0 + pos
  let h := AlphaGradBlock.forward net.block0 h attnMask
  let h := AlphaGradBlock.forward net.block1 h attnMask
  let policy3 : T #[batch, vertexDim, 1] := Linear.forward3d net.policyHead h
  let policy2 : T #[batch, vertexDim] := reshape policy3 #[batch, vertexDim]
  let pooled : T #[batch, MODEL_DIM] := nn.meanDim h 1 false
  let value : T #[batch, 1] := Linear.forward2d net.valueHead pooled
  (policy2, value)

end AlphaGradNet

def buildEvalNet
    {vertexDim tokenDim : UInt64}
    (net : AlphaGradNet vertexDim tokenDim) :
    IO (EvalNet vertexDim tokenDim) := do
  let tokenProj ← buildEvalLinear net.tokenProj
  let posEmb ← data.tensorToFloatArray' (nn.eraseShape net.posEmb)
  let buildBlock (blk : AlphaGradBlock MODEL_DIM FF_DIM HEAD_DIM NUM_HEADS) : IO EvalBlock := do
    let (ln1Weight, ln1Bias) ← buildEvalLayerNorm blk.ln1
    let qProj ← buildEvalLinear blk.attn.qProj
    let kProj ← buildEvalLinear blk.attn.kProj
    let vProj ← buildEvalLinear blk.attn.vProj
    let oProj ← buildEvalLinear blk.attn.oProj
    let (ln2Weight, ln2Bias) ← buildEvalLayerNorm blk.ln2
    let fc1 ← buildEvalLinear blk.mlp.fc1
    let fc2 ← buildEvalLinear blk.mlp.fc2
    pure {
      ln1Weight := ln1Weight
      ln1Bias := ln1Bias
      attn := {
        modelDim := modelDimNat
        numHeads := numHeadsNat
        headDim := headDimNat
        qProj := qProj
        kProj := kProj
        vProj := vProj
        oProj := oProj
      }
      ln2Weight := ln2Weight
      ln2Bias := ln2Bias
      mlp := { fc1 := fc1, fc2 := fc2 }
    }
  let block0 ← buildBlock net.block0
  let block1 ← buildBlock net.block1
  let policyHead ← buildEvalLinear net.policyHead
  let valueHead ← buildEvalLinear net.valueHead
  pure {
    tokenProj := tokenProj
    posEmb := posEmb
    block0 := block0
    block1 := block1
    policyHead := policyHead
    valueHead := valueHead
  }

def evalStatePolicyValue
    {vertexDim tokenDim : UInt64}
    (net : EvalNet vertexDim tokenDim)
    (envCfg : AlphaGradMctxConfig)
    (s : AlphaGradState) :
    Array Float × Float := Id.run do
  let flat := exportObservationFlat envCfg s
  let vertexNat := vertexDim.toNat
  let tokenNat := tokenDim.toNat
  let attnMask := attentionMaskFromObservationFlat flat vertexNat tokenNat
  let mut rawTokens : Array (Array Float) := #[]
  for i in [:vertexNat] do
    rawTokens := rawTokens.push (flat.extract (i * tokenNat) ((i + 1) * tokenNat))

  let tokens := (Array.range vertexNat).map fun i =>
    let base := evalLinear net.tokenProj (rawTokens.getD i #[])
    addVec base (posSlice net.posEmb i modelDimNat)
  let tokens := evalBlock net.block0 tokens (some attnMask)
  let tokens := evalBlock net.block1 tokens (some attnMask)

  let vertexLogits := tokens.map fun tok => (evalLinear net.policyHead tok).getD 0 0.0
  let pooled :=
    if tokens.isEmpty then
      Array.replicate modelDimNat 0.0
    else
      let denom := Float.ofNat tokens.size
      (Array.range modelDimNat).map fun i =>
        tokens.foldl (init := 0.0) (fun acc tok => acc + tok.getD i 0.0) / denom
  let value := (evalLinear net.valueHead pooled).getD 0 0.0

  let invalid := invalidActionMask envCfg s
  let actionLogits :=
    s.actionVertices.map fun vertex =>
      if vertex = 0 || vertex > vertexNat then
        -1.0e30
      else
        vertexLogits.getD (vertex - 1) (-1.0e30)
  let masked :=
    (Array.range actionLogits.size).map fun i =>
      if invalid.getD i false then -1.0e30 else actionLogits.getD i (-1.0e30)
  (masked, value)

def obsRowsToTensor3d?
    (rows : Array (Array Float))
    (vertexDim tokenDim : UInt64) :
    Except String (Σ n : UInt64, T #[n, vertexDim, tokenDim]) := do
  let expected := vertexDim.toNat * tokenDim.toNat
  for i in [:rows.size] do
    let row := rows.getD i #[]
    if row.size != expected then
      throw s!"Observation row {i} width mismatch: expected {expected}, got {row.size}."
  let mut flat : Array Float := #[]
  for row in rows do
    for x in row do
      flat := flat.push x
  let n : UInt64 := rows.size.toUInt64
  let tDyn := data.fromFloatArray flat
  let t : T #[n, vertexDim, tokenDim] := reshape tDyn #[n, vertexDim, tokenDim]
  pure ⟨n, t⟩

def attentionMaskTensor?
    (rows : Array (Array Float))
    (vertexDim tokenDim : UInt64) :
    Except String (Σ n : UInt64, T #[n, vertexDim]) := do
  let expected := vertexDim.toNat * tokenDim.toNat
  let mut flat : Array Float := #[]
  for i in [:rows.size] do
    let row := rows.getD i #[]
    if row.size != expected then
      throw s!"Attention mask row {i} width mismatch: expected {expected}, got {row.size}."
    let maskRow := attentionMaskFromObservationFlat row vertexDim.toNat tokenDim.toNat
    for x in maskRow do
      flat := flat.push x
  let n : UInt64 := rows.size.toUInt64
  let tDyn := data.fromFloatArray flat
  let t : T #[n, vertexDim] := reshape tDyn #[n, vertexDim]
  pure ⟨n, t⟩

def actionIndexTensor
    (batchSize : UInt64)
    (actionVertices : Array VertexId1) :
    Except String (Σ actionDim : UInt64, T #[batchSize, actionDim]) := do
  let actionDim : UInt64 := actionVertices.size.toUInt64
  let mut flat : Array Int64 := #[]
  for _ in [:batchSize.toNat] do
    for vertex in actionVertices do
      if vertex = 0 then
        throw "Action surface contains vertex 0; expected 1-based vertex IDs."
      flat := flat.push (Int64.ofNat (vertex - 1))
  let dyn := data.fromInt64Array flat
  let t0 : T #[batchSize, actionDim] := reshape dyn #[batchSize, actionDim]
  pure ⟨actionDim, data.toLong t0⟩

end Examples.AlphaGradPort
