/-!
  Examples/BranchingFlows/MixedTrainDemo.lean

  Mixed continuous + discrete branching demo (full Julia-style loop).
  Generates synthetic data, trains a tiny model, and writes .pt tensors for plotting.
-/
import Tyr
import Tyr.Optim
import Tyr.Model.BranchingFlows
import Tyr.Model.BranchingFlowsTrain

namespace Examples.BranchingFlows

open torch
open torch.branching

private def clamp01 (x : Float) : Float :=
  max 0.0 (min x 1.0)

private def uniformTimeDist : TimeDist :=
  { cdf := fun t => clamp01 t,
    pdf := fun t => if t < 0.0 || t > 1.0 then 0.0 else 1.0,
    quantile := fun p => clamp01 p }

private def linBridge2d (_ : Unit) (x0 x1 : T #[2]) (t0 t1 : Float) : T #[2] :=
  let dt := t1 - t0
  x0 + (x1 - x0) * dt

private def discreteBridge (_ : Unit) (x0 x1 : Nat) (_t0 _t1 : Float) : Nat :=
  x1

structure MixParams (vocab hidden : UInt64) where
  contW : T #[2, hidden]
  contB : T #[hidden]
  timeW : T #[1, hidden]
  timeB : T #[hidden]
  embed : T #[vocab, hidden]
  locW : T #[hidden, 2]
  locB : T #[2]
  proj : T #[hidden, vocab]
  tokB : T #[vocab]
  splitW : T #[1, hidden]
  splitB : T #[1]
  delW : T #[1, hidden]
  delB : T #[1]
  deriving TensorStruct

namespace MixParams

def init (vocab hidden : UInt64) : IO (MixParams vocab hidden) := do
  let contW ← torch.randn #[2, hidden]
  let contB ← torch.zeros #[hidden]
  let timeW ← torch.randn #[1, hidden]
  let timeB ← torch.zeros #[hidden]
  let embed ← torch.randn #[vocab, hidden]
  let locW ← torch.randn #[hidden, 2]
  let locB ← torch.zeros #[2]
  let proj ← torch.randn #[hidden, vocab]
  let tokB ← torch.zeros #[vocab]
  let splitW ← torch.randn #[1, hidden]
  let splitB ← torch.zeros #[1]
  let delW ← torch.randn #[1, hidden]
  let delB ← torch.zeros #[1]
  pure { contW, contB, timeW, timeB, embed, locW, locB, proj, tokB, splitW, splitB, delW, delB }

end MixParams

private def randLen (rng : Rng) : Nat × Rng := Id.run do
  let (u, rng') := randFloat rng
  if u < 0.5 then
    let (k, rng'') := randPoisson rng' 4.0
    (k + 1, rng'')
  else
    let (k, rng'') := randPoisson rng' 20.0
    (k + 1, rng'')

private def randBounds (rng : Rng) : (Float × Float) × Rng := Id.run do
  let (u1, rng') := randFloat rng
  let (u2, rng'') := randFloat rng'
  let left := -u1
  let right := u2
  ((left, right), rng'')

private def f (x : Float) : Float :=
  x + Float.sin (3.0 * x)

private def mkPoint (x y : Float) : T #[2] :=
  let xi := reshape (torch.full #[] x) #[1]
  let yi := reshape (torch.full #[] y) #[1]
  reshape (torch.data.stack1d #[xi, yi] 0) #[2]

private def X1target (rng : Rng) : BranchingState (T #[2] × Nat) × Rng := Id.run do
  let (n, rng') := randLen rng
  let ((left, right), rng'') := randBounds rng'
  let step := (right - left) / (n.toFloat)
  let mut points : Array (T #[2]) := #[]
  for i in [:n] do
    let x := left + step * i.toFloat
    let y := f x
    points := points.push (mkPoint x y)
  let mut toks : Array Nat := Array.replicate n 1
  let (u, rng''') := randFloat rng''
  let odd : Nat := if u < 0.5 then 0 else 1
  for i in [:n] do
    if i % 2 == odd then
      toks := toks.set! i 2
  let state := Array.zipWith (fun p t => (p, t)) points toks
  let groups := Array.replicate n 0
  ({ state := state, groupings := groups, del := Array.replicate n false, ids := Array.ofFn (fun i => Int.ofNat (i.val + 1)),
     branchmask := Array.replicate n true, flowmask := Array.replicate n true, padmask := Array.replicate n true }, rng''')

private def mergeTuple (a b : (T #[2] × Nat)) (w1 w2 : Nat) : (T #[2] × Nat) :=
  let cont := canonicalAnchorMerge a.1 b.1 w1 w2
  (cont, 0)

private def splitResult
    (res : BranchingBridgeResult (T #[2] × Nat))
    : BranchingBridgeResult (T #[2]) × BranchingBridgeResult Nat := Id.run do
  let mapSegCont := fun (s : Segment (T #[2] × Nat)) =>
    { Xt := s.Xt.1, t := s.t, anchor := s.anchor.1, descendants := s.descendants, del := s.del,
      branchable := s.branchable, flowable := s.flowable, group := s.group, lastCoalescence := s.lastCoalescence, id := s.id }
  let mapSegDisc := fun (s : Segment (T #[2] × Nat)) =>
    { Xt := s.Xt.2, t := s.t, anchor := s.anchor.2, descendants := s.descendants, del := s.del,
      branchable := s.branchable, flowable := s.flowable, group := s.group, lastCoalescence := s.lastCoalescence, id := s.id }
  let segsCont := res.segments.map (fun arr => arr.map mapSegCont)
  let segsDisc := res.segments.map (fun arr => arr.map mapSegDisc)
  let XtCont := res.Xt.map (fun x =>
    { state := x.state.map (fun v => v.1), groupings := x.groupings, del := x.del, ids := x.ids,
      branchmask := x.branchmask, flowmask := x.flowmask, padmask := x.padmask })
  let XtDisc := res.Xt.map (fun x =>
    { state := x.state.map (fun v => v.2), groupings := x.groupings, del := x.del, ids := x.ids,
      branchmask := x.branchmask, flowmask := x.flowmask, padmask := x.padmask })
  let anchorCont := res.X1anchor.map (fun arr => arr.map (fun v => v.1))
  let anchorDisc := res.X1anchor.map (fun arr => arr.map (fun v => v.2))
  let resCont : BranchingBridgeResult (T #[2]) :=
    { t := res.t, segments := segsCont, Xt := XtCont, X1anchor := anchorCont,
      descendants := res.descendants, del := res.del, splitsTarget := res.splitsTarget, prevCoalescence := res.prevCoalescence }
  let resDisc : BranchingBridgeResult Nat :=
    { t := res.t, segments := segsDisc, Xt := XtDisc, X1anchor := anchorDisc,
      descendants := res.descendants, del := res.del, splitsTarget := res.splitsTarget, prevCoalescence := res.prevCoalescence }
  (resCont, resDisc)

private def maskedCrossEntropy {batch maxLen vocab : UInt64}
    (logits : T #[batch, maxLen, vocab])
    (targets : T #[batch, maxLen])
    (mask : T #[batch, maxLen]) : T #[] :=
  let n : UInt64 := batch * maxLen
  let logits2 := reshape logits #[n, vocab]
  let targets2 := reshape targets #[n]
  let mask2 := reshape mask #[n]
  let per := torch.nn.cross_entropy_none logits2 targets2
  let masked := per * mask2
  let denom := torch.nn.sumAll mask2
  (torch.nn.sumAll masked) / (denom + 1.0e-8)

private def maskedMSE {batch maxLen : UInt64}
    (pred target mask : T #[batch, maxLen]) : T #[] :=
  let diff := pred - target
  let sq := diff * diff
  let masked := sq * mask
  let denom := torch.nn.sumAll mask
  (torch.nn.sumAll masked) / (denom + 1.0e-8)

private def maskedMSE3d {batch maxLen dim : UInt64}
    (pred target : T #[batch, maxLen, dim]) (mask : T #[batch, maxLen]) : T #[] :=
  let mask3 := torch.expand (torch.unsqueeze mask 2) #[batch, maxLen, dim]
  let diff := pred - target
  let sq := diff * diff
  let masked := sq * mask3
  let denom := torch.nn.sumAll mask3
  (torch.nn.sumAll masked) / (denom + 1.0e-8)

private def maskedBCEWithLogits {batch maxLen : UInt64}
    (logits target mask : T #[batch, maxLen]) : T #[] :=
  let probs := torch.nn.sigmoid logits
  let loss := torch.nn.binary_cross_entropy probs target (some mask) "sum"
  let denom := torch.nn.sumAll mask
  loss / (denom + 1.0e-8)

def runDemo : IO Unit := do
  let vocab : UInt64 := 3
  let hidden : UInt64 := 32
  let maxLen : UInt64 := 64
  let steps : Nat := 200
  let batch : Nat := 16

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.1
    delWeight := 0.1
    gradClip := 1.0
  }

  let params ← MixParams.init vocab hidden
  let opt := Optim.adamw (lr := 1.0e-3)
  let mut optState := opt.init params
  let mut params := params
  let mut rng : Rng := { state := 0 }

  for i in [:steps] do
    let mut batchStates : Array (BranchingState (T #[2] × Nat)) := #[]
    for _ in [:batch] do
      let (x1, rng') := X1target rng
      rng := rng'
      batchStates := batchStates.push x1
    let mut times : Array Float := #[]
    for _ in [:batch] do
      let (u, rng') := randFloat rng
      rng := rng'
      times := times.push u

    let (bridgeRes, rng') :=
      branchingBridge
        (fun _ x0 x1 t0 t1 => (linBridge2d () x0.1 x1.1 t0 t1, discreteBridge () x0.2 x1.2 t0 t1))
        ()
        (fun root => (root.data.1, 0))
        batchStates times
        uniformTimeDist uniformTimeDist
        (sequentialUniformPolicy (T #[2] × Nat))
        mergeTuple
        (useBranchingTimeProb := 0.5)
        (deletionPad := 1.5)
        (maxLen := some maxLen.toNat)
        (rng := rng)
    rng := rng'

    let (resCont, resDisc) := splitResult bridgeRes
    let ⟨batchU, packedCont⟩ ← packBranchingTensor (dim := 2) cfg resCont
    let ⟨_, packedDisc⟩ ← packBranchingNat cfg resDisc

    let params := TensorStruct.zeroGrads params
    let t := packedDisc.t
    let contEmb := torch.affine3d packedCont.state params.contW params.contB
    let discEmb := torch.nn.embedding (batch := batchU) (seq := maxLen) (vocab := vocab) (embed := hidden) packedDisc.state params.embed
    let t1 := torch.unsqueeze t 1
    let t2 := torch.unsqueeze t1 1
    let tExp := torch.expand t2 #[batchU, maxLen, 1]
    let timeEmb := torch.affine3d tExp params.timeW params.timeB
    let hiddenState := torch.tanh (contEmb + discEmb + timeEmb)

    let predLoc := torch.affine3d hiddenState params.locW params.locB
    let predTok := torch.affine3d hiddenState params.proj params.tokB
    let split3 := torch.affine3d hiddenState params.splitW params.splitB
    let del3 := torch.affine3d hiddenState params.delW params.delB
    let splitLogits := torch.squeeze split3 2
    let delLogits := torch.squeeze del3 2

    let mask := packedDisc.padmask * packedDisc.flowmask
    let contLoss := maskedMSE3d predLoc packedCont.anchor mask
    let discLoss := maskedCrossEntropy predTok packedDisc.anchor mask
    let splitLoss := maskedMSE splitLogits packedDisc.splitsTarget mask
    let delLoss := maskedBCEWithLogits delLogits packedDisc.delTarget mask
    let total := contLoss + discLoss + splitLoss + delLoss

    torch.autograd.backwardLoss total
    if cfg.gradClip > 0 then
      pure ()

    let grads := TensorStruct.grads params
    let (params', optState') := Optim.step opt params grads optState
    params := params'
    optState := optState'

    if i % 20 == 0 then
      let tl := torch.nn.item total
      IO.println s!"step {i}: loss={tl}"

  -- Write plotting tensors for one batch
  let (bridgeRes, _rng') :=
    branchingBridge
      (fun _ x0 x1 t0 t1 => (linBridge2d () x0.1 x1.1 t0 t1, discreteBridge () x0.2 x1.2 t0 t1))
      ()
      (fun root => (root.data.1, 0))
      #[(X1target rng).1] #[0.5]
      uniformTimeDist uniformTimeDist
      (sequentialUniformPolicy (T #[2] × Nat))
      mergeTuple
      (maxLen := some maxLen.toNat)
      (rng := rng)
  let (resCont, resDisc) := splitResult bridgeRes
  let ⟨batchU, packedCont⟩ ← packBranchingTensor (dim := 2) cfg resCont
  let ⟨_, packedDisc⟩ ← packBranchingNat cfg resDisc
  let contEmb := torch.affine3d packedCont.state params.contW params.contB
  let discEmb := torch.nn.embedding (batch := batchU) (seq := maxLen) (vocab := vocab) (embed := hidden) packedDisc.state params.embed
  let t1 := torch.unsqueeze packedDisc.t 1
  let t2 := torch.unsqueeze t1 1
  let tExp := torch.expand t2 #[batchU, maxLen, 1]
  let timeEmb := torch.affine3d tExp params.timeW params.timeB
  let hiddenState := torch.tanh (contEmb + discEmb + timeEmb)
  let predLoc := torch.affine3d hiddenState params.locW params.locB

  torch.data.saveTensor predLoc "examples_branching_mixed_pred.pt"
  torch.data.saveTensor packedCont.anchor "examples_branching_mixed_anchor.pt"
  torch.data.saveTensor packedDisc.anchor "examples_branching_mixed_tokens.pt"
  let plotMask := packedDisc.padmask * packedDisc.flowmask
  torch.data.saveTensor plotMask "examples_branching_mixed_mask.pt"
  IO.println "Wrote examples_branching_mixed_pred.pt, examples_branching_mixed_anchor.pt, examples_branching_mixed_tokens.pt, examples_branching_mixed_mask.pt"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
