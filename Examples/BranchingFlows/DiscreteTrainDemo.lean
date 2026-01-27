/-!
  Examples/BranchingFlows/DiscreteTrainDemo.lean

  Minimal discrete-flow training demo using BranchingFlows + Torch.
  This mirrors the Julia demo structure with a lightweight token model.
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

private def discreteBridge (_ : Unit) (x0 x1 : Nat) (_t0 _t1 : Float) : Nat :=
  x1

structure ToyParams (vocab hidden : UInt64) where
  embed : T #[vocab, hidden]
  proj : T #[hidden, vocab]
  splitW : T #[1, hidden]
  delW : T #[1, hidden]
  deriving TensorStruct

namespace ToyParams

def init (vocab hidden : UInt64) : IO (ToyParams vocab hidden) := do
  let embed ← torch.randn #[vocab, hidden]
  let proj ← torch.randn #[hidden, vocab]
  let splitW ← torch.randn #[1, hidden]
  let delW ← torch.randn #[1, hidden]
  pure { embed, proj, splitW, delW }

end ToyParams

def toyModel {maxLen vocab hidden : UInt64} : BranchingModel maxLen vocab (ToyParams vocab hidden) :=
  { forward := fun {batch} params state _t => do
      let emb := torch.nn.embedding (batch := batch) (seq := maxLen) (vocab := vocab) (embed := hidden) state params.embed
      let bias : T #[vocab] := torch.zeros #[vocab]
      let logits := torch.affine3d emb params.proj bias
      let split3 := torch.linear3d (x := emb) (weight := params.splitW)
      let del3 := torch.linear3d (x := emb) (weight := params.delW)
      let splitLogits := torch.squeeze split3 2
      let delLogits := torch.squeeze del3 2
      pure (logits, splitLogits, delLogits) }

def runDemo : IO Unit := do
  let vocab : UInt64 := 6
  let hidden : UInt64 := 8
  let maxLen : UInt64 := 16

  let tokens : Array Nat := #[1, 2, 3, 4, 2, 1]
  let groups : Array Int := Array.replicate tokens.size 0
  let x1 : BranchingState Nat := BranchingState.mkDefault tokens groups
  let x1s := #[x1]
  let times := #[0.4]
  let x0Sampler : FlowNode Nat → Nat := fun node => node.data
  let merger := fun (a b : Nat) (_w1 _w2 : Nat) => a

  let (bridgeResult, _rng) :=
    branchingBridge discreteBridge () x0Sampler x1s times
      uniformTimeDist uniformTimeDist (sequentialUniformPolicy Nat) merger
      (maxLen := some maxLen.toNat)
      (rng := { state := 7 })

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    padToken := 0
    anchorWeight := 1.0
    splitsWeight := 0.1
    delWeight := 0.1
    gradClip := 1.0
  }

  let params ← ToyParams.init vocab hidden
  let opt := Optim.adamw (lr := 1.0e-3)
  let optState := opt.init params
  let (_params', _optState', report) ←
    trainStep (maxLen := maxLen) (vocab := vocab) cfg (toyModel (maxLen := maxLen) (vocab := vocab) (hidden := hidden))
      params optState bridgeResult 1.0e-3
  IO.println s!"Discrete branching training step: {report}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
