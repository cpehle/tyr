/-!
  Examples/BranchingFlows/ContinuousTrainDemo.lean

  Minimal continuous-flow training demo using BranchingFlows + Torch.
  Shows how to build a branching batch and run a single training step.
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

private def linBridge {dim : UInt64} (_ : Unit)
    (x0 x1 : T #[dim]) (t0 t1 : Float) : T #[dim] :=
  let dt := t1 - t0
  x0 + (x1 - x0) * dt

structure ToyParams (dim : UInt64) where
  W : T #[dim, dim]
  b : T #[dim]
  deriving TensorStruct

namespace ToyParams

def init (dim : UInt64) : IO (ToyParams dim) := do
  let W ← torch.randn #[dim, dim]
  let b ← torch.zeros #[dim]
  pure { W, b }

end ToyParams

def toyModel {maxLen dim : UInt64} : BranchingModelContinuous maxLen dim (ToyParams dim) :=
  { forward := fun {batch} params state t => do
      let anchorPred := torch.affine3d state params.W params.b
      let splitLogits := torch.zeros #[batch, maxLen]
      let delLogits := torch.zeros #[batch, maxLen]
      pure (anchorPred, splitLogits, delLogits) }

private def randomState {dim : UInt64} (len : Nat) : IO (BranchingState (T #[dim])) := do
  let mut xs : Array (T #[dim]) := #[]
  for _ in [:len] do
    let x ← torch.randn #[dim]
    xs := xs.push x
  let groups := Array.replicate len 0
  pure (BranchingState.mkDefault xs groups)

def runDemo : IO Unit := do
  let dim : UInt64 := 2
  let maxLen : UInt64 := 16
  let x1 ← randomState (dim := dim) 6
  let x1s := #[x1]
  let times := #[0.5]
  let x0Sampler : FlowNode (T #[dim]) → T #[dim] := fun node => node.data

  let (bridgeResult, _rng) :=
    branchingBridge (linBridge (dim := dim)) () x0Sampler x1s times
      uniformTimeDist uniformTimeDist (sequentialUniformPolicy (T #[dim]))
      canonicalAnchorMerge
      (maxLen := some maxLen.toNat)
      (rng := { state := 123 })

  let cfg : BranchingTrainConfig := {
    maxLen := maxLen
    anchorWeight := 1.0
    splitsWeight := 0.1
    delWeight := 0.1
    gradClip := 1.0
  }

  let params ← ToyParams.init dim
  let opt := Optim.adamw (lr := 1.0e-3)
  let optState := opt.init params
  let (params', _optState', report) ←
    trainStepContinuous (maxLen := maxLen) (dim := dim) cfg (toyModel (maxLen := maxLen) (dim := dim))
      params optState bridgeResult 1.0e-3
  let _params' : ToyParams dim := params'
  IO.println s!"Branching training step: {report}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
