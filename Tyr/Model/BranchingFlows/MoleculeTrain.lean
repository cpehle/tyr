import Tyr.Model.BranchingFlows.Molecule
import Tyr.Model.BranchingFlowsTrain
import Tyr.Optim

/-!
  Tensor packing helpers for molecule-shaped BranchingFlows batches.

  The generic training module has separate packers for token states and
  continuous coordinate states.  Molecule generation needs both in one element:
  3D coordinates plus a discrete atom label.
-/

namespace torch.branching

open torch
open torch.nn

structure BranchingMoleculeBatch (batch maxLen : UInt64) where
  t : T #[batch]
  coord : T #[batch, maxLen, 3]
  coordAnchor : T #[batch, maxLen, 3]
  label : T #[batch, maxLen]
  labelAnchor : T #[batch, maxLen]
  labelLossScale : T #[batch, maxLen]
  padmask : T #[batch, maxLen]
  flowmask : T #[batch, maxLen]
  splitsTarget : T #[batch, maxLen]
  delTarget : T #[batch, maxLen]
  deriving Repr

structure BranchingMoleculeModel (maxLen vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → T #[batch, maxLen, 3] → T #[batch, maxLen] → T #[batch]
    → T #[batch, maxLen]
    → IO (T #[batch, maxLen, 3] × T #[batch, maxLen, vocab] × T #[batch, maxLen] × T #[batch, maxLen])

private def coordOffset (maxLenNat bi j k : Nat) : Nat :=
  (bi * maxLenNat + j) * 3 + k

private def writeVec3 (arr : Array Float) (offset : Nat) (v : Vec3) : Array Float :=
  let arr := arr.set! offset v.x
  let arr := arr.set! (offset + 1) v.y
  arr.set! (offset + 2) v.z

def packBranchingMolecule (cfg : BranchingTrainConfig)
    (result : BranchingBridgeResult MoleculeAtom)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    : IO (Sigma fun batch => BranchingMoleculeBatch batch cfg.maxLen) := do
  let batchNat := result.t.size
  if result.Xt.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.Xt size mismatch")
  if result.X1anchor.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.X1anchor size mismatch")
  if result.splitsTarget.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.splitsTarget size mismatch")
  if result.del.size != batchNat then
    throw (IO.userError "BranchingBridgeResult.del size mismatch")

  let maxLenNat := cfg.maxLen.toNat
  let total := batchNat * maxLenNat
  let coordTotal := total * 3

  let mut coordArr : Array Float := Array.replicate coordTotal 0.0
  let mut coordAnchorArr : Array Float := Array.replicate coordTotal 0.0
  let mut labelArr : Array Int64 := Array.replicate total cfg.padToken
  let mut labelAnchorArr : Array Int64 := Array.replicate total cfg.padToken
  let mut labelScaleArr : Array Float := Array.replicate total 0.0
  let mut padArr : Array Int64 := Array.replicate total 0
  let mut flowArr : Array Int64 := Array.replicate total 0
  let mut splitsArr : Array Int64 := Array.replicate total 0
  let mut delArr : Array Int64 := Array.replicate total 0

  for bi in [:batchNat] do
    let x := result.Xt[bi]!
    let anchors := result.X1anchor[bi]!
    let splits := result.splitsTarget[bi]!
    let dels := result.del[bi]!
    if anchors.size != x.state.size then
      throw (IO.userError "X1anchor length mismatch")
    if splits.size != x.state.size then
      throw (IO.userError "splitsTarget length mismatch")
    if dels.size != x.state.size then
      throw (IO.userError "del length mismatch")
    if x.state.size > maxLenNat then
      throw (IO.userError "Branching sample exceeds maxLen; increase maxLen or resample")

    for j in [:maxLenNat] do
      let idx := bi * maxLenNat + j
      if h : j < x.state.size then
        let atom := x.state[j]'h
        let anchor := anchors.getD j default
        let coordIdx := coordOffset maxLenNat bi j 0
        coordArr := writeVec3 coordArr coordIdx atom.coord
        coordAnchorArr := writeVec3 coordAnchorArr coordIdx anchor.coord
        labelArr := labelArr.set! idx (Int64.ofNat atom.label)
        labelAnchorArr := labelAnchorArr.set! idx (Int64.ofNat anchor.label)
        labelScaleArr := labelScaleArr.set! idx
          (match labelDFM with
           | some dfm => dfm.lossScale (result.t.getD bi 0.0)
           | none => 1.0)
        splitsArr := splitsArr.set! idx (Int64.ofNat (splits.getD j 0))
        delArr := delArr.set! idx (if dels.getD j false then 1 else 0)
        padArr := padArr.set! idx 1
        flowArr := flowArr.set! idx (if x.flowmask.getD j false then 1 else 0)
      else
        pure ()

  let batchU : UInt64 := batchNat.toUInt64
  let coord := reshape (data.fromFloatArray coordArr) #[batchU, cfg.maxLen, 3]
  let coordAnchor := reshape (data.fromFloatArray coordAnchorArr) #[batchU, cfg.maxLen, 3]
  let label := reshape (data.fromInt64Array labelArr) #[batchU, cfg.maxLen]
  let labelAnchor := reshape (data.fromInt64Array labelAnchorArr) #[batchU, cfg.maxLen]
  let labelLossScale := reshape (data.fromFloatArray labelScaleArr) #[batchU, cfg.maxLen]
  let padmask := toFloat' (reshape (data.fromInt64Array padArr) #[batchU, cfg.maxLen])
  let flowmask := toFloat' (reshape (data.fromInt64Array flowArr) #[batchU, cfg.maxLen])
  let splitsTarget := toFloat' (reshape (data.fromInt64Array splitsArr) #[batchU, cfg.maxLen])
  let delTarget := toFloat' (reshape (data.fromInt64Array delArr) #[batchU, cfg.maxLen])

  let timeScaled := result.t.map (fun t => ((t * cfg.timeScale).toUInt64).toInt64)
  let t := reshape (data.fromInt64Array timeScaled) #[batchU]
  let t := (toFloat' t) / cfg.timeScale

  return ⟨batchU, {
    t,
    coord,
    coordAnchor,
    label,
    labelAnchor,
    labelLossScale,
    padmask,
    flowmask,
    splitsTarget,
    delTarget
  }⟩

def packBranchingMoleculeWithDFM (cfg : BranchingTrainConfig)
    (labelDFM : DistNoisyDiscreteConfig)
    (result : BranchingBridgeResult MoleculeAtom)
    : IO (Sigma fun batch => BranchingMoleculeBatch batch cfg.maxLen) :=
  packBranchingMolecule cfg result (labelDFM := some labelDFM)

private def moleculeMask {batch maxLen : UInt64}
    (pad flow : T #[batch, maxLen]) : T #[batch, maxLen] :=
  pad * flow

private def moleculeMaskedWeightedCrossEntropy {batch maxLen vocab : UInt64}
    (logits : T #[batch, maxLen, vocab])
    (targets : T #[batch, maxLen])
    (mask : T #[batch, maxLen])
    (weights : T #[batch, maxLen]) : T #[] :=
  let n : UInt64 := batch * maxLen
  let logits2 := reshape logits #[n, vocab]
  let targets2 := reshape targets #[n]
  let mask2 := reshape mask #[n]
  let weights2 := reshape weights #[n]
  let per := nn.cross_entropy_none logits2 targets2
  let masked := per * mask2 * weights2
  let denom := nn.sumAll mask2
  nn.div (nn.sumAll masked) (denom + (1.0e-8 : Float))

private def moleculeMaskedMSE3d {batch maxLen dim : UInt64}
    (pred target : T #[batch, maxLen, dim]) (mask : T #[batch, maxLen]) : T #[] :=
  let mask3 := nn.expand (nn.unsqueeze mask 2) #[batch, maxLen, dim]
  let diff := pred - target
  let sq := diff * diff
  let masked := sq * mask3
  let denom := nn.sumAll mask3
  nn.div (nn.sumAll masked) (denom + (1.0e-8 : Float))

structure BranchingMoleculeLossReport where
  total : Float
  coord : Float
  label : Float
  splits : Float
  del : Float
  deriving Repr

def moleculeLosses {batch maxLen vocab : UInt64}
    (cfg : BranchingTrainConfig)
    (packed : BranchingMoleculeBatch batch maxLen)
    (coordPred : T #[batch, maxLen, 3])
    (labelLogits : T #[batch, maxLen, vocab])
    (splitLogits : T #[batch, maxLen])
    (delLogits : T #[batch, maxLen]) :
    T #[] × BranchingMoleculeLossReport :=
  let mask := moleculeMask packed.padmask packed.flowmask
  let coordLoss := moleculeMaskedMSE3d coordPred packed.coordAnchor mask
  let labelLoss := moleculeMaskedWeightedCrossEntropy labelLogits packed.labelAnchor mask packed.labelLossScale
  let splitLoss := maskedSplitPoissonLoss splitLogits packed.splitsTarget mask
  let delLoss := maskedBCEWithLogits delLogits packed.delTarget mask
  let totalLoss :=
    coordLoss * cfg.anchorWeight +
    labelLoss * cfg.anchorWeight +
    splitLoss * cfg.splitsWeight +
    delLoss * cfg.delWeight
  (totalLoss, {
    total := nn.item totalLoss,
    coord := nn.item coordLoss,
    label := nn.item labelLoss,
    splits := nn.item splitLoss,
    del := nn.item delLoss
  })

def trainStepMolecule {maxLen vocab : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : BranchingTrainConfig)
    (model : BranchingMoleculeModel maxLen vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (result : BranchingBridgeResult MoleculeAtom)
    (lr : Float)
    (labelDFM : Option DistNoisyDiscreteConfig := none)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × BranchingMoleculeLossReport) := do
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)
  let ⟨batch, packed⟩ ← packBranchingMolecule cfg result labelDFM
  let (coordPred, labelLogits, splitLogits, delLogits) ←
    model.forward (batch := batch) params packed.coord packed.label packed.t packed.padmask
  let (totalLoss, report) :=
    moleculeLosses (vocab := vocab) cfg packed coordPred labelLogits splitLogits delLogits

  autograd.backwardLoss totalLoss
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip

  let grads := TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState

  return (params', optState', report)

end torch.branching
