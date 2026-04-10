import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim

/-!
  Flow language model helpers.

  This module ports the core tensor contracts from the sibling FLM repository:
  continuous one-hot corruption, flow-matching loss, Euler sampling, and the
  PSD semigroup target used by FMLM. The transformer itself is supplied by the
  caller through `FlowDenoiser`, which keeps this module independent of any
  particular backbone implementation.
-/

namespace torch.flm

open torch

/-- Runtime knobs shared by FLM and FMLM losses. -/
structure FlowConfig where
  tMin : Float := 0.0
  tMax : Float := 1.0
  softcap : Float := 30.0
  eps : Float := 1.0e-5
  weightDecay : Float := 0.0
  gradClip : Float := 0.0
  deriving Repr

/-- Continuous/discrete time conversion.

  The Python FLM code uses LUTs for alpha/gamma conversion on discrete states.
  Tyr callers can plug that mapping in here; the identity map gives the standard
  linear flow interpolation used by the core FLM equations. -/
structure TimeMap where
  tauToT : {batch : UInt64} → T #[batch] → T #[batch]
  tToTau : {batch : UInt64} → T #[batch] → T #[batch]

def TimeMap.identity : TimeMap :=
  { tauToT := fun {_batch} tau => tau
    tToTau := fun {_batch} t => t }

/-- Backbone interface for continuous-token denoisers.

  `x` has shape `[batch, seq, vocab]`. `tau` is the source time, and
  `tauPrime` is present for flow-map/FMLM denoisers that condition on a target
  time. The returned tensor is raw logits; this module applies softcap and
  log-softmax before losses or semigroup targets consume it. -/
structure FlowDenoiser (seq vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → T #[batch, seq, vocab] → T #[batch]
    → Option (T #[batch]) → IO (T #[batch, seq, vocab])
  lossWeight : Option ({batch : UInt64} → Params → T #[batch] → Option (T #[batch])
    → IO (T #[batch])) := none

structure LossReport where
  loss : Float
  nlls : Float
  numTokens : Float
  deriving Repr

structure SemigroupTimes (batch : UInt64) where
  tauS : T #[batch]
  tauU : T #[batch]
  tauT : T #[batch]

private def expandBatch {batch seq : UInt64} (x : T #[batch]) : T #[batch, seq] :=
  nn.expand (nn.unsqueeze x 1) #[batch, seq]

private def expandTime3 {batch seq vocab : UInt64} (x : T #[batch])
    : T #[batch, seq, vocab] :=
  nn.expand (nn.unsqueeze (nn.unsqueeze x 1) 2) #[batch, seq, vocab]

/-- Token ids to one-hot vectors using `eye` as the embedding table. -/
def oneHot {batch seq vocab : UInt64} (tokens : T #[batch, seq])
    : T #[batch, seq, vocab] :=
  nn.embedding tokens (torch.eye vocab)

/-- Sample uniformly from the configured training time interval. -/
def sampleTau (cfg : FlowConfig) (batch : UInt64) : IO (T #[batch]) := do
  let eps ← torch.rand #[batch]
  pure (eps * (cfg.tMax - cfg.tMin) + cfg.tMin)

/-- Continuous FLM corruption: `x_t = (1 - t) noise + t one_hot(x_1)`. -/
def corruptContinuous {batch seq vocab : UInt64}
    (tokens : T #[batch, seq])
    (t : T #[batch])
    : IO (T #[batch, seq, vocab] × T #[batch, seq, vocab]) := do
  let target := oneHot (vocab := vocab) tokens
  let noise ← torch.randn #[batch, seq, vocab]
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let one := torch.ones_like target
  pure (((one - t3) * noise + t3 * target), target)

def cappedLogSoftmax {s : Shape} (cfg : FlowConfig) (logits : T s) : T s :=
  nn.log_softmax (nn.softcap logits cfg.softcap) (-1)

def forwardLogProbs {seq vocab batch : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (x : T #[batch, seq, vocab])
    (tau : T #[batch])
    (tauPrime : Option (T #[batch]) := none)
    : IO (T #[batch, seq, vocab]) := do
  let logits ← model.forward params x tau tauPrime
  pure (cappedLogSoftmax cfg logits)

/-- Per-token cross entropy against a dense one-hot/probability target. -/
def denseTargetNll {batch seq vocab : UInt64}
    (logProbs target : T #[batch, seq, vocab]) : T #[batch, seq] :=
  torch.mul_scalar (nn.sumDim (target * logProbs) 2) (-1.0)

private def maybeApplyLossWeight {batch seq : UInt64} {Params : Type}
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tau : T #[batch])
    (tauPrime : Option (T #[batch]))
    (loss : T #[batch, seq])
    : IO (T #[batch, seq]) := do
  match model.lossWeight with
  | none => pure loss
  | some weightFn =>
      let w ← weightFn params tau tauPrime
      let w2 := expandBatch (seq := seq) w
      pure (nn.exp (torch.mul_scalar w2 (-1.0)) * loss + w2)

/-- FLM flow-matching loss at caller-provided times. -/
def flowLossGivenTau {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : T #[batch, seq])
    (tau : T #[batch])
    : IO (T #[batch, seq]) := do
  let t := timeMap.tauToT tau
  let (xT, target) ← corruptContinuous (vocab := vocab) tokens t
  let logProbs ← forwardLogProbs cfg model params xT tau none
  let loss := denseTargetNll logProbs target
  maybeApplyLossWeight model params tau none loss

/-- FLM flow-matching loss with uniformly sampled training times. -/
def flowLoss {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : T #[batch, seq])
    : IO (T #[batch, seq]) := do
  let tau ← sampleTau cfg batch
  flowLossGivenTau cfg timeMap model params tokens tau

def maskedMean {batch seq : UInt64} (loss validTokens : T #[batch, seq])
    (eps : Float := 1.0e-8) : T #[] :=
  let nlls := nn.sumAll (loss * validTokens)
  let denom := nn.sumAll validTokens
  nn.div nlls (denom + eps)

/-- Optimizer step for an FLM loss. `validTokens` is a float mask. -/
def trainStep {seq vocab batch : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (tokens : T #[batch, seq])
    (validTokens : T #[batch, seq])
    (lr : Float)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × LossReport) := do
  let params := TensorStruct.zeroGrads params
  let perToken ← flowLoss cfg timeMap model params tokens
  let lossT := maskedMean perToken validTokens
  autograd.backwardLoss lossT
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip
  let grads := TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState
  let nlls := nn.sumAll (perToken * validTokens)
  let ntok := nn.sumAll validTokens
  let report : LossReport :=
    { loss := nn.item lossT
      nlls := nn.item nlls
      numTokens := nn.item ntok }
  pure (params', optState', report)

private def eulerFlowStep {batch seq vocab : UInt64}
    (cfg : FlowConfig)
    (z dataPred : T #[batch, seq, vocab])
    (t dt : T #[batch]) : T #[batch, seq, vocab] :=
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let dt3 := expandTime3 (seq := seq) (vocab := vocab) dt
  let one := torch.ones_like z
  let v := nn.div (dataPred - z) (one - t3 + cfg.eps)
  z + dt3 * v

/-- Generate tokens by Euler integration from Gaussian noise to one-hot data. -/
def generateFLM {seq vocab batch : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (steps : Nat)
    : IO (T #[batch, seq]) := do
  if steps == 0 then
    throw (IO.userError "generateFLM requires steps > 0")
  let mut z ← torch.randn #[batch, seq, vocab]
  for i in [:steps] do
    let tauCurrF := cfg.tMin + (cfg.tMax - cfg.tMin) * (i.toFloat / steps.toFloat)
    let tauNextF := cfg.tMin + (cfg.tMax - cfg.tMin) * ((i + 1).toFloat / steps.toFloat)
    let tauCurr := torch.full #[batch] tauCurrF
    let tauNext := torch.full #[batch] tauNextF
    let tCurr := timeMap.tauToT tauCurr
    let tNext := timeMap.tauToT tauNext
    let logPred ← forwardLogProbs cfg model params z tauCurr none
    let dataPred := nn.exp logPred
    if i + 1 == steps then
      z := dataPred
    else
      z := eulerFlowStep cfg z dataPred tCurr (tNext - tCurr)
  pure (nn.argmax z 2)

/-- Interpolate from a source state toward a predicted data point. -/
def flowInterpolate {batch seq vocab : UInt64}
    (cfg : FlowConfig)
    (xS dataPred : T #[batch, seq, vocab])
    (s t : T #[batch])
    : T #[batch, seq, vocab] :=
  let s3 := expandTime3 (seq := seq) (vocab := vocab) s
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let one := torch.ones_like xS
  let denom := one - s3 + cfg.eps
  nn.div (one - t3) denom * xS + nn.div (t3 - s3) denom * dataPred

/-- PSD semigroup mixing coefficient from the FMLM paper/code. -/
def psdLambda {batch seq vocab : UInt64}
    (cfg : FlowConfig)
    (s u t : T #[batch])
    : T #[batch, seq, vocab] :=
  let s3 := expandTime3 (seq := seq) (vocab := vocab) s
  let u3 := expandTime3 (seq := seq) (vocab := vocab) u
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let one := torch.ones_like s3
  let num := (one - t3) * (u3 - s3)
  let den := (one - u3) * (t3 - s3) + cfg.eps
  nn.div num den

/-- FMLM PSD target: `lambda D_su + (1-lambda) D_ut`. -/
def psdTarget {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (xS : T #[batch, seq, vocab])
    (tauS tauU tauT : T #[batch])
    : IO (T #[batch, seq, vocab]) := do
  let s := timeMap.tauToT tauS
  let u := timeMap.tauToT tauU
  let t := timeMap.tauToT tauT
  let logDSU ← forwardLogProbs cfg model params xS tauS (some tauU)
  let dSU := nn.exp logDSU
  let xSU := flowInterpolate cfg xS dSU s u
  let logDUT ← forwardLogProbs cfg model params xSU tauU (some tauT)
  let dUT := nn.exp logDUT
  let lam := psdLambda (seq := seq) (vocab := vocab) cfg s u t
  let one := torch.ones_like lam
  pure (lam * dSU + (one - lam) * dUT)

/-- FMLM PSD per-token loss for caller-provided semigroup times. -/
def psdLossGivenTau {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : T #[batch, seq])
    (tauS tauU tauT : T #[batch])
    : IO (T #[batch, seq]) := do
  let s := timeMap.tauToT tauS
  let (xS, _) ← corruptContinuous (vocab := vocab) tokens s
  let target ← psdTarget cfg timeMap model params xS tauS tauU tauT
  let logDST ← forwardLogProbs cfg model params xS tauS (some tauT)
  let loss := denseTargetNll logDST target
  maybeApplyLossWeight model params tauS (some tauT) loss

/-- Diagonal FMLM loss. If no teacher target is supplied, this reduces to FLM. -/
def diagonalLossGivenTau {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : T #[batch, seq])
    (tau : T #[batch])
    (teacherTarget : Option (T #[batch, seq, vocab]) := none)
    : IO (T #[batch, seq]) := do
  let t := timeMap.tauToT tau
  let (xT, oneHotTarget) ← corruptContinuous (vocab := vocab) tokens t
  let target := teacherTarget.getD oneHotTarget
  let logProbs ← forwardLogProbs cfg model params xT tau (some tau)
  let loss := denseTargetNll logProbs target
  maybeApplyLossWeight model params tau (some tau) loss

/-- Sample off-diagonal PSD times with midpoint `u`. -/
def samplePsdTimes (batch : UInt64) : IO (SemigroupTimes batch) := do
  let dTau ← torch.rand #[batch]
  let eps ← torch.rand #[batch]
  let one := torch.ones_like dTau
  let tauS := eps * (one - dTau)
  let tauT := tauS + dTau
  let tauU := (tauS + tauT) * (0.5 : Float)
  pure { tauS, tauU, tauT }

end torch.flm
