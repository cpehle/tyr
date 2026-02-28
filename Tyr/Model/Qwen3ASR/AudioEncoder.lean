/-
  Tyr/Model/Qwen3ASR/AudioEncoder.lean

  Lean4 port of Qwen3-ASR audio encoder:
  - 3x Conv2d downsampler
  - sinusoidal absolute position embedding
  - transformer encoder stack
  - output projection to thinker hidden space
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.LayerNorm
import Tyr.Model.Qwen3ASR.Config

namespace torch.qwen3asr

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

private def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

private def activate {s : Shape} (name : String) (x : T s) : T s :=
  if name == "relu" then
    nn.relu x
  else if name == "silu" then
    nn.silu x
  else
    nn.gelu x

private def sinusoidPosition {seq dim : UInt64} (maxTimescale : Float := 10000.0) : T #[seq, dim] :=
  if dim % 2 == 0 then
    let half := dim / 2
    let pos : T #[seq] := toFloat' (torch.arange 0 seq 1)
    let idx : T #[half] := toFloat' (torch.arange 0 half 1)
    let denom : Float := if half <= 1 then 1.0 else (half - 1).toFloat
    let logInc := Float.log maxTimescale / denom
    let invTimescales : T #[half] := nn.exp (mul_scalar idx (-logInc))
    let scaled : T #[seq, half] := reshape (einsum2 "i,j->ij" pos invTimescales) #[seq, half]
    let s := nn.sin scaled
    let c := nn.cos scaled
    reshape (nn.cat s c 1) #[seq, dim]
  else
    torch.zeros #[seq, dim]

/-- Multi-head self-attention used by the audio encoder. -/
structure AudioAttention (cfg : AudioEncoderConfig) where
  qProjWeight : T #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel]
  qProjBias : T #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  kProjWeight : T #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel]
  kProjBias : T #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  vProjWeight : T #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel]
  vProjBias : T #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  outProjWeight : T #[cfg.dModel, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  outProjBias : T #[cfg.dModel]
  deriving TensorStruct

namespace AudioAttention

def init (cfg : AudioEncoderConfig) : IO (AudioAttention cfg) := do
  let qProjWeight ← initWeight
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let qProjBias := initBias #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let kProjWeight ← initWeight
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let kProjBias := initBias #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let vProjWeight ← initWeight
    #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg, cfg.dModel] cfg.dModel
  let vProjBias := initBias #[cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  let outProjWeight ← initWeight
    #[cfg.dModel, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
    (cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg)
  let outProjBias := initBias #[cfg.dModel]
  pure {
    qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias
  }

def forward {batch seq : UInt64}
    (m : AudioAttention cfg)
    (x : T #[batch, seq, cfg.dModel])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.dModel] :=
  let q : T #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    affine3d x m.qProjWeight m.qProjBias
  let k : T #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    affine3d x m.kProjWeight m.kProjBias
  let v : T #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    affine3d x m.vProjWeight m.vProjBias

  let q : T #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] :=
    reshape q #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]
  let k : T #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] :=
    reshape k #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]
  let v : T #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg] :=
    reshape v #[batch, seq, cfg.encoderAttentionHeads, AudioEncoderConfig.headDim cfg]

  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  let out :=
    match attnMask with
    | some mask =>
      nn.scaledDotProductAttentionGQAMask
        q k v mask cfg.attentionDropout false true
    | none =>
      nn.scaledDotProductAttentionGQA
        q k v cfg.attentionDropout false true

  let out := nn.transpose_from_attention out
  let out : T #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg] :=
    reshape out #[batch, seq, cfg.encoderAttentionHeads * AudioEncoderConfig.headDim cfg]
  affine3d out m.outProjWeight m.outProjBias

end AudioAttention

/-- One transformer block in the audio encoder. -/
structure AudioEncoderLayer (cfg : AudioEncoderConfig) where
  selfAttn : AudioAttention cfg
  selfAttnLayerNorm : LayerNorm cfg.dModel
  fc1Weight : T #[cfg.encoderFfnDim, cfg.dModel]
  fc1Bias : T #[cfg.encoderFfnDim]
  fc2Weight : T #[cfg.dModel, cfg.encoderFfnDim]
  fc2Bias : T #[cfg.dModel]
  finalLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace AudioEncoderLayer

def init (cfg : AudioEncoderConfig) : IO (AudioEncoderLayer cfg) := do
  let selfAttn ← AudioAttention.init cfg
  let selfAttnLayerNorm := LayerNorm.init cfg.dModel 1e-5
  let fc1Weight ← initWeight #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias := initBias #[cfg.encoderFfnDim]
  let fc2Weight ← initWeight #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias := initBias #[cfg.dModel]
  let finalLayerNorm := LayerNorm.init cfg.dModel 1e-5
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

def forward {batch seq : UInt64}
    (m : AudioEncoderLayer cfg)
    (x : T #[batch, seq, cfg.dModel])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.dModel] :=
  let residual0 := x
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forward h0 attnMask
  let h2 := residual0 + h1

  let residual1 := h2
  let h3 := m.finalLayerNorm.forward3d h2
  let h4 : T #[batch, seq, cfg.encoderFfnDim] := affine3d h3 m.fc1Weight m.fc1Bias
  let h5 := activate cfg.activationFunction h4
  let h6 : T #[batch, seq, cfg.dModel] := affine3d h5 m.fc2Weight m.fc2Bias
  residual1 + h6

end AudioEncoderLayer

/-- Full Qwen3-ASR audio encoder. -/
structure AudioEncoder (cfg : AudioEncoderConfig) where
  conv2d1Weight : T #[cfg.downsampleHiddenSize, 1, 3, 3]
  conv2d1Bias : T #[cfg.downsampleHiddenSize]
  conv2d2Weight : T #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  conv2d2Bias : T #[cfg.downsampleHiddenSize]
  conv2d3Weight : T #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  conv2d3Bias : T #[cfg.downsampleHiddenSize]
  convOutWeight : T #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg]
  layers : Array (AudioEncoderLayer cfg)
  lnPost : LayerNorm cfg.dModel
  proj1Weight : T #[cfg.dModel, cfg.dModel]
  proj1Bias : T #[cfg.dModel]
  proj2Weight : T #[cfg.outputDim, cfg.dModel]
  proj2Bias : T #[cfg.outputDim]
  deriving TensorStruct

namespace AudioEncoder

def init (cfg : AudioEncoderConfig) : IO (AudioEncoder cfg) := do
  if cfg.dModel % cfg.encoderAttentionHeads != 0 then
    throw <| IO.userError
      s!"Audio encoder requires dModel divisible by encoderAttentionHeads, got {cfg.dModel} and {cfg.encoderAttentionHeads}"

  let conv2d1Weight ← initWeight #[cfg.downsampleHiddenSize, 1, 3, 3] 1
  let conv2d1Bias := initBias #[cfg.downsampleHiddenSize]
  let conv2d2Weight ← initWeight #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] cfg.downsampleHiddenSize
  let conv2d2Bias := initBias #[cfg.downsampleHiddenSize]
  let conv2d3Weight ← initWeight #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3] cfg.downsampleHiddenSize
  let conv2d3Bias := initBias #[cfg.downsampleHiddenSize]

  let convOutWeight ← initWeight #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg] (AudioEncoderConfig.convOutInDim cfg)

  let mut layers : Array (AudioEncoderLayer cfg) := #[]
  for _ in [:cfg.encoderLayers.toNat] do
    let layer ← AudioEncoderLayer.init cfg
    layers := layers.push layer

  let lnPost := LayerNorm.init cfg.dModel 1e-5
  let proj1Weight ← initWeight #[cfg.dModel, cfg.dModel] cfg.dModel
  let proj1Bias := initBias #[cfg.dModel]
  let proj2Weight ← initWeight #[cfg.outputDim, cfg.dModel] cfg.dModel
  let proj2Bias := initBias #[cfg.outputDim]

  pure {
    conv2d1Weight, conv2d1Bias, conv2d2Weight, conv2d2Bias, conv2d3Weight, conv2d3Bias,
    convOutWeight, layers, lnPost, proj1Weight, proj1Bias, proj2Weight, proj2Bias
  }

/-- Port of reference chunk planner used before convolutional downsampling.
    For each feature length `L`, emits `ceil(L / (2*n_window))` chunk lengths where
    all chunks are `2*n_window` except the last chunk (`L % (2*n_window)` or full). -/
def buildChunkLengths (cfg : AudioEncoderConfig) (featureLens : Array UInt64) : Array UInt64 :=
  let step := cfg.nWindow * 2
  if step == 0 then
    featureLens
  else
    Id.run do
      let mut out : Array UInt64 := #[]
      for len in featureLens do
        if len == 0 then
          pure ()
        else
          let chunkNum := (len + step - 1) / step
          for j in [:chunkNum.toNat] do
            if j + 1 == chunkNum.toNat then
              let tail := len % step
              out := out.push (if tail == 0 then step else tail)
            else
              out := out.push step
      out

/-- Per-chunk length transform through `_get_feat_extract_output_lengths`. -/
def chunkLengthsAfterCnn (_cfg : AudioEncoderConfig) (chunkLens : Array UInt64) : Array UInt64 :=
  chunkLens.map AudioEncoderConfig.featExtractOutputLength

/-- Port of `cu_chunk_lens` creation in reference audio encoder forward.
    Returns inclusive prefix-sum source with initial `0`. -/
def buildCuChunkLensCumsum
    (cfg : AudioEncoderConfig)
    (afterCnnLens : Array UInt64)
    (paddedAfterCnnWidth : UInt64)
    : Array UInt64 :=
  let denom := cfg.nWindow * 2
  let ratio := if denom == 0 then 1 else cfg.nWindowInfer / denom
  let windowAfterCnn0 := paddedAfterCnnWidth * (if ratio == 0 then 1 else ratio)
  let windowAfterCnn := if windowAfterCnn0 == 0 then 1 else windowAfterCnn0
  Id.run do
    let mut segs : Array UInt64 := #[0]
    for cnnLen in afterCnnLens do
      let full := cnnLen / windowAfterCnn
      for _ in [:full.toNat] do
        segs := segs.push windowAfterCnn
      let rem := cnnLen % windowAfterCnn
      if rem != 0 then
        segs := segs.push rem
    let mut csum : UInt64 := 0
    let mut out : Array UInt64 := #[]
    for s in segs do
      csum := csum + s
      out := out.push csum
    out

def forward {batch frames : UInt64}
    (m : AudioEncoder cfg)
    (inputFeatures : T #[batch, cfg.numMelBins, frames])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim] :=
  let t1 := AudioEncoderConfig.downsampleOnce frames
  let t2 := AudioEncoderConfig.downsampleTwice frames
  let t3 := AudioEncoderConfig.framesAfterConv3 cfg frames

  let x0 : T #[batch, 1, cfg.numMelBins, frames] := reshape inputFeatures #[batch, 1, cfg.numMelBins, frames]

  let x1 : T #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1] :=
    reshape (nn.gelu (nn.conv2d_bias x0 m.conv2d1Weight m.conv2d1Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv1 cfg, t1]
  let x2 : T #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2] :=
    reshape (nn.gelu (nn.conv2d_bias x1 m.conv2d2Weight m.conv2d2Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv2 cfg, t2]
  let x3 : T #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3] :=
    reshape (nn.gelu (nn.conv2d_bias x2 m.conv2d3Weight m.conv2d3Bias #[2, 2] #[1, 1]))
      #[batch, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg, t3]

  let x3t : T #[batch, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg] :=
    reshape (permute x3 #[0, 3, 1, 2]) #[batch, t3, cfg.downsampleHiddenSize, AudioEncoderConfig.melAfterConv3 cfg]

  let xFlat : T #[batch, t3, AudioEncoderConfig.convOutInDim cfg] :=
    reshape x3t #[batch, t3, AudioEncoderConfig.convOutInDim cfg]
  let xEmb : T #[batch, t3, cfg.dModel] := linear3d xFlat m.convOutWeight

  let pos : T #[t3, cfg.dModel] := sinusoidPosition (seq := t3) (dim := cfg.dModel)
  let posBatch : T #[batch, t3, cfg.dModel] := nn.expand (reshape pos #[1, t3, cfg.dModel]) #[batch, t3, cfg.dModel]
  let hidden0 := xEmb + posBatch

  let hidden := match attnMask with
    | some mask => m.layers.foldl (fun h layer => layer.forward h (some mask)) hidden0
    | none => m.layers.foldl (fun h layer => layer.forward h none) hidden0

  let hidden := m.lnPost.forward3d hidden
  let hidden : T #[batch, t3, cfg.dModel] := affine3d hidden m.proj1Weight m.proj1Bias
  let hidden := activate cfg.activationFunction hidden
  affine3d hidden m.proj2Weight m.proj2Bias

end AudioEncoder

instance {cfg : AudioEncoderConfig} {batch frames : UInt64} :
    Module (AudioEncoder cfg)
      (T #[batch, cfg.numMelBins, frames])
      (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg frames, cfg.outputDim]) where
  forward := fun m x => m.forward x none

end torch.qwen3asr
