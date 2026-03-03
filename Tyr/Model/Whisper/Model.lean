/-
  Tyr/Model/Whisper/Model.lean

  Native Whisper encoder-decoder model in Tyr.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.LayerNorm
import Tyr.Model.Whisper.Config

namespace torch.whisper

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

private def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

private def activate {s : Shape} (name : String) (x : T s) : T s :=
  if name == "relu" then
    nn.relu x
  else
    nn.gelu x

structure WhisperAttention (dModel nHeads : UInt64) where
  qProjWeight : T #[dModel, dModel]
  qProjBias : T #[dModel]
  kProjWeight : T #[dModel, dModel]
  kProjBias : T #[dModel]
  vProjWeight : T #[dModel, dModel]
  vProjBias : T #[dModel]
  outProjWeight : T #[dModel, dModel]
  outProjBias : T #[dModel]
  deriving TensorStruct

namespace WhisperAttention

def init (dModel nHeads : UInt64) : IO (WhisperAttention dModel nHeads) := do
  if nHeads == 0 || dModel % nHeads != 0 then
    throw <| IO.userError s!"WhisperAttention requires d_model divisible by n_heads, got {dModel} and {nHeads}"
  let qProjWeight ← initWeight #[dModel, dModel] dModel
  let qProjBias := initBias #[dModel]
  let kProjWeight ← initWeight #[dModel, dModel] dModel
  let kProjBias := initBias #[dModel]
  let vProjWeight ← initWeight #[dModel, dModel] dModel
  let vProjBias := initBias #[dModel]
  let outProjWeight ← initWeight #[dModel, dModel] dModel
  let outProjBias := initBias #[dModel]
  pure { qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias }

def forwardCross {batch qSeq kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (queryStates : T #[batch, qSeq, dModel])
    (keyValueStates : T #[batch, kvSeq, dModel])
    (isCausal : Bool := false)
    : T #[batch, qSeq, dModel] :=
  let headDim : UInt64 := dModel / nHeads
  let q : T #[batch, qSeq, dModel] := affine3d queryStates m.qProjWeight m.qProjBias
  let k : T #[batch, kvSeq, dModel] := affine3d keyValueStates m.kProjWeight m.kProjBias
  let v : T #[batch, kvSeq, dModel] := affine3d keyValueStates m.vProjWeight m.vProjBias

  let q : T #[batch, qSeq, nHeads, headDim] := reshape q #[batch, qSeq, nHeads, headDim]
  let k : T #[batch, kvSeq, nHeads, headDim] := reshape k #[batch, kvSeq, nHeads, headDim]
  let v : T #[batch, kvSeq, nHeads, headDim] := reshape v #[batch, kvSeq, nHeads, headDim]

  let qh : T #[batch, nHeads, qSeq, headDim] := nn.transpose_for_attention q
  let kh : T #[batch, nHeads, kvSeq, headDim] := nn.transpose_for_attention k
  let vh : T #[batch, nHeads, kvSeq, headDim] := nn.transpose_for_attention v

  let attn : T #[batch, nHeads, qSeq, headDim] :=
    nn.scaledDotProductAttentionGQAQKV
      (n_kv_head := nHeads)
      qh
      kh
      vh
      0.0
      isCausal
      true
  let out : T #[batch, qSeq, nHeads, headDim] := nn.transpose_from_attention attn
  let out : T #[batch, qSeq, dModel] := reshape out #[batch, qSeq, dModel]
  affine3d out m.outProjWeight m.outProjBias

def forwardSelf {batch seq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (x : T #[batch, seq, dModel])
    (isCausal : Bool := false)
    : T #[batch, seq, dModel] :=
  forwardCross m x x isCausal

end WhisperAttention

structure WhisperEncoderLayer (cfg : WhisperConfig) where
  selfAttn : WhisperAttention cfg.dModel cfg.encoderAttentionHeads
  selfAttnLayerNorm : LayerNorm cfg.dModel
  fc1Weight : T #[cfg.encoderFfnDim, cfg.dModel]
  fc1Bias : T #[cfg.encoderFfnDim]
  fc2Weight : T #[cfg.dModel, cfg.encoderFfnDim]
  fc2Bias : T #[cfg.dModel]
  finalLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace WhisperEncoderLayer

def init (cfg : WhisperConfig) : IO (WhisperEncoderLayer cfg) := do
  let selfAttn ← WhisperAttention.init cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeight #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias := initBias #[cfg.encoderFfnDim]
  let fc2Weight ← initWeight #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias := initBias #[cfg.dModel]
  let finalLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

def forward {batch seq : UInt64}
    (m : WhisperEncoderLayer cfg)
    (x : T #[batch, seq, cfg.dModel])
    : T #[batch, seq, cfg.dModel] :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf h0 (isCausal := false)
  let x1 := x + h1
  let h2 := m.finalLayerNorm.forward3d x1
  let h3 : T #[batch, seq, cfg.encoderFfnDim] := affine3d h2 m.fc1Weight m.fc1Bias
  let h4 := activate cfg.activationFunction h3
  let h5 : T #[batch, seq, cfg.dModel] := affine3d h4 m.fc2Weight m.fc2Bias
  x1 + h5

end WhisperEncoderLayer

structure WhisperDecoderLayer (cfg : WhisperConfig) where
  selfAttn : WhisperAttention cfg.dModel cfg.decoderAttentionHeads
  selfAttnLayerNorm : LayerNorm cfg.dModel
  encoderAttn : WhisperAttention cfg.dModel cfg.decoderAttentionHeads
  encoderAttnLayerNorm : LayerNorm cfg.dModel
  fc1Weight : T #[cfg.decoderFfnDim, cfg.dModel]
  fc1Bias : T #[cfg.decoderFfnDim]
  fc2Weight : T #[cfg.dModel, cfg.decoderFfnDim]
  fc2Bias : T #[cfg.dModel]
  finalLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace WhisperDecoderLayer

def init (cfg : WhisperConfig) : IO (WhisperDecoderLayer cfg) := do
  let selfAttn ← WhisperAttention.init cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  let encoderAttn ← WhisperAttention.init cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeight #[cfg.decoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias := initBias #[cfg.decoderFfnDim]
  let fc2Weight ← initWeight #[cfg.dModel, cfg.decoderFfnDim] cfg.decoderFfnDim
  let fc2Bias := initBias #[cfg.dModel]
  let finalLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps
  pure {
    selfAttn
    selfAttnLayerNorm
    encoderAttn
    encoderAttnLayerNorm
    fc1Weight
    fc1Bias
    fc2Weight
    fc2Bias
    finalLayerNorm
  }

def forward {batch seq encSeq : UInt64}
    (m : WhisperDecoderLayer cfg)
    (x : T #[batch, seq, cfg.dModel])
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : T #[batch, seq, cfg.dModel] :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf h0 (isCausal := true)
  let x1 := x + h1
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardCross h2 encoderHidden (isCausal := false)
  let x2 := x1 + h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : T #[batch, seq, cfg.decoderFfnDim] := affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : T #[batch, seq, cfg.dModel] := affine3d h6 m.fc2Weight m.fc2Bias
  x2 + h7

end WhisperDecoderLayer

structure WhisperModel (cfg : WhisperConfig) where
  encoderConv1Weight : T #[cfg.dModel, cfg.numMelBins, 3]
  encoderConv1Bias : T #[cfg.dModel]
  encoderConv2Weight : T #[cfg.dModel, cfg.dModel, 3]
  encoderConv2Bias : T #[cfg.dModel]
  encoderPositionalEmbedding : T #[cfg.maxSourcePositions, cfg.dModel]
  encoderLayers : Array (WhisperEncoderLayer cfg)
  encoderLayerNorm : LayerNorm cfg.dModel

  decoderTokenEmbedding : T #[cfg.vocabSize, cfg.dModel]
  decoderPositionalEmbedding : T #[cfg.maxTargetPositions, cfg.dModel]
  decoderLayers : Array (WhisperDecoderLayer cfg)
  decoderLayerNorm : LayerNorm cfg.dModel
  deriving TensorStruct

namespace WhisperModel

def init (cfg : WhisperConfig) : IO (WhisperModel cfg) := do
  if !cfg.hasValidHeads then
    throw <| IO.userError
      s!"Invalid Whisper config: d_model={cfg.dModel}, encoder_heads={cfg.encoderAttentionHeads}, decoder_heads={cfg.decoderAttentionHeads}"
  let encoderConv1Weight ← initWeight #[cfg.dModel, cfg.numMelBins, 3] cfg.numMelBins
  let encoderConv1Bias := initBias #[cfg.dModel]
  let encoderConv2Weight ← initWeight #[cfg.dModel, cfg.dModel, 3] cfg.dModel
  let encoderConv2Bias := initBias #[cfg.dModel]
  let encoderPositionalEmbedding ← initWeight #[cfg.maxSourcePositions, cfg.dModel] cfg.dModel

  let mut encoderLayers : Array (WhisperEncoderLayer cfg) := #[]
  for _ in [:cfg.encoderLayers.toNat] do
    encoderLayers := encoderLayers.push (← WhisperEncoderLayer.init cfg)
  let encoderLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ← initWeight #[cfg.vocabSize, cfg.dModel] cfg.dModel
  let decoderPositionalEmbedding ← initWeight #[cfg.maxTargetPositions, cfg.dModel] cfg.dModel
  let mut decoderLayers : Array (WhisperDecoderLayer cfg) := #[]
  for _ in [:cfg.decoderLayers.toNat] do
    decoderLayers := decoderLayers.push (← WhisperDecoderLayer.init cfg)
  let decoderLayerNorm := LayerNorm.init cfg.dModel cfg.layerNormEps

  pure {
    encoderConv1Weight
    encoderConv1Bias
    encoderConv2Weight
    encoderConv2Bias
    encoderPositionalEmbedding
    encoderLayers
    encoderLayerNorm
    decoderTokenEmbedding
    decoderPositionalEmbedding
    decoderLayers
    decoderLayerNorm
  }

def encode {batch frames : UInt64}
    (m : WhisperModel cfg)
    (inputFeatures : T #[batch, cfg.numMelBins, frames])
    : IO (T #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel]) := do
  let x1 : T #[batch, cfg.dModel, frames] :=
    reshape
      (nn.gelu (nn.conv1d_group_bias inputFeatures m.encoderConv1Weight m.encoderConv1Bias 1 1 1 1))
      #[batch, cfg.dModel, frames]
  let outSeq := WhisperConfig.conv2OutputSeq frames
  if outSeq > cfg.maxSourcePositions then
    throw <| IO.userError
      s!"Whisper encoder sequence {outSeq} exceeds max_source_positions {cfg.maxSourcePositions}"
  let x2 : T #[batch, cfg.dModel, outSeq] :=
    reshape
      (nn.gelu (nn.conv1d_group_bias x1 m.encoderConv2Weight m.encoderConv2Bias 2 1 1 1))
      #[batch, cfg.dModel, outSeq]
  let x2t : T #[batch, outSeq, cfg.dModel] := permute x2 #[0, 2, 1]
  let pos : T #[outSeq, cfg.dModel] := data.slice m.encoderPositionalEmbedding 0 0 outSeq
  let posB : T #[batch, outSeq, cfg.dModel] :=
    nn.expand (reshape pos #[1, outSeq, cfg.dModel]) #[batch, outSeq, cfg.dModel]
  let mut h : T #[batch, outSeq, cfg.dModel] := x2t + posB
  for layer in m.encoderLayers do
    h := layer.forward h
  pure (m.encoderLayerNorm.forward3d h)

def decode {batch seq encSeq : UInt64}
    (m : WhisperModel cfg)
    (inputIds : T #[batch, seq])
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : IO (T #[batch, seq, cfg.dModel]) := do
  if seq > cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder sequence {seq} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : T #[batch, seq, cfg.dModel] := nn.embedding inputIds m.decoderTokenEmbedding
  let pos : T #[seq, cfg.dModel] := data.slice m.decoderPositionalEmbedding 0 0 seq
  let posB : T #[batch, seq, cfg.dModel] :=
    nn.expand (reshape pos #[1, seq, cfg.dModel]) #[batch, seq, cfg.dModel]
  let mut h : T #[batch, seq, cfg.dModel] := tokEmb + posB
  for layer in m.decoderLayers do
    h := layer.forward h encoderHidden
  pure (m.decoderLayerNorm.forward3d h)

end WhisperModel

structure WhisperForConditionalGeneration (cfg : WhisperConfig) where
  model : WhisperModel cfg
  projOut : T #[cfg.vocabSize, cfg.dModel]
  deriving TensorStruct

namespace WhisperForConditionalGeneration

def init (cfg : WhisperConfig) : IO (WhisperForConditionalGeneration cfg) := do
  let model ← WhisperModel.init cfg
  let projOut ← initWeight #[cfg.vocabSize, cfg.dModel] cfg.dModel
  pure { model, projOut }

def encode {batch frames : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.numMelBins, frames])
    : IO (T #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel]) :=
  m.model.encode inputFeatures

def decode {batch seq encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (encoderHidden : T #[batch, encSeq, cfg.dModel])
    : IO (T #[batch, seq, cfg.vocabSize]) := do
  let hidden ← m.model.decode inputIds encoderHidden
  pure (linear3d hidden m.projOut)

end WhisperForConditionalGeneration

end torch.whisper
