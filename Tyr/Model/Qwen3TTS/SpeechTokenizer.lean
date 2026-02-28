/-
  Tyr/Model/Qwen3TTS/SpeechTokenizer.lean

  Lean-native Qwen3-TTS 12Hz speech-tokenizer decoder (codes -> waveform).
  This mirrors the Python `Qwen3TTSTokenizerV2Decoder` compute path using real
  `speech_tokenizer/model.safetensors` weights.

  Scope in this module:
  - pure Lean decoder compute
  - real weight loading from SafeTensors
  - native WAV emission via Tyr runtime

  Note: this implementation currently targets the released 12Hz config
  (16 quantizers, 1024 latent dim, 2x convnext upsample, 4 decoder blocks).
-/
import Tyr.Torch
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.RoPE
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3tts

open Lean

/-! ## Config parsing -/

structure SpeechTokenizer12HzDecoderConfig where
  codebookSize : UInt64 := 2048
  codebookDim : UInt64 := 512
  latentDim : UInt64 := 1024
  hiddenSize : UInt64 := 512
  intermediateSize : UInt64 := 1024
  layerScaleInitialScale : Float := 0.01
  rmsNormEps : Float := 1e-5
  ropeTheta : Float := 10000.0
  headDim : UInt64 := 64
  numAttentionHeads : UInt64 := 16
  numKeyValueHeads : UInt64 := 16
  numHiddenLayers : UInt64 := 8
  numQuantizers : UInt64 := 16
  numSemanticQuantizers : UInt64 := 1
  slidingWindow : UInt64 := 72
  upsampleRates : Array UInt64 := #[8, 5, 4, 3]
  upsamplingRatios : Array UInt64 := #[2, 2]
  decoderDim : UInt64 := 1536
  deriving Repr, Inhabited

structure SpeechTokenizer12HzConfig where
  decoder : SpeechTokenizer12HzDecoderConfig := {}
  outputSampleRate : UInt64 := 24000
  decodeUpsampleRate : UInt64 := 1920
  deriving Repr, Inhabited

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr a => some a
  | _ => none

private def getNat? (j : Json) : Option Nat :=
  match (FromJson.fromJson? j : Except String Nat) with
  | .ok n => some n
  | .error _ => none

private def getFloat? (j : Json) : Option Float :=
  match (FromJson.fromJson? j : Except String Float) with
  | .ok x => some x
  | .error _ => (getNat? j).map (·.toFloat)

private def getNatFieldD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getNat? with
  | some n => n.toUInt64
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def getUInt64ArrayFieldD (j : Json) (key : String) (d : Array UInt64) : Array UInt64 :=
  match getObjVal? j key >>= getArr? with
  | some arr =>
    Id.run do
      let mut out : Array UInt64 := #[]
      for item in arr do
        match getNat? item with
        | some n => out := out.push n.toUInt64
        | none => pure ()
      if out.isEmpty then d else out
  | none => d

namespace SpeechTokenizer12HzConfig

private def parseDecoderConfig (j : Json) (d : SpeechTokenizer12HzDecoderConfig := {})
    : SpeechTokenizer12HzDecoderConfig := {
  codebookSize := getNatFieldD j "codebook_size" d.codebookSize
  codebookDim := getNatFieldD j "codebook_dim" d.codebookDim
  latentDim := getNatFieldD j "latent_dim" d.latentDim
  hiddenSize := getNatFieldD j "hidden_size" d.hiddenSize
  intermediateSize := getNatFieldD j "intermediate_size" d.intermediateSize
  layerScaleInitialScale := getFloatFieldD j "layer_scale_initial_scale" d.layerScaleInitialScale
  rmsNormEps := getFloatFieldD j "rms_norm_eps" d.rmsNormEps
  ropeTheta := getFloatFieldD j "rope_theta" d.ropeTheta
  headDim := getNatFieldD j "head_dim" d.headDim
  numAttentionHeads := getNatFieldD j "num_attention_heads" d.numAttentionHeads
  numKeyValueHeads := getNatFieldD j "num_key_value_heads" d.numKeyValueHeads
  numHiddenLayers := getNatFieldD j "num_hidden_layers" d.numHiddenLayers
  numQuantizers := getNatFieldD j "num_quantizers" d.numQuantizers
  numSemanticQuantizers := getNatFieldD j "num_semantic_quantizers" d.numSemanticQuantizers
  slidingWindow := getNatFieldD j "sliding_window" d.slidingWindow
  upsampleRates := getUInt64ArrayFieldD j "upsample_rates" d.upsampleRates
  upsamplingRatios := getUInt64ArrayFieldD j "upsampling_ratios" d.upsamplingRatios
  decoderDim := getNatFieldD j "decoder_dim" d.decoderDim
}

private def requireTrue (ok : Bool) (msg : String) : IO Unit := do
  unless ok do
    throw (IO.userError msg)

/-- Load tokenizer config from `speech_tokenizer/config.json`. -/
def loadFromFile (path : String) (defaults : SpeechTokenizer12HzConfig := {})
    : IO SpeechTokenizer12HzConfig := do
  let root ← parseJsonFile path
  let decoder :=
    match getObjVal? root "decoder_config" with
    | some j => parseDecoderConfig j defaults.decoder
    | none => defaults.decoder
  pure {
    decoder
    outputSampleRate := getNatFieldD root "output_sample_rate" defaults.outputSampleRate
    decodeUpsampleRate := getNatFieldD root "decode_upsample_rate" defaults.decodeUpsampleRate
  }

/-- Validate this Lean implementation supports the provided tokenizer config. -/
def validateSupported (cfg : SpeechTokenizer12HzConfig) : IO Unit := do
  let d := cfg.decoder
  requireTrue (d.codebookSize == 2048) s!"Unsupported speech tokenizer codebook_size={d.codebookSize} (expected 2048)"
  requireTrue (d.codebookDim == 512) s!"Unsupported speech tokenizer codebook_dim={d.codebookDim} (expected 512)"
  requireTrue (d.latentDim == 1024) s!"Unsupported speech tokenizer latent_dim={d.latentDim} (expected 1024)"
  requireTrue (d.hiddenSize == 512) s!"Unsupported speech tokenizer hidden_size={d.hiddenSize} (expected 512)"
  requireTrue (d.intermediateSize == 1024) s!"Unsupported speech tokenizer intermediate_size={d.intermediateSize} (expected 1024)"
  requireTrue (d.headDim == 64) s!"Unsupported speech tokenizer head_dim={d.headDim} (expected 64)"
  requireTrue (d.numAttentionHeads == 16) s!"Unsupported speech tokenizer num_attention_heads={d.numAttentionHeads} (expected 16)"
  requireTrue (d.numKeyValueHeads == 16) s!"Unsupported speech tokenizer num_key_value_heads={d.numKeyValueHeads} (expected 16)"
  requireTrue (d.numHiddenLayers == 8) s!"Unsupported speech tokenizer num_hidden_layers={d.numHiddenLayers} (expected 8)"
  requireTrue (d.numQuantizers == 16) s!"Unsupported speech tokenizer num_quantizers={d.numQuantizers} (expected 16)"
  requireTrue (d.numSemanticQuantizers == 1) s!"Unsupported speech tokenizer num_semantic_quantizers={d.numSemanticQuantizers} (expected 1)"
  requireTrue (d.slidingWindow == 72) s!"Unsupported speech tokenizer sliding_window={d.slidingWindow} (expected 72)"
  requireTrue (d.upsampleRates == #[8, 5, 4, 3]) s!"Unsupported speech tokenizer upsample_rates={d.upsampleRates} (expected #[8,5,4,3])"
  requireTrue (d.upsamplingRatios == #[2, 2]) s!"Unsupported speech tokenizer upsampling_ratios={d.upsamplingRatios} (expected #[2,2])"
  requireTrue (d.decoderDim == 1536) s!"Unsupported speech tokenizer decoder_dim={d.decoderDim} (expected 1536)"
  requireTrue (cfg.decodeUpsampleRate == 1920) s!"Unsupported decode_upsample_rate={cfg.decodeUpsampleRate} (expected 1920)"

end SpeechTokenizer12HzConfig

/-! ## Decoder parameters -/

private def freeze {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (toFloat' t) false

private def loadFrozen (path : String) (name : String) (s : Shape) : IO (T s) := do
  let t ← safetensors.loadTensor path name s
  pure (freeze t)

private def addBias3d {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (bias : T #[channels]) : T #[batch, channels, frames] :=
  let b : T #[1, channels, 1] := reshape bias #[1, channels, 1]
  x + nn.expand b #[batch, channels, frames]

private def scale3d {batch seq dim : UInt64}
    (x : T #[batch, seq, dim])
    (scale : T #[dim]) : T #[batch, seq, dim] :=
  let s : T #[1, 1, dim] := reshape scale #[1, 1, dim]
  x * nn.expand s #[batch, seq, dim]

private def clampWave {s : Shape} (x : T s) : T s :=
  let minT : T s := torch.full s (-1.0)
  let maxT : T s := torch.full s 1.0
  let x1 : T s := where_ (lt_scalar x (-1.0)) minT x
  where_ (gt x1 maxT) maxT x1

private def clampMin1d {n : UInt64} (x : T #[n]) (minVal : Float) : T #[n] :=
  let minT : T #[n] := torch.full #[n] minVal
  where_ (lt_scalar x minVal) minT x

structure QuantizerCodebook where
  clusterUsage : T #[2048]
  embeddingSum : T #[2048, 256]

structure RVQDecode where
  outputProj : T #[512, 256, 1]
  codebooks : Array QuantizerCodebook

structure DecoderTransformerLayer where
  inputLayernormWeight : T #[512]
  qProj : T #[1024, 512]
  kProj : T #[1024, 512]
  vProj : T #[1024, 512]
  oProj : T #[512, 1024]
  selfAttnLayerScale : T #[512]
  postAttentionLayernormWeight : T #[512]
  gateProj : T #[1024, 512]
  upProj : T #[1024, 512]
  downProj : T #[512, 1024]
  mlpLayerScale : T #[512]

structure UpsampleConvNeXtBlock where
  dwconvWeight : T #[1024, 1, 7]
  dwconvBias : T #[1024]
  normWeight : T #[1024]
  normBias : T #[1024]
  pwconv1Weight : T #[4096, 1024]
  pwconv1Bias : T #[4096]
  pwconv2Weight : T #[1024, 4096]
  pwconv2Bias : T #[1024]
  gamma : T #[1024]

structure UpsampleStage where
  transConvWeight : T #[1024, 1024, 2]
  transConvBias : T #[1024]
  convNeXt : UpsampleConvNeXtBlock

structure DecoderResidualUnit (dim : UInt64) where
  conv1Dilation : UInt64 := 1
  act1Alpha : T #[dim]
  act1Beta : T #[dim]
  conv1Weight : T #[dim, dim, 7]
  conv1Bias : T #[dim]
  act2Alpha : T #[dim]
  act2Beta : T #[dim]
  conv2Weight : T #[dim, dim, 1]
  conv2Bias : T #[dim]

structure DecoderBlock (inDim outDim upRate : UInt64) where
  snakeAlpha : T #[inDim]
  snakeBeta : T #[inDim]
  transWeight : T #[inDim, outDim, 2 * upRate]
  transBias : T #[outDim]
  res1 : DecoderResidualUnit outDim
  res2 : DecoderResidualUnit outDim
  res3 : DecoderResidualUnit outDim

structure SpeechTokenizer12HzDecoder where
  rvqFirst : RVQDecode
  rvqRest : RVQDecode

  preConvWeight : T #[1024, 512, 3]
  preConvBias : T #[1024]

  preTransformerInputProj : T #[512, 1024]
  preTransformerInputBias : T #[512]
  preTransformerLayers : Array DecoderTransformerLayer
  preTransformerNormWeight : T #[512]
  preTransformerOutputProj : T #[1024, 512]
  preTransformerOutputBias : T #[1024]

  upsample0 : UpsampleStage
  upsample1 : UpsampleStage

  decoderConv0Weight : T #[1536, 1024, 7]
  decoderConv0Bias : T #[1536]

  decoderBlock1 : DecoderBlock 1536 768 8
  decoderBlock2 : DecoderBlock 768 384 5
  decoderBlock3 : DecoderBlock 384 192 4
  decoderBlock4 : DecoderBlock 192 96 3

  finalSnakeAlpha : T #[96]
  finalSnakeBeta : T #[96]
  finalConvWeight : T #[1, 96, 7]
  finalConvBias : T #[1]

  outputSampleRate : UInt64 := 24000
  decodeUpsampleRate : UInt64 := 1920

namespace SpeechTokenizer12HzDecoder

private def loadCodebook (weightsPath : String) (namePrefix : String) : IO QuantizerCodebook := do
  let clusterUsage ← loadFrozen weightsPath s!"{namePrefix}.cluster_usage" #[2048]
  let embeddingSum ← loadFrozen weightsPath s!"{namePrefix}.embedding_sum" #[2048, 256]
  pure { clusterUsage, embeddingSum }

private def loadRVQ (weightsPath : String) (namePrefix : String) (nCodebooks : Nat) : IO RVQDecode := do
  let outputProj ← loadFrozen weightsPath s!"{namePrefix}.output_proj.weight" #[512, 256, 1]
  let mut codebooks : Array QuantizerCodebook := #[]
  for i in [:nCodebooks] do
    let cb ← loadCodebook weightsPath s!"{namePrefix}.vq.layers.{i}._codebook"
    codebooks := codebooks.push cb
  pure { outputProj, codebooks }

private def loadTransformerLayer (weightsPath : String) (i : Nat) : IO DecoderTransformerLayer := do
  let p := s!"decoder.pre_transformer.layers.{i}"
  let inputLayernormWeight ← loadFrozen weightsPath s!"{p}.input_layernorm.weight" #[512]
  let qProj ← loadFrozen weightsPath s!"{p}.self_attn.q_proj.weight" #[1024, 512]
  let kProj ← loadFrozen weightsPath s!"{p}.self_attn.k_proj.weight" #[1024, 512]
  let vProj ← loadFrozen weightsPath s!"{p}.self_attn.v_proj.weight" #[1024, 512]
  let oProj ← loadFrozen weightsPath s!"{p}.self_attn.o_proj.weight" #[512, 1024]
  let selfAttnLayerScale ← loadFrozen weightsPath s!"{p}.self_attn_layer_scale.scale" #[512]
  let postAttentionLayernormWeight ← loadFrozen weightsPath s!"{p}.post_attention_layernorm.weight" #[512]
  let gateProj ← loadFrozen weightsPath s!"{p}.mlp.gate_proj.weight" #[1024, 512]
  let upProj ← loadFrozen weightsPath s!"{p}.mlp.up_proj.weight" #[1024, 512]
  let downProj ← loadFrozen weightsPath s!"{p}.mlp.down_proj.weight" #[512, 1024]
  let mlpLayerScale ← loadFrozen weightsPath s!"{p}.mlp_layer_scale.scale" #[512]
  pure {
    inputLayernormWeight
    qProj
    kProj
    vProj
    oProj
    selfAttnLayerScale
    postAttentionLayernormWeight
    gateProj
    upProj
    downProj
    mlpLayerScale
  }

private def loadUpsampleConvNeXt (weightsPath : String) (namePrefix : String) : IO UpsampleConvNeXtBlock := do
  let dwconvWeight ← loadFrozen weightsPath s!"{namePrefix}.dwconv.conv.weight" #[1024, 1, 7]
  let dwconvBias ← loadFrozen weightsPath s!"{namePrefix}.dwconv.conv.bias" #[1024]
  let normWeight ← loadFrozen weightsPath s!"{namePrefix}.norm.weight" #[1024]
  let normBias ← loadFrozen weightsPath s!"{namePrefix}.norm.bias" #[1024]
  let pwconv1Weight ← loadFrozen weightsPath s!"{namePrefix}.pwconv1.weight" #[4096, 1024]
  let pwconv1Bias ← loadFrozen weightsPath s!"{namePrefix}.pwconv1.bias" #[4096]
  let pwconv2Weight ← loadFrozen weightsPath s!"{namePrefix}.pwconv2.weight" #[1024, 4096]
  let pwconv2Bias ← loadFrozen weightsPath s!"{namePrefix}.pwconv2.bias" #[1024]
  let gamma ← loadFrozen weightsPath s!"{namePrefix}.gamma" #[1024]
  pure {
    dwconvWeight
    dwconvBias
    normWeight
    normBias
    pwconv1Weight
    pwconv1Bias
    pwconv2Weight
    pwconv2Bias
    gamma
  }

private def loadUpsampleStage (weightsPath : String) (idx : Nat) : IO UpsampleStage := do
  let p := s!"decoder.upsample.{idx}"
  let transConvWeight ← loadFrozen weightsPath s!"{p}.0.conv.weight" #[1024, 1024, 2]
  let transConvBias ← loadFrozen weightsPath s!"{p}.0.conv.bias" #[1024]
  let convNeXt ← loadUpsampleConvNeXt weightsPath s!"{p}.1"
  pure { transConvWeight, transConvBias, convNeXt }

private def loadResidualUnit (weightsPath : String) (namePrefix : String) (dim : UInt64) (dilation : UInt64)
    : IO (DecoderResidualUnit dim) := do
  let act1Alpha ← loadFrozen weightsPath s!"{namePrefix}.act1.alpha" #[dim]
  let act1Beta ← loadFrozen weightsPath s!"{namePrefix}.act1.beta" #[dim]
  let conv1Weight ← loadFrozen weightsPath s!"{namePrefix}.conv1.conv.weight" #[dim, dim, 7]
  let conv1Bias ← loadFrozen weightsPath s!"{namePrefix}.conv1.conv.bias" #[dim]
  let act2Alpha ← loadFrozen weightsPath s!"{namePrefix}.act2.alpha" #[dim]
  let act2Beta ← loadFrozen weightsPath s!"{namePrefix}.act2.beta" #[dim]
  let conv2Weight ← loadFrozen weightsPath s!"{namePrefix}.conv2.conv.weight" #[dim, dim, 1]
  let conv2Bias ← loadFrozen weightsPath s!"{namePrefix}.conv2.conv.bias" #[dim]
  pure {
    conv1Dilation := dilation
    act1Alpha
    act1Beta
    conv1Weight
    conv1Bias
    act2Alpha
    act2Beta
    conv2Weight
    conv2Bias
  }

private def loadDecoderBlock
    (weightsPath : String)
    (idx : Nat)
    (inDim outDim upRate : UInt64)
    : IO (DecoderBlock inDim outDim upRate) := do
  let p := s!"decoder.decoder.{idx}.block"
  let snakeAlpha ← loadFrozen weightsPath s!"{p}.0.alpha" #[inDim]
  let snakeBeta ← loadFrozen weightsPath s!"{p}.0.beta" #[inDim]
  let transWeight ← loadFrozen weightsPath s!"{p}.1.conv.weight" #[inDim, outDim, 2 * upRate]
  let transBias ← loadFrozen weightsPath s!"{p}.1.conv.bias" #[outDim]
  let res1 ← loadResidualUnit weightsPath s!"{p}.2" outDim 1
  let res2 ← loadResidualUnit weightsPath s!"{p}.3" outDim 3
  let res3 ← loadResidualUnit weightsPath s!"{p}.4" outDim 9
  pure { snakeAlpha, snakeBeta, transWeight, transBias, res1, res2, res3 }

/-- Load decoder weights from `speech_tokenizer` directory. -/
def loadFromDir (speechTokenizerDir : String) : IO SpeechTokenizer12HzDecoder := do
  let cfg ← SpeechTokenizer12HzConfig.loadFromFile s!"{speechTokenizerDir}/config.json"
  SpeechTokenizer12HzConfig.validateSupported cfg

  let weightsPath := s!"{speechTokenizerDir}/model.safetensors"

  let rvqFirst ← loadRVQ weightsPath "decoder.quantizer.rvq_first" 1
  let rvqRest ← loadRVQ weightsPath "decoder.quantizer.rvq_rest" 15

  let preConvWeight ← loadFrozen weightsPath "decoder.pre_conv.conv.weight" #[1024, 512, 3]
  let preConvBias ← loadFrozen weightsPath "decoder.pre_conv.conv.bias" #[1024]

  let preTransformerInputProj ← loadFrozen weightsPath "decoder.pre_transformer.input_proj.weight" #[512, 1024]
  let preTransformerInputBias ← loadFrozen weightsPath "decoder.pre_transformer.input_proj.bias" #[512]

  let mut preTransformerLayers : Array DecoderTransformerLayer := #[]
  for i in [:8] do
    let layer ← loadTransformerLayer weightsPath i
    preTransformerLayers := preTransformerLayers.push layer

  let preTransformerNormWeight ← loadFrozen weightsPath "decoder.pre_transformer.norm.weight" #[512]
  let preTransformerOutputProj ← loadFrozen weightsPath "decoder.pre_transformer.output_proj.weight" #[1024, 512]
  let preTransformerOutputBias ← loadFrozen weightsPath "decoder.pre_transformer.output_proj.bias" #[1024]

  let upsample0 ← loadUpsampleStage weightsPath 0
  let upsample1 ← loadUpsampleStage weightsPath 1

  let decoderConv0Weight ← loadFrozen weightsPath "decoder.decoder.0.conv.weight" #[1536, 1024, 7]
  let decoderConv0Bias ← loadFrozen weightsPath "decoder.decoder.0.conv.bias" #[1536]

  let decoderBlock1 ← loadDecoderBlock weightsPath 1 1536 768 8
  let decoderBlock2 ← loadDecoderBlock weightsPath 2 768 384 5
  let decoderBlock3 ← loadDecoderBlock weightsPath 3 384 192 4
  let decoderBlock4 ← loadDecoderBlock weightsPath 4 192 96 3

  let finalSnakeAlpha ← loadFrozen weightsPath "decoder.decoder.5.alpha" #[96]
  let finalSnakeBeta ← loadFrozen weightsPath "decoder.decoder.5.beta" #[96]
  let finalConvWeight ← loadFrozen weightsPath "decoder.decoder.6.conv.weight" #[1, 96, 7]
  let finalConvBias ← loadFrozen weightsPath "decoder.decoder.6.conv.bias" #[1]

  pure {
    rvqFirst
    rvqRest
    preConvWeight
    preConvBias
    preTransformerInputProj
    preTransformerInputBias
    preTransformerLayers
    preTransformerNormWeight
    preTransformerOutputProj
    preTransformerOutputBias
    upsample0
    upsample1
    decoderConv0Weight
    decoderConv0Bias
    decoderBlock1
    decoderBlock2
    decoderBlock3
    decoderBlock4
    finalSnakeAlpha
    finalSnakeBeta
    finalConvWeight
    finalConvBias
    outputSampleRate := cfg.outputSampleRate
    decodeUpsampleRate := cfg.decodeUpsampleRate
  }

private def decodeCodebook {batch frames : UInt64}
    (cb : QuantizerCodebook)
    (codes : T #[batch, frames]) : T #[batch, 256, frames] :=
  let usage : T #[2048] := clampMin1d cb.clusterUsage 1e-5
  let usage2d : T #[2048, 256] := nn.expand (reshape usage #[2048, 1]) #[2048, 256]
  let embedding : T #[2048, 256] := nn.div cb.embeddingSum usage2d
  let q : T #[batch, frames, 256] := nn.embedding codes embedding
  reshape (nn.transpose q 1 2) #[batch, 256, frames]

private def decodeRVQSlice {batch groups frames : UInt64}
    (rvq : RVQDecode)
    (codes : T #[batch, groups, frames])
    (start : UInt64)
    : T #[batch, 512, frames] :=
  Id.run do
    let mut quantized : T #[batch, 256, frames] := torch.zeros #[batch, 256, frames]
    let mut off : UInt64 := 0
    for cb in rvq.codebooks do
      let idx := start + off
      let code3 : T #[batch, 1, frames] := data.slice codes 1 idx 1
      let code2 : T #[batch, frames] := reshape code3 #[batch, frames]
      let q := decodeCodebook cb code2
      quantized := quantized + q
      off := off + 1
    let out : T #[batch, 512, frames] :=
      reshape (nn.conv1d quantized rvq.outputProj 1 0 1) #[batch, 512, frames]
    out

private def decodeSplitRVQ {batch frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[batch, 16, frames]) : T #[batch, 512, frames] :=
  let first := decodeRVQSlice m.rvqFirst codes 0
  let rest := decodeRVQSlice m.rvqRest codes 1
  first + rest

private def causalConv1d {batch inC outC frames kernel : UInt64}
    (x : T #[batch, inC, frames])
    (weight : T #[outC, inC, kernel])
    (bias : T #[outC])
    (dilation : UInt64 := 1)
    : T #[batch, outC, frames] :=
  let leftPad : UInt64 := dilation * (kernel - 1)
  let left : T #[batch, inC, leftPad] := torch.zeros #[batch, inC, leftPad]
  let xPad : T #[batch, inC, frames + leftPad] := nn.cat left x 2
  let y0 : T #[batch, outC, frames] := reshape (nn.conv1d xPad weight 1 0 dilation) #[batch, outC, frames]
  addBias3d y0 bias

private def causalDepthwiseConv1d {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (weight : T #[channels, 1, 7])
    (bias : T #[channels])
    : T #[batch, channels, frames] :=
  let left : T #[batch, channels, 6] := torch.zeros #[batch, channels, 6]
  let xPad : T #[batch, channels, frames + 6] := nn.cat left x 2
  let y : T #[batch, channels, frames] :=
    reshape (nn.conv1d_group_bias xPad weight bias 1 0 1 channels) #[batch, channels, frames]
  y

private def causalTransConv1d {batch inC outC frames kernel stride rightPad : UInt64}
    (x : T #[batch, inC, frames])
    (weight : T #[inC, outC, kernel])
    (bias : T #[outC])
    : T #[batch, outC, ((frames - 1) * stride + kernel) - rightPad] :=
  let outLen : UInt64 := (frames - 1) * stride + kernel
  let y0 : T #[batch, outC, outLen] :=
    reshape (nn.conv_transpose1d_bias x weight bias stride 0 0 1) #[batch, outC, outLen]
  reshape (data.slice y0 2 0 (outLen - rightPad)) #[batch, outC, outLen - rightPad]

private def snakeBeta {batch channels frames : UInt64}
    (x : T #[batch, channels, frames])
    (alpha : T #[channels])
    (beta : T #[channels]) : T #[batch, channels, frames] :=
  let alphaE : T #[batch, channels, frames] :=
    nn.expand (reshape (nn.exp alpha) #[1, channels, 1]) #[batch, channels, frames]
  let betaE : T #[batch, channels, frames] :=
    nn.expand (reshape (nn.exp beta) #[1, channels, 1]) #[batch, channels, frames]
  let sin2 : T #[batch, channels, frames] :=
    nn.pow (nn.sin (x * alphaE)) 2.0
  let eps : T #[batch, channels, frames] := torch.full #[batch, channels, frames] 0.000000001
  let denom := betaE + eps
  let invDenom : T #[batch, channels, frames] := nn.div (torch.ones #[batch, channels, frames]) denom
  x + invDenom * sin2

private def forwardConvNeXt {batch frames : UInt64}
    (blk : UpsampleConvNeXtBlock)
    (x : T #[batch, 1024, frames]) : T #[batch, 1024, frames] :=
  let residual := x
  let h0 := causalDepthwiseConv1d x blk.dwconvWeight blk.dwconvBias
  let h1 : T #[batch, frames, 1024] := reshape (nn.transpose h0 1 2) #[batch, frames, 1024]
  let h2 : T #[batch, frames, 1024] := nn.layer_norm h1 blk.normWeight blk.normBias 1e-6
  let h3 : T #[batch, frames, 4096] := affine3d h2 blk.pwconv1Weight blk.pwconv1Bias
  let h4 : T #[batch, frames, 4096] := nn.gelu h3
  let h5 : T #[batch, frames, 1024] := affine3d h4 blk.pwconv2Weight blk.pwconv2Bias
  let h6 : T #[batch, frames, 1024] := scale3d h5 blk.gamma
  let h7 : T #[batch, 1024, frames] := reshape (nn.transpose h6 1 2) #[batch, 1024, frames]
  residual + h7

private def forwardAttention {batch seq : UInt64}
    (layer : DecoderTransformerLayer)
    (x : T #[batch, seq, 512])
    (cos sin : T #[seq, 32]) : T #[batch, seq, 512] :=
  let q0 : T #[batch, seq, 1024] := linear3d x layer.qProj
  let k0 : T #[batch, seq, 1024] := linear3d x layer.kProj
  let v0 : T #[batch, seq, 1024] := linear3d x layer.vProj

  let q : T #[batch, seq, 16, 64] := reshape q0 #[batch, seq, 16, 64]
  let k : T #[batch, seq, 16, 64] := reshape k0 #[batch, seq, 16, 64]
  let v : T #[batch, seq, 16, 64] := reshape v0 #[batch, seq, 16, 64]

  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  let qh : T #[batch, 16, seq, 64] := nn.transpose_for_attention q
  let kh : T #[batch, 16, seq, 64] := nn.transpose_for_attention k
  let vh : T #[batch, 16, seq, 64] := nn.transpose_for_attention v

  let attn : T #[batch, 16, seq, 64] :=
    nn.scaledDotProductAttentionGQAWindow qh kh vh 0.0 true true 72
  let attn : T #[batch, seq, 16, 64] := nn.transpose_from_attention attn
  let attn : T #[batch, seq, 1024] := reshape attn #[batch, seq, 1024]
  linear3d attn layer.oProj

private def forwardTransformerLayer {batch seq : UInt64}
    (layer : DecoderTransformerLayer)
    (x : T #[batch, seq, 512])
    (cos sin : T #[seq, 32])
    : T #[batch, seq, 512] :=
  let residual1 := x
  let norm1 : RMSNorm 512 := { weight := layer.inputLayernormWeight, eps := ⟨1e-5⟩ }
  let h1 : T #[batch, seq, 512] := norm1.forward3d x
  let h2 : T #[batch, seq, 512] := forwardAttention layer h1 cos sin
  let h3 : T #[batch, seq, 512] := residual1 + scale3d h2 layer.selfAttnLayerScale

  let residual2 := h3
  let norm2 : RMSNorm 512 := { weight := layer.postAttentionLayernormWeight, eps := ⟨1e-5⟩ }
  let h4 : T #[batch, seq, 512] := norm2.forward3d h3
  let gate : T #[batch, seq, 1024] := linear3d h4 layer.gateProj
  let up : T #[batch, seq, 1024] := linear3d h4 layer.upProj
  let mlpIn : T #[batch, seq, 1024] := nn.silu gate * up
  let mlpOut : T #[batch, seq, 512] := linear3d mlpIn layer.downProj
  residual2 + scale3d mlpOut layer.mlpLayerScale

private def forwardPreTransformer {batch frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (x : T #[batch, 1024, frames]) : T #[batch, 1024, frames] :=
  Id.run do
    let x0 : T #[batch, frames, 1024] := reshape (nn.transpose x 1 2) #[batch, frames, 1024]
    let mut h : T #[batch, frames, 512] := affine3d x0 m.preTransformerInputProj m.preTransformerInputBias
    let (cos, sin) := rotary.computeFreqsPure frames 64 10000.0
    for layer in m.preTransformerLayers do
      h := forwardTransformerLayer layer h cos sin
    let norm : RMSNorm 512 := { weight := m.preTransformerNormWeight, eps := ⟨1e-5⟩ }
    let hNorm : T #[batch, frames, 512] := norm.forward3d h
    let hOut : T #[batch, frames, 1024] := affine3d hNorm m.preTransformerOutputProj m.preTransformerOutputBias
    reshape (nn.transpose hOut 1 2) #[batch, 1024, frames]

private def forwardResidualUnit {batch frames dim : UInt64}
    (u : DecoderResidualUnit dim)
    (x : T #[batch, dim, frames]) : T #[batch, dim, frames] :=
  let residual := x
  let h1 := snakeBeta x u.act1Alpha u.act1Beta
  let h2 := causalConv1d h1 u.conv1Weight u.conv1Bias u.conv1Dilation
  let h3 := snakeBeta h2 u.act2Alpha u.act2Beta
  let h4 := causalConv1d h3 u.conv2Weight u.conv2Bias
  residual + h4

private def forwardDecoderBlock {batch frames inDim outDim upRate : UInt64}
    (blk : DecoderBlock inDim outDim upRate)
    (x : T #[batch, inDim, frames]) : T #[batch, outDim, frames * upRate] :=
  let h0 := snakeBeta x blk.snakeAlpha blk.snakeBeta
  let h1 : T #[batch, outDim, frames * upRate] :=
    reshape (causalTransConv1d (batch := batch) (inC := inDim) (outC := outDim)
      (frames := frames) (kernel := 2 * upRate) (stride := upRate) (rightPad := upRate)
      h0 blk.transWeight blk.transBias) #[batch, outDim, frames * upRate]
  let h2 := forwardResidualUnit blk.res1 h1
  let h3 := forwardResidualUnit blk.res2 h2
  forwardResidualUnit blk.res3 h3

private def forwardUpsampleStage {batch frames : UInt64}
    (stage : UpsampleStage)
    (x : T #[batch, 1024, frames]) : T #[batch, 1024, frames * 2] :=
  let h0 : T #[batch, 1024, frames * 2] :=
    reshape (causalTransConv1d (batch := batch) (inC := 1024) (outC := 1024)
      (frames := frames) (kernel := 2) (stride := 2) (rightPad := 0)
      x stage.transConvWeight stage.transConvBias) #[batch, 1024, frames * 2]
  forwardConvNeXt stage.convNeXt h0

/-- Decode quantizer codes `[batch, 16, frames]` to waveform `[batch, 1, frames*1920]`. -/
def decode {batch frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[batch, 16, frames]) : T #[batch, 1, frames * 1920] :=
  let q : T #[batch, 512, frames] := decodeSplitRVQ m codes
  let h0 : T #[batch, 1024, frames] := causalConv1d q m.preConvWeight m.preConvBias
  let h1 : T #[batch, 1024, frames] := forwardPreTransformer m h0

  let u0 : T #[batch, 1024, frames * 2] := forwardUpsampleStage m.upsample0 h1
  let u1 : T #[batch, 1024, frames * 4] := forwardUpsampleStage m.upsample1 u0

  let d0 : T #[batch, 1536, frames * 4] := causalConv1d u1 m.decoderConv0Weight m.decoderConv0Bias
  let d1 : T #[batch, 768, frames * 32] :=
    reshape (forwardDecoderBlock m.decoderBlock1 d0) #[batch, 768, frames * 32]
  let d2 : T #[batch, 384, frames * 160] :=
    reshape (forwardDecoderBlock m.decoderBlock2 d1) #[batch, 384, frames * 160]
  let d3 : T #[batch, 192, frames * 640] :=
    reshape (forwardDecoderBlock m.decoderBlock3 d2) #[batch, 192, frames * 640]
  let d4 : T #[batch, 96, frames * 1920] :=
    reshape (forwardDecoderBlock m.decoderBlock4 d3) #[batch, 96, frames * 1920]

  let f0 : T #[batch, 96, frames * 1920] := snakeBeta d4 m.finalSnakeAlpha m.finalSnakeBeta
  let f1 : T #[batch, 1, frames * 1920] := causalConv1d f0 m.finalConvWeight m.finalConvBias
  clampWave f1

/-- Chunked decoder for long-form/streaming inference.
    Mirrors upstream chunk strategy:
    - decode chunks of `chunkSize` codec frames
    - prepend up to `leftContextSize` previous frames for continuity
    - drop context samples from each chunk output before concatenation -/
def decodeChunked {batch frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[batch, 16, frames])
    (chunkSize : UInt64 := 300)
    (leftContextSize : UInt64 := 25)
    : T #[batch, 1, frames * 1920] :=
  if frames == 0 then
    reshape (torch.zeros #[batch, 1, 0]) #[batch, 1, frames * 1920]
  else
    Id.run do
      let mut start : UInt64 := 0
      let mut chunks : Array (T #[]) := #[]
      while start < frames do
        let remaining := frames - start
        let curChunk := if remaining < chunkSize then remaining else chunkSize
        let context := if start < leftContextSize then start else leftContextSize
        let chunkStart := start - context
        let chunkFrames := context + curChunk
        let codesChunk : T #[batch, 16, chunkFrames] := data.slice codes 2 chunkStart chunkFrames
        let wavChunk : T #[batch, 1, chunkFrames * 1920] := decode m codesChunk
        let keepStart : UInt64 := context * 1920
        let keepSamples : UInt64 := curChunk * 1920
        let wavKeep : T #[batch, 1, keepSamples] := data.slice wavChunk 2 keepStart keepSamples
        chunks := chunks.push (nn.eraseShape wavKeep)
        start := start + curChunk
      let wavDyn : T #[] := nn.cat_dyn chunks 2
      reshape wavDyn #[batch, 1, frames * 1920]

/-- Convenience decode for frame-major codec tensor `[frames, 16]`. -/
def decodeFrameMajor {frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[frames, 16]) : T #[1, 1, frames * 1920] :=
  let c0 : T #[1, frames, 16] := reshape codes #[1, frames, 16]
  let c1 : T #[1, 16, frames] := reshape (nn.transpose c0 1 2) #[1, 16, frames]
  decode m c1

/-- Chunked convenience decode for frame-major codec tensor `[frames, 16]`. -/
def decodeFrameMajorChunked {frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[frames, 16])
    (chunkSize : UInt64 := 300)
    (leftContextSize : UInt64 := 25)
    : T #[1, 1, frames * 1920] :=
  let c0 : T #[1, frames, 16] := reshape codes #[1, frames, 16]
  let c1 : T #[1, 16, frames] := reshape (nn.transpose c0 1 2) #[1, 16, frames]
  decodeChunked m c1 chunkSize leftContextSize

/-- Decode `[frames,16]` codec tensor and write mono WAV. -/
def decodeFrameMajorToWav {frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[frames, 16])
    (wavPath : String)
    : IO Unit := do
  let wav : T #[1, 1, frames * 1920] := decodeFrameMajor m codes
  data.saveWav wav wavPath m.outputSampleRate

/-- Chunked decode `[frames,16]` codec tensor and write mono WAV. -/
def decodeFrameMajorChunkedToWav {frames : UInt64}
    (m : SpeechTokenizer12HzDecoder)
    (codes : T #[frames, 16])
    (wavPath : String)
    (chunkSize : UInt64 := 300)
    (leftContextSize : UInt64 := 25)
    : IO Unit := do
  let wav : T #[1, 1, frames * 1920] := decodeFrameMajorChunked m codes chunkSize leftContextSize
  data.saveWav wav wavPath m.outputSampleRate

end SpeechTokenizer12HzDecoder

end torch.qwen3tts
