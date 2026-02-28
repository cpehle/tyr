/-
  Tyr/Model/Qwen3TTS/Talker.lean

  Lean4 port of Qwen3-TTS talker components:
  - talker text/codec decoder
  - sub-talker (codec group predictor)
  - conditional generation heads

  Note: This is a Lean-native port that preserves the architecture shape
  and interfaces, while using Tyr's existing transformer primitives.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3TTS.Config

namespace torch.qwen3tts

abbrev TalkerLayer (cfg : TalkerConfig) :=
  qwen.QwenLayer cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize

abbrev CodePredictorLayer (cfg : TalkerCodePredictorConfig) :=
  qwen.QwenLayer cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize

private def initWeight (shape : Shape) (hidden : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / hidden.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

private def getOrFirst! [Inhabited α] (xs : Array α) (i : Nat) : α :=
  match xs[i]? with
  | some x => x
  | none =>
    match xs[0]? with
    | some x => x
    | none => panic! "empty array"

private def ensureAtLeastOne (n : Nat) : Nat :=
  if n == 0 then 1 else n

/-- Main talker decoder (text + codec embeddings -> hidden states). -/
structure TalkerModel (cfg : TalkerConfig) where
  codecEmbedding : T #[cfg.vocabSize, cfg.hiddenSize]
  textEmbedding : T #[cfg.textVocabSize, cfg.textHiddenSize]
  textProjectionFc1 : T #[cfg.textHiddenSize, cfg.textHiddenSize]
  textProjectionFc1Bias : T #[cfg.textHiddenSize]
  textProjectionFc2 : T #[cfg.hiddenSize, cfg.textHiddenSize]
  textProjectionFc2Bias : T #[cfg.hiddenSize]
  layers : Array (TalkerLayer cfg)
  norm : RMSNorm cfg.hiddenSize
  deriving TensorStruct

namespace TalkerModel

def init (cfg : TalkerConfig) : IO (TalkerModel cfg) := do
  let codecEmbedding ← initWeight #[cfg.vocabSize, cfg.hiddenSize] cfg.hiddenSize
  let textEmbedding ← initWeight #[cfg.textVocabSize, cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc1 ← initWeight #[cfg.textHiddenSize, cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc1Bias ← initWeight #[cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc2 ← initWeight #[cfg.hiddenSize, cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc2Bias ← initWeight #[cfg.hiddenSize] cfg.textHiddenSize

  let mut layers := #[]
  for _ in [:cfg.numHiddenLayers.toNat] do
    let layer ← qwen.QwenLayer.init
      cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads
      cfg.headDim cfg.intermediateSize cfg.rmsNormEps
    layers := layers.push layer

  let norm := RMSNorm.init cfg.hiddenSize cfg.rmsNormEps
  pure {
    codecEmbedding
    textEmbedding
    textProjectionFc1
    textProjectionFc1Bias
    textProjectionFc2
    textProjectionFc2Bias
    layers
    norm
  }

/-- Embed codec token ids with talker codec embedding table. -/
def embedCodec {batch seq : UInt64} (m : TalkerModel cfg)
    (codecIds : T #[batch, seq]) : T #[batch, seq, cfg.hiddenSize] :=
  nn.embedding codecIds m.codecEmbedding

/-- Embed text token ids and project to talker hidden size. -/
def embedText {batch seq : UInt64} (m : TalkerModel cfg)
    (textIds : T #[batch, seq]) : T #[batch, seq, cfg.hiddenSize] :=
  let t := nn.embedding textIds m.textEmbedding
  let h := affine3d t m.textProjectionFc1 m.textProjectionFc1Bias
  let h := nn.silu h
  affine3d h m.textProjectionFc2 m.textProjectionFc2Bias

/-- Prepend codec BOS embedding to a text-conditioning embedding sequence. -/
def prependCodecBos {batch seq : UInt64} (cfg : TalkerConfig) (m : TalkerModel cfg)
    (textEmbeds : T #[batch, seq, cfg.hiddenSize]) : T #[batch, 1 + seq, cfg.hiddenSize] :=
  let bosIds : T #[batch, 1] := torch.full_int #[batch, 1] (Int64.ofNat cfg.codecBosId.toNat)
  let bosEmb : T #[batch, 1, cfg.hiddenSize] := nn.embedding bosIds m.codecEmbedding
  nn.cat bosEmb textEmbeds 1

/-- Build talker input embeddings from text token IDs. -/
def buildInputsFromText {batch textSeq : UInt64} (cfg : TalkerConfig) (m : TalkerModel cfg)
    (textIds : T #[batch, textSeq]) : T #[batch, 1 + textSeq, cfg.hiddenSize] :=
  prependCodecBos cfg m (embedText m textIds)

/-- Build talker input embeddings from instruct + text token IDs. -/
def buildInputsFromInstructText {batch instructSeq textSeq : UInt64} (cfg : TalkerConfig) (m : TalkerModel cfg)
    (instructIds : T #[batch, instructSeq])
    (textIds : T #[batch, textSeq])
    : T #[batch, 1 + (instructSeq + textSeq), cfg.hiddenSize] :=
  let instructEmb := embedText m instructIds
  let textEmb := embedText m textIds
  let merged : T #[batch, instructSeq + textSeq, cfg.hiddenSize] := nn.cat instructEmb textEmb 1
  prependCodecBos cfg m merged

/-- Forward over already-built talker input embeddings. -/
def forwardEmbeds {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerModel cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hiddenSize])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.hiddenSize] :=
  let (cos, sin) := rotary.computeFreqsPure seq cfg.headDim cfg.ropeTheta
  let hidden := match attnMask with
    | some mask =>
      m.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  m.norm.forward3d hidden

/-- Convenience forward from codec token ids. -/
def forwardCodecIds {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerModel cfg)
    (codecIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.hiddenSize] :=
  forwardEmbeds cfg m (embedCodec m codecIds) attnMask

end TalkerModel

/-- Sub-talker model that predicts residual codec groups. -/
structure TalkerCodePredictor (cfg : TalkerConfig) where
  inputEmbeddings : Array (T #[cfg.codePredictorConfig.vocabSize, cfg.hiddenSize])
  inputProjection : T #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize]
  inputProjectionBias : T #[cfg.codePredictorConfig.hiddenSize]
  layers : Array (CodePredictorLayer cfg.codePredictorConfig)
  norm : RMSNorm cfg.codePredictorConfig.hiddenSize
  lmHeads : Array (T #[cfg.codePredictorConfig.vocabSize, cfg.codePredictorConfig.hiddenSize])
  deriving TensorStruct

namespace TalkerCodePredictor

def init (cfg : TalkerConfig) : IO (TalkerCodePredictor cfg) := do
  let numCodebooks := ensureAtLeastOne (cfg.numCodeGroups.toNat - 1)

  let mut inputEmbeddings : Array (T #[cfg.codePredictorConfig.vocabSize, cfg.hiddenSize]) := #[]
  for _ in [:numCodebooks] do
    let emb ← initWeight #[cfg.codePredictorConfig.vocabSize, cfg.hiddenSize] cfg.hiddenSize
    inputEmbeddings := inputEmbeddings.push emb

  let inputProjection ← initWeight
    #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize]
    cfg.hiddenSize
  let inputProjectionBias ← initWeight #[cfg.codePredictorConfig.hiddenSize] cfg.hiddenSize

  let mut layers := #[]
  for _ in [:cfg.codePredictorConfig.numHiddenLayers.toNat] do
    let layer ← qwen.QwenLayer.init
      cfg.codePredictorConfig.hiddenSize
      cfg.codePredictorConfig.numAttentionHeads
      cfg.codePredictorConfig.numKeyValueHeads
      cfg.codePredictorConfig.headDim
      cfg.codePredictorConfig.intermediateSize
      cfg.codePredictorConfig.rmsNormEps
    layers := layers.push layer

  let norm := RMSNorm.init cfg.codePredictorConfig.hiddenSize cfg.codePredictorConfig.rmsNormEps

  let mut lmHeads : Array (T #[cfg.codePredictorConfig.vocabSize, cfg.codePredictorConfig.hiddenSize]) := #[]
  for _ in [:numCodebooks] do
    let head ← initWeight
      #[cfg.codePredictorConfig.vocabSize, cfg.codePredictorConfig.hiddenSize]
      cfg.codePredictorConfig.hiddenSize
    lmHeads := lmHeads.push head

  pure {
    inputEmbeddings
    inputProjection
    inputProjectionBias
    layers
    norm
    lmHeads
  }

/-- Embed tokens for a specific residual codebook group. -/
def embedGroupTokens {batch seq : UInt64} (m : TalkerCodePredictor cfg)
    (groupIdx : Nat) (tokenIds : T #[batch, seq]) : T #[batch, seq, cfg.hiddenSize] :=
  let emb := getOrFirst! m.inputEmbeddings groupIdx
  nn.embedding tokenIds emb

/-- Project talker hidden state to sub-talker hidden size. -/
def projectInputs {batch seq : UInt64} (m : TalkerCodePredictor cfg)
    (x : T #[batch, seq, cfg.hiddenSize])
    : T #[batch, seq, cfg.codePredictorConfig.hiddenSize] :=
  affine3d x m.inputProjection m.inputProjectionBias

/-- Forward hidden states for the sub-talker decoder stack. -/
def forwardHidden {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerCodePredictor cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hiddenSize])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.codePredictorConfig.hiddenSize] :=
  let hidden0 := projectInputs m inputsEmbeds
  let cpCfg := cfg.codePredictorConfig
  let (cos, sin) := rotary.computeFreqsPure seq cpCfg.headDim cpCfg.ropeTheta
  let hidden := match attnMask with
    | some mask =>
      m.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        hidden0
    | none =>
      m.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        hidden0
  m.norm.forward3d hidden

/-- Compute logits for one residual codebook group. -/
def logitsForGroup {batch seq : UInt64} (m : TalkerCodePredictor cfg)
    (groupIdx : Nat)
    (hidden : T #[batch, seq, cfg.codePredictorConfig.hiddenSize])
    : T #[batch, seq, cfg.codePredictorConfig.vocabSize] :=
  let head := getOrFirst! m.lmHeads groupIdx
  linear3d hidden head

end TalkerCodePredictor

/-- Talker + sub-talker generation heads. -/
structure TalkerForConditionalGeneration (cfg : TalkerConfig) where
  model : TalkerModel cfg
  codePredictor : TalkerCodePredictor cfg
  codecHead : T #[cfg.vocabSize, cfg.hiddenSize]
  deriving TensorStruct

namespace TalkerForConditionalGeneration

structure CodeGenerationOutput (batch maxFrames numCodeGroups : UInt64) where
  codes : T #[batch, maxFrames, numCodeGroups]
  lengths : Array UInt64

structure FrameGenerationOutput (batch numCodeGroups hiddenSize : UInt64) where
  codes : T #[batch, numCodeGroups]
  summedEmbedding : T #[batch, 1, hiddenSize]

def init (cfg : TalkerConfig) : IO (TalkerForConditionalGeneration cfg) := do
  let model ← TalkerModel.init cfg
  let codePredictor ← TalkerCodePredictor.init cfg
  let codecHead ← initWeight #[cfg.vocabSize, cfg.hiddenSize] cfg.hiddenSize
  pure { model, codePredictor, codecHead }

/-- Forward talker decoder and return first-codebook logits. -/
def forward {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration cfg)
    (talkerInputs : T #[batch, seq, cfg.hiddenSize])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocabSize] :=
  let hidden := TalkerModel.forwardEmbeds cfg m.model talkerInputs attnMask
  linear3d hidden m.codecHead

private def applySampling {batch vocab : UInt64}
    (logits : T #[batch, vocab])
    (temperature : Float)
    (topK : UInt64)
    (topP : Float)
    : IO (T #[batch]) := do
  let scaled :=
    if temperature == 1.0 then logits
    else mul_scalar logits (1.0 / temperature)
  let filtered :=
    if topK == 0 then scaled
    else nn.topKFilter scaled topK
  let filtered :=
    if topP >= 1.0 then filtered
    else nn.topPFilter filtered topP
  let probs := nn.softmax filtered (-1)
  let sampled ← nn.multinomial probs 1 false
  pure (reshape (nn.squeezeDim sampled (-1)) #[batch])

private def applySuppressTail {batch vocab : UInt64}
    (logits : T #[batch, vocab])
    (suppressTail : UInt64)
    (eosTokenId : UInt64)
    : T #[batch, vocab] :=
  if suppressTail == 0 then
    logits
  else
    let start : UInt64 := if suppressTail >= vocab then 0 else vocab - suppressTail
    let idx : T #[vocab] := torch.arange 0 vocab 1
    let startVec : T #[vocab] := torch.full_int #[vocab] (Int64.ofNat start.toNat)
    let endVec : T #[vocab] := torch.full_int #[vocab] (Int64.ofNat vocab.toNat)
    let geMask : T #[vocab] := torch.ge idx startVec
    let ltMask : T #[vocab] := torch.lt idx endVec
    let tailMask : T #[vocab] := torch.logical_and geMask ltMask
    let notEos : T #[vocab] := torch.logical_not (torch.eq_scalar idx (Int64.ofNat eosTokenId.toNat))
    let suppress1d : T #[vocab] := torch.logical_and tailMask notEos
    let suppress2d : T #[batch, vocab] := nn.expand (reshape suppress1d #[1, vocab]) #[batch, vocab]
    nn.masked_fill logits suppress2d (-1e9)

private def applySuppressEos {batch vocab : UInt64}
    (logits : T #[batch, vocab])
    (eosTokenId : UInt64)
    : T #[batch, vocab] :=
  let idx : T #[vocab] := torch.arange 0 vocab 1
  let eosMask1d : T #[vocab] := torch.eq_scalar idx (Int64.ofNat eosTokenId.toNat)
  let eosMask2d : T #[batch, vocab] := nn.expand (reshape eosMask1d #[1, vocab]) #[batch, vocab]
  nn.masked_fill logits eosMask2d (-1e9)

private def applyRepetitionPenalty {batch vocab histLen : UInt64}
    (logits : T #[batch, vocab])
    (history : T #[batch, histLen])
    (penalty : Float)
    : T #[batch, vocab] :=
  if histLen == 0 || penalty == 1.0 then
    logits
  else
    let gathered : T #[batch, histLen] := torch.gather logits (-1) history
    let negMask : T #[batch, histLen] := torch.lt_scalar gathered 0.0
    let penalizedNeg := mul_scalar gathered penalty
    let penalizedPos := div_scalar gathered penalty
    let penalized : T #[batch, histLen] := torch.where_ negMask penalizedNeg penalizedPos
    torch.scatter_2d logits (-1) history penalized

/-- Sample a full codec frame from the current talker hidden state `[batch,1,hidden]`. -/
private def generateFrameFromLastHidden {batch : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration cfg)
    (lastHidden : T #[batch, 1, cfg.hiddenSize])
    (historyFirstCodes : Option (Sigma fun histLen => T #[batch, histLen]) := none)
    (allowEos : Bool := true)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    : IO (FrameGenerationOutput batch cfg.numCodeGroups cfg.hiddenSize) := do
  let firstLogits3 := linear3d lastHidden m.codecHead
  let firstLogits0 : T #[batch, cfg.vocabSize] := reshape firstLogits3 #[batch, cfg.vocabSize]
  let firstLogits1 := applySuppressTail firstLogits0 suppressTail cfg.codecEosTokenId
  let firstLogits2 :=
    if allowEos then firstLogits1 else applySuppressEos firstLogits1 cfg.codecEosTokenId
  let firstLogits :=
    match historyFirstCodes with
    | some ⟨_, hist⟩ => applyRepetitionPenalty firstLogits2 hist repetitionPenalty
    | none => firstLogits2
  let firstCode ← applySampling firstLogits temperature topK topP
  let firstCode2d : T #[batch, 1] := reshape firstCode #[batch, 1]

  -- Collect per-codebook sampled tokens as `[batch, 1]` columns.
  let mut tokenCols : Array (T #[batch, 1]) := #[firstCode2d]

  -- Build autoregressive sub-talker context from (past hidden + codec embeddings).
  let firstEmb : T #[batch, 1, cfg.hiddenSize] := nn.embedding firstCode2d m.model.codecEmbedding
  let mut summedEmb : T #[batch, 1, cfg.hiddenSize] := firstEmb
  let mut cpEmbCols : Array (T #[batch, 1, cfg.hiddenSize]) := #[lastHidden, firstEmb]

  let residualGroups := cfg.numCodeGroups.toNat - 1
  for g in [:residualGroups] do
    let curLen : UInt64 := cpEmbCols.size.toUInt64
    let cpDyn := nn.cat_impl cpEmbCols 1
    let cpInputs : T #[batch, curLen, cfg.hiddenSize] := reshape cpDyn #[batch, curLen, cfg.hiddenSize]

    let cpHidden := TalkerCodePredictor.forwardHidden cfg m.codePredictor cpInputs none
    let cpLogits3 := TalkerCodePredictor.logitsForGroup m.codePredictor g cpHidden
    let cpLast := data.slice cpLogits3 1 (curLen - 1) 1
    let cpLogits : T #[batch, cfg.codePredictorConfig.vocabSize] :=
      reshape cpLast #[batch, cfg.codePredictorConfig.vocabSize]

    let nextCode ← applySampling cpLogits subtalkerTemperature subtalkerTopK subtalkerTopP
    let nextCode2d : T #[batch, 1] := reshape nextCode #[batch, 1]
    tokenCols := tokenCols.push nextCode2d

    let nextEmb := TalkerCodePredictor.embedGroupTokens m.codePredictor g nextCode2d
    cpEmbCols := cpEmbCols.push nextEmb
    summedEmb := summedEmb + nextEmb

  let frameDyn := nn.cat_impl tokenCols 1
  pure {
    codes := reshape frameDyn #[batch, cfg.numCodeGroups]
    summedEmbedding := summedEmb
  }

/-- Generate one frame of codec tokens `[batch, numCodeGroups]` from talker context.
    This mirrors the two-stage talker/sub-talker generation in Qwen3-TTS. -/
def generateFrame {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration cfg)
    (talkerInputs : T #[batch, seq, cfg.hiddenSize])
    (attnMask : Option (T #[batch, seq]) := none)
    (historyFirstCodes : Option (Sigma fun histLen => T #[batch, histLen]) := none)
    (allowEos : Bool := true)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    : IO (FrameGenerationOutput batch cfg.numCodeGroups cfg.hiddenSize) := do
  if cfg.numCodeGroups == 0 then
    return {
      codes := torch.zeros #[batch, 0]
      summedEmbedding := torch.zeros #[batch, 1, cfg.hiddenSize]
    }

  let hidden := TalkerModel.forwardEmbeds cfg m.model talkerInputs attnMask
  let lastHidden := data.slice hidden 1 (seq - 1) 1
  generateFrameFromLastHidden cfg m lastHidden historyFirstCodes allowEos
    temperature topK topP repetitionPenalty suppressTail
    subtalkerTemperature subtalkerTopK subtalkerTopP

/-- Autoregressively generate a fixed number of codec frames.
    Returns `[batch, maxFrames, numCodeGroups]`. -/
def generateCodesWithLengths {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration cfg)
    (talkerInputs : T #[batch, seq, cfg.hiddenSize])
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (trailingTextHidden : Option (Sigma fun trailingSeq => T #[batch, trailingSeq, cfg.hiddenSize]) := none)
    (ttsPadEmbed : Option (T #[batch, 1, cfg.hiddenSize]) := none)
    : IO (CodeGenerationOutput batch maxFrames cfg.numCodeGroups) := do
  if maxFrames == 0 then
    return {
      codes := torch.zeros #[batch, 0, cfg.numCodeGroups]
      lengths := Array.replicate batch.toNat 0
    }

  let padFrame : T #[batch, 1, cfg.numCodeGroups] :=
    torch.full_int #[batch, 1, cfg.numCodeGroups] (Int64.ofNat cfg.codecPadId.toNat)
  let mut frames : Array (T #[batch, 1, cfg.numCodeGroups]) := #[]
  let mut curInputsDyn : T #[] := reshape talkerInputs #[]
  let mut curSeq : UInt64 := seq
  let mut done : Array Bool := Array.replicate batch.toNat false
  let mut lengths : Array UInt64 := Array.replicate batch.toNat maxFrames
  let mut historyCols : Array (T #[batch, 1]) := #[]
  let defaultPadEmbed : T #[batch, 1, cfg.hiddenSize] := torch.zeros #[batch, 1, cfg.hiddenSize]
  let padEmbed : T #[batch, 1, cfg.hiddenSize] := ttsPadEmbed.getD defaultPadEmbed

  for step in [:maxFrames.toNat] do
    let allDone := done.foldl (fun acc d => acc && d) true
    if allDone then
      break

    let curInputs : T #[batch, curSeq, cfg.hiddenSize] := reshape curInputsDyn #[batch, curSeq, cfg.hiddenSize]
    let historyOpt :=
      if historyCols.isEmpty then
        none
      else
        let histLen : UInt64 := historyCols.size.toUInt64
        let histDyn := nn.cat_impl historyCols 1
        let hist : T #[batch, histLen] := reshape histDyn #[batch, histLen]
        some ⟨histLen, hist⟩
    let allowEos := step.toUInt64 >= minNewTokens
    let frameOut ← generateFrame cfg m curInputs none historyOpt
      allowEos temperature topK topP repetitionPenalty suppressTail
      subtalkerTemperature subtalkerTopK subtalkerTopP
    let frame := frameOut.codes
    let frame3 : T #[batch, 1, cfg.numCodeGroups] := nn.unsqueeze frame 1
    frames := frames.push frame3

    let firstCode : T #[batch, 1] := data.slice frame 1 0 1
    historyCols := historyCols.push firstCode
    let firstCode1d : T #[batch] := reshape firstCode #[batch]
    let firstCodes ← data.tensorToUInt64Array firstCode1d
    for i in [:batch.toNat] do
      let isDone := done.getD i false
      let tok := firstCodes.getD i cfg.codecEosTokenId
      if !isDone && tok == cfg.codecEosTokenId then
        done := done.set! i true
        lengths := lengths.set! i step.toUInt64

    let trailingAdd : T #[batch, 1, cfg.hiddenSize] :=
      match trailingTextHidden with
      | some ⟨trailingSeq, trailing⟩ =>
          if step.toUInt64 < trailingSeq then
            data.slice trailing 1 step.toUInt64 1
          else
            padEmbed
      | none => padEmbed
    let nextEmb : T #[batch, 1, cfg.hiddenSize] := frameOut.summedEmbedding + trailingAdd
    let nextInputs := nn.cat curInputs nextEmb 1
    curInputsDyn := reshape nextInputs #[]
    curSeq := curSeq + 1

  for _ in [frames.size:maxFrames.toNat] do
    frames := frames.push padFrame

  let outDyn := nn.cat_impl frames 1
  let codes : T #[batch, maxFrames, cfg.numCodeGroups] :=
    reshape outDyn #[batch, maxFrames, cfg.numCodeGroups]
  pure { codes, lengths }

/-- Stream codec frames one step at a time.
    `onFrame` is called for each generated frame in generation order.
    Returns per-sample EOS lengths (same convention as `generateCodesWithLengths`). -/
def streamCodes {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration cfg)
    (talkerInputs : T #[batch, seq, cfg.hiddenSize])
    (onFrame : UInt64 → T #[batch, cfg.numCodeGroups] → IO Unit)
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (trailingTextHidden : Option (Sigma fun trailingSeq => T #[batch, trailingSeq, cfg.hiddenSize]) := none)
    (ttsPadEmbed : Option (T #[batch, 1, cfg.hiddenSize]) := none)
    : IO (Array UInt64) :=
  autograd.no_grad do
    if maxFrames == 0 then
      return Array.replicate batch.toNat 0

    if m.model.layers.size != cfg.numHiddenLayers.toNat then
      throw <| IO.userError
        s!"Talker layer/cache mismatch: model has {m.model.layers.size} layers but cfg.numHiddenLayers={cfg.numHiddenLayers}."

    let cacheDevice := talkerInputs.device
    let cacheMaxLen : UInt64 := seq + maxFrames + 1
    let mut caches : Array (qwen.QwenAttention.KVCache batch cfg.numKeyValueHeads cfg.headDim) := #[]
    for _ in [:cfg.numHiddenLayers.toNat] do
      caches := caches.push
        (qwen.QwenAttention.initKVCache
          cacheMaxLen
          (batch := batch) (num_kv_heads := cfg.numKeyValueHeads) (head_dim := cfg.headDim)
          cacheDevice)

    -- Prefill KV cache from prompt/context so per-frame generation only runs one token step.
    let (cosAll, sinAll) := rotary.computeFreqsPure (seq + maxFrames + 1) cfg.headDim cfg.ropeTheta
    let mut lastHidden : T #[batch, 1, cfg.hiddenSize] := torch.zeros #[batch, 1, cfg.hiddenSize] false cacheDevice
    let mut pos : UInt64 := 0
    while pos < seq do
      let xIn : T #[batch, 1, cfg.hiddenSize] := data.slice talkerInputs 1 pos 1
      let cos : T #[1, cfg.headDim / 2] := data.slice cosAll 0 pos 1
      let sin : T #[1, cfg.headDim / 2] := data.slice sinAll 0 pos 1
      let mut h : T #[batch, 1, cfg.hiddenSize] := xIn
      for i in [:m.model.layers.size] do
        let layer : TalkerLayer cfg ←
          match m.model.layers[i]? with
          | some layer => pure layer
          | none => throw <| IO.userError s!"missing talker layer at index {i}"
        let cache : qwen.QwenAttention.KVCache batch cfg.numKeyValueHeads cfg.headDim ←
          match caches[i]? with
          | some cache => pure cache
          | none => throw <| IO.userError s!"missing talker cache at index {i}"
        let (h', cache') := qwen.QwenLayer.forwardStep layer h cos sin cache
        h := h'
        caches := caches.set! i cache'
      lastHidden := m.model.norm.forward3d h
      pos := pos + 1

    let mut done : Array Bool := Array.replicate batch.toNat false
    let mut lengths : Array UInt64 := Array.replicate batch.toNat maxFrames
    let mut historyCols : Array (T #[batch, 1]) := #[]
    let defaultPadEmbed : T #[batch, 1, cfg.hiddenSize] := torch.zeros #[batch, 1, cfg.hiddenSize] false cacheDevice
    let padEmbed : T #[batch, 1, cfg.hiddenSize] := ttsPadEmbed.getD defaultPadEmbed

    for step in [:maxFrames.toNat] do
      let allDone := done.foldl (fun acc d => acc && d) true
      if allDone then
        break

      let historyOpt :=
        if historyCols.isEmpty then
          none
        else
          let histLen : UInt64 := historyCols.size.toUInt64
          let histDyn := nn.cat_impl historyCols 1
          let hist : T #[batch, histLen] := reshape histDyn #[batch, histLen]
          some ⟨histLen, hist⟩
      let allowEos := step.toUInt64 >= minNewTokens
      let frameOut ← generateFrameFromLastHidden cfg m lastHidden historyOpt
        allowEos temperature topK topP repetitionPenalty suppressTail
        subtalkerTemperature subtalkerTopK subtalkerTopP
      let frame := frameOut.codes
      onFrame step.toUInt64 frame

      let firstCode : T #[batch, 1] := data.slice frame 1 0 1
      historyCols := historyCols.push firstCode
      let firstCode1d : T #[batch] := reshape firstCode #[batch]
      let firstCodes ← data.tensorToUInt64Array firstCode1d
      for i in [:batch.toNat] do
        let isDone := done.getD i false
        let tok := firstCodes.getD i cfg.codecEosTokenId
        if !isDone && tok == cfg.codecEosTokenId then
          done := done.set! i true
          lengths := lengths.set! i step.toUInt64

      let trailingAdd : T #[batch, 1, cfg.hiddenSize] :=
        match trailingTextHidden with
        | some ⟨trailingSeq, trailing⟩ =>
            if step.toUInt64 < trailingSeq then
              data.slice trailing 1 step.toUInt64 1
            else
              padEmbed
        | none => padEmbed
      let nextEmb : T #[batch, 1, cfg.hiddenSize] := frameOut.summedEmbedding + trailingAdd
      let nextPos : UInt64 := seq + step.toUInt64
      let cos : T #[1, cfg.headDim / 2] := data.slice cosAll 0 nextPos 1
      let sin : T #[1, cfg.headDim / 2] := data.slice sinAll 0 nextPos 1
      let mut h : T #[batch, 1, cfg.hiddenSize] := nextEmb
      for i in [:m.model.layers.size] do
        let layer : TalkerLayer cfg ←
          match m.model.layers[i]? with
          | some layer => pure layer
          | none => throw <| IO.userError s!"missing talker layer at index {i}"
        let cache : qwen.QwenAttention.KVCache batch cfg.numKeyValueHeads cfg.headDim ←
          match caches[i]? with
          | some cache => pure cache
          | none => throw <| IO.userError s!"missing talker cache at index {i}"
        let (h', cache') := qwen.QwenLayer.forwardStep layer h cos sin cache
        h := h'
        caches := caches.set! i cache'
      lastHidden := m.model.norm.forward3d h

    pure lengths

/-- Autoregressively generate a fixed number of codec frames.
    Returns `[batch, maxFrames, numCodeGroups]`. -/
def generateCodes {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration cfg)
    (talkerInputs : T #[batch, seq, cfg.hiddenSize])
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (trailingTextHidden : Option (Sigma fun trailingSeq => T #[batch, trailingSeq, cfg.hiddenSize]) := none)
    (ttsPadEmbed : Option (T #[batch, 1, cfg.hiddenSize]) := none)
    : IO (T #[batch, maxFrames, cfg.numCodeGroups]) := do
  let out ← generateCodesWithLengths cfg m talkerInputs maxFrames minNewTokens
    temperature topK topP
    subtalkerTemperature subtalkerTopK subtalkerTopP
    repetitionPenalty suppressTail
    trailingTextHidden ttsPadEmbed
  pure out.codes

end TalkerForConditionalGeneration

end torch.qwen3tts
