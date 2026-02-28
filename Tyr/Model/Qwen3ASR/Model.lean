/-
  Tyr/Model/Qwen3ASR/Model.lean

  Lean-native Qwen3-ASR thinker + top-level model:
  - audio tower
  - text decoder
  - LM head (ASR / forced-aligner shape aware)
  - rope-delta helpers + greedy generation utilities
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.AudioEncoder

namespace torch.qwen3asr

abbrev TextQwenConfig (cfg : ThinkerConfig) : qwen.QwenConfig :=
  TextConfig.toQwenConfig cfg.textConfig

abbrev ThinkerLmVocabSize (cfg : ThinkerConfig) : UInt64 :=
  ThinkerConfig.lmHeadOutDim cfg

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

private def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

private def buildLengthMask {batch maxLen : UInt64} (lengths : Array UInt64) : T #[batch, maxLen] :=
  Id.run do
    let mut vals : Array Int64 := #[]
    for i in [:batch.toNat] do
      let len := lengths.getD i 0
      for j in [:maxLen.toNat] do
        let v : Int64 := if j.toUInt64 < len then 1 else 0
        vals := vals.push v
    reshape (data.fromInt64Array vals) #[batch, maxLen]

private def maskRowCounts {batch seq : UInt64} (attentionMask : T #[batch, seq]) : IO (Array UInt64) := do
  let flat : T #[batch * seq] := reshape (data.toLong attentionMask) #[batch * seq]
  let vals ← data.tensorToUInt64Array flat
  let mut counts : Array UInt64 := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let mut c : UInt64 := 0
    for t in [:seq.toNat] do
      if vals.getD (b * seq.toNat + t) 0 != 0 then
        c := c + 1
    counts := counts.push c
  pure counts

private def allInSet (xs : Array UInt64) (allow : Array UInt64) : Bool :=
  xs.all (fun x => allow.contains x)

/-- Output container mirroring `...CausalLMOutputWithPast` fields that are
    currently represented in the Lean port. -/
structure Qwen3ASRThinkerCausalLMOutput (cfg : ThinkerConfig) (batch seq : UInt64) where
  logits : T #[batch, seq, ThinkerLmVocabSize cfg]
  ropeDeltas : Option (T #[batch, 1]) := none

/-- Small Lean equivalent of generation input preparation payload. -/
structure ThinkerGenerationInputs (cfg : ThinkerConfig) (batch seq frames : UInt64) where
  inputIds : T #[batch, seq]
  inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none
  featureAttentionMask : Option (T #[batch, frames]) := none

/-- Thinker model: audio encoder + text decoder + LM head. -/
structure Qwen3ASRThinkerForConditionalGeneration (cfg : ThinkerConfig) where
  audioTower : AudioEncoder cfg.audioConfig
  textModel : qwen.Qwen3Model (TextQwenConfig cfg)
  audioProjectionWeight : T #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim]
  audioProjectionBias : T #[cfg.textConfig.hiddenSize]
  lmHead : T #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize]
  deriving TensorStruct

namespace Qwen3ASRThinkerForConditionalGeneration

def init (cfg : ThinkerConfig) : IO (Qwen3ASRThinkerForConditionalGeneration cfg) := do
  let audioTower ← AudioEncoder.init cfg.audioConfig
  let textModel ← qwen.Qwen3Model.init (TextQwenConfig cfg)
  let audioProjectionWeight ←
    if h : cfg.audioConfig.outputDim = cfg.textConfig.hiddenSize then
      let eye : T #[cfg.textConfig.hiddenSize, cfg.textConfig.hiddenSize] :=
        autograd.set_requires_grad (torch.eye cfg.textConfig.hiddenSize false) true
      let eye' : T #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] := by
        simpa [h] using eye
      pure eye'
    else
      throw <| IO.userError
        s!"Qwen3-ASR expects audio output dim == text hidden dim, got {cfg.audioConfig.outputDim} and {cfg.textConfig.hiddenSize}"
  let audioProjectionBias := initBias #[cfg.textConfig.hiddenSize]
  let lmHead ← initWeight #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize] cfg.textConfig.hiddenSize
  pure { audioTower, textModel, audioProjectionWeight, audioProjectionBias, lmHead }

def embedText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    : T #[batch, textSeq, cfg.textConfig.hiddenSize] :=
  nn.embedding textIds m.textModel.embed_tokens

def projectAudio {batch audioSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (audioFeatures : T #[batch, audioSeq, cfg.audioConfig.outputDim])
    : T #[batch, audioSeq, cfg.textConfig.hiddenSize] :=
  affine3d audioFeatures m.audioProjectionWeight m.audioProjectionBias

def buildInputsFromText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    : T #[batch, textSeq, cfg.textConfig.hiddenSize] :=
  embedText m textIds

def buildInputsFromAudioAndText {batch audioSeq textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (audioFeatures : T #[batch, audioSeq, cfg.audioConfig.outputDim])
    (textIds : T #[batch, textSeq])
    : T #[batch, audioSeq + textSeq, cfg.textConfig.hiddenSize] :=
  let audioEmb := projectAudio m audioFeatures
  let textEmb := embedText m textIds
  nn.cat audioEmb textEmb 1

def forwardEmbeds {batch seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputsEmbeds : T #[batch, seq, cfg.textConfig.hiddenSize])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, ThinkerLmVocabSize cfg] :=
  let (cos, sin) := rotary.computeFreqsPure seq cfg.textConfig.headDim cfg.textConfig.ropeTheta
  let hidden := match attnMask with
    | some mask =>
      m.textModel.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.textModel.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  let hidden := m.textModel.norm.forward3d hidden
  linear3d hidden m.lmHead

def forwardText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, textSeq]) := none)
    : T #[batch, textSeq, ThinkerLmVocabSize cfg] :=
  forwardEmbeds m (buildInputsFromText m textIds) attnMask

def forwardAudioText {batch audioSeq textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (audioFeatures : T #[batch, audioSeq, cfg.audioConfig.outputDim])
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, audioSeq + textSeq]) := none)
    : T #[batch, audioSeq + textSeq, ThinkerLmVocabSize cfg] :=
  forwardEmbeds m (buildInputsFromAudioAndText m audioFeatures textIds) attnMask

def encodeAudio {batch frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.audioConfig.numMelBins, frames])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames, cfg.audioConfig.outputDim] :=
  m.audioTower.forward inputFeatures attnMask

private def getPlaceholderMask (cfg : ThinkerConfig) {batch seq : UInt64}
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.textConfig.hiddenSize] :=
  let mask2d : T #[batch, seq] := eq_scalar inputIds (Int64.ofNat cfg.audioTokenId.toNat)
  nn.expand (reshape mask2d #[batch, seq, 1]) #[batch, seq, cfg.textConfig.hiddenSize]

private def audioValidMaskFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq audioSeq : UInt64}
    (featureAttentionMask : T #[batch, featureSeq])
    : IO (T #[batch, audioSeq, cfg.textConfig.hiddenSize]) := do
  let featureLensTensor : T #[batch] := nn.sumDim (data.toLong featureAttentionMask) 1 false
  let featureLens ← data.tensorToUInt64Array featureLensTensor
  let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
  let valid2d : T #[batch, audioSeq] := buildLengthMask audioLens
  let valid2dBool : T #[batch, audioSeq] := eq_scalar valid2d 1
  pure <| nn.expand (reshape valid2dBool #[batch, audioSeq, 1]) #[batch, audioSeq, cfg.textConfig.hiddenSize]

/-- Port of reference `get_rope_index` calculation from attention masks. -/
def getRopeIndex {batch seq : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (attentionMask : T #[batch, seq])
    : IO (T #[3, batch, seq] × T #[batch, 1]) := do
  let flat : T #[batch * seq] := reshape (data.toLong attentionMask) #[batch * seq]
  let vals ← data.tensorToUInt64Array flat

  let mut posVals : Array Int64 := Array.mkEmpty (3 * batch.toNat * seq.toNat)
  let mut deltaVals : Array Int64 := Array.mkEmpty batch.toNat

  for b in [:batch.toNat] do
    let mut row : Array UInt64 := Array.mkEmpty seq.toNat
    let mut running : UInt64 := 0
    let mut maxPos : UInt64 := 1
    let mut validCount : UInt64 := 0

    for t in [:seq.toNat] do
      let v := vals.getD (b * seq.toNat + t) 0
      if v == 0 then
        row := row.push 1
      else
        let p := running
        row := row.push p
        running := running + 1
        validCount := validCount + 1
        if p > maxPos then
          maxPos := p

    let delta := (maxPos + 1) - validCount
    deltaVals := deltaVals.push (Int64.ofNat delta.toNat)

    for _ in [:3] do
      for p in row do
        posVals := posVals.push (Int64.ofNat p.toNat)

  let positionIds : T #[3, batch, seq] := reshape (data.fromInt64Array posVals) #[3, batch, seq]
  let ropeDeltas : T #[batch, 1] := reshape (data.fromInt64Array deltaVals) #[batch, 1]
  pure (positionIds, ropeDeltas)

/-- Lean analogue of `prepare_inputs_for_generation` behavior for audio features:
    only feed audio on the first generation step (`cachePosition == 0`). -/
def prepareInputsForGeneration {batch seq frames : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (cachePosition : UInt64 := 0)
    : ThinkerGenerationInputs cfg batch seq frames :=
  if cachePosition == 0 then
    { inputIds, inputFeatures, featureAttentionMask }
  else
    { inputIds, inputFeatures := none, featureAttentionMask }

/-- Faithful thinker forward shape: text embeddings with in-place audio placeholder replacement. -/
def forwardWithAux {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (Qwen3ASRThinkerCausalLMOutput cfg batch seq) := do
  let inputsEmbeds0 := embedText m inputIds
  let inputsEmbeds ←
    match inputFeatures with
    | none => pure inputsEmbeds0
    | some feats =>
      let audioSeq := AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames
      let audioFeatures := projectAudio m (encodeAudio m feats)
      let source ←
        match featureAttentionMask with
        | some fm =>
          let validMask ← audioValidMaskFromFeatureMask cfg (audioSeq := audioSeq) fm
          pure (nn.masked_select audioFeatures validMask)
        | none =>
          pure (reshape audioFeatures #[batch * audioSeq * cfg.textConfig.hiddenSize])
      let placeholderMask := getPlaceholderMask cfg inputIds
      pure (nn.masked_scatter inputsEmbeds0 placeholderMask source)

  let logits := forwardEmbeds m inputsEmbeds attentionMask

  let ropeDeltas ←
    match attentionMask with
    | none => pure none
    | some mask =>
      let (_positionIds, ropeRaw) ← m.getRopeIndex mask
      let rawVals : Array UInt64 ←
        data.tensorToUInt64Array (reshape (data.toLong ropeRaw) #[batch])
      let validCounts ← maskRowCounts mask
      let mut out : Array Int64 := Array.mkEmpty batch.toNat
      for b in [:batch.toNat] do
        let raw := rawVals.getD b 0
        let valid := validCounts.getD b 0
        let pad := seq - valid
        let adj := if raw >= pad then raw - pad else 0
        out := out.push (Int64.ofNat adj.toNat)
      let deltas : T #[batch, 1] := reshape (data.fromInt64Array out) #[batch, 1]
      pure (some deltas)

  pure { logits, ropeDeltas }

def forward {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, ThinkerLmVocabSize cfg]) := do
  return (← m.forwardWithAux inputIds inputFeatures featureAttentionMask attentionMask).logits

private partial def greedyLoop {batch frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]))
    (featureAttentionMask : Option (T #[batch, frames]))
    (eosTokenIds : Array UInt64)
    (remaining : Nat)
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"

  let logits ← m.forward curIds inputFeatures featureAttentionMask none
  let lastPos := curSeq - 1
  let last3 : T #[batch, 1, ThinkerLmVocabSize cfg] :=
    reshape (data.slice logits 1 lastPos 1) #[batch, 1, ThinkerLmVocabSize cfg]
  let last2 : T #[batch, ThinkerLmVocabSize cfg] := reshape last3 #[batch, ThinkerLmVocabSize cfg]
  let nextTok : T #[batch] := nn.argmax last2 1
  let nextVals ← data.tensorToUInt64Array nextTok

  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    greedyLoop m inputFeatures featureAttentionMask eosTokenIds (remaining - 1) appended

/-- Greedy generation utility (Lean-only, no vLLM dependency).
    Returns generated full sequence `[prompt || new_tokens]` with dynamic output length. -/
def generateGreedy {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  greedyLoop m inputFeatures featureAttentionMask eosTokenIds maxNewTokens.toNat inputIds

def forwardFromMel {batch frames textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.audioConfig.numMelBins, frames])
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames + textSeq]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames + textSeq, ThinkerLmVocabSize cfg] :=
  let audio := m.encodeAudio inputFeatures
  m.forwardAudioText audio textIds attnMask

end Qwen3ASRThinkerForConditionalGeneration

/-- Top-level Qwen3-ASR model wrapper (thinker-only inference in this Lean port). -/
structure Qwen3ASRForConditionalGeneration (cfg : Qwen3ASRConfig) where
  thinker : Qwen3ASRThinkerForConditionalGeneration cfg.thinkerConfig
  supportLanguages : Array String := cfg.supportLanguages
  deriving TensorStruct

namespace Qwen3ASRForConditionalGeneration

def init (cfg : Qwen3ASRConfig) : IO (Qwen3ASRForConditionalGeneration cfg) := do
  let thinker ← Qwen3ASRThinkerForConditionalGeneration.init cfg.thinkerConfig
  pure { thinker, supportLanguages := cfg.supportLanguages }

def getSupportedLanguages (m : Qwen3ASRForConditionalGeneration cfg) : Array String :=
  m.supportLanguages

def forwardText {batch textSeq : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, textSeq]) := none)
    : T #[batch, textSeq, ThinkerLmVocabSize cfg.thinkerConfig] :=
  m.thinker.forwardText textIds attnMask

def forwardFromMel {batch frames textSeq : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputFeatures : T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames])
    (textIds : T #[batch, textSeq])
    (attnMask : Option (T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq]) := none)
    : T #[batch, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq, ThinkerLmVocabSize cfg.thinkerConfig] :=
  m.thinker.forwardFromMel inputFeatures textIds attnMask

def forwardWithAux {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (Qwen3ASRThinkerCausalLMOutput cfg.thinkerConfig batch seq) :=
  m.thinker.forwardWithAux inputIds inputFeatures featureAttentionMask attentionMask

def forward {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (attentionMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, ThinkerLmVocabSize cfg.thinkerConfig]) :=
  m.thinker.forward inputIds inputFeatures featureAttentionMask attentionMask

def generateGreedy {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (inputFeatures : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]) := none)
    (featureAttentionMask : Option (T #[batch, frames]) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  m.thinker.generateGreedy inputIds inputFeatures featureAttentionMask maxNewTokens eosTokenIds

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
