/-
  Tyr/Model/Whisper/Transcribe.lean

  Offline Whisper transcription pipeline in Tyr.
-/
import Tyr.Model.Whisper.ConfigIO
import Tyr.Model.Whisper.Model
import Tyr.Model.Whisper.Weights
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Tyr.Tokenizer.Qwen3

namespace torch.whisper

open torch.qwen3asr

structure WhisperTranscription where
  language : String
  text : String
  tokenIds : Array UInt32 := #[]
  deriving Repr, Inhabited

structure WhisperBundle where
  cfg : WhisperConfig
  model : WhisperForConditionalGeneration cfg
  tok : tokenizer.qwen3.QwenTokenizer
  preprocessor : PreprocessorConfig

private def sampleRate16k : UInt64 := 16000

private def normalizeLanguageCode (language : String) : String :=
  let l := language.trimAscii.toString.toLower
  if l.isEmpty then
    "en"
  else if l == "english" then
    "en"
  else if l == "chinese" then
    "zh"
  else if l == "japanese" then
    "ja"
  else if l == "korean" then
    "ko"
  else
    l

private def languageTokenString (language : String) : String :=
  let code := normalizeLanguageCode language
  if code.startsWith "<|" && code.endsWith "|>" then
    code
  else
    s!"<|{code}|>"

private def tokenIdByText? (tok : tokenizer.qwen3.QwenTokenizer) (text : String) : Option UInt32 :=
  tok.specialTokens.get? text <|> tok.tokenToId.get? text

private def buildPromptIds
    (cfg : WhisperConfig)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (language : String)
    (noTimestamps : Bool := true)
    : Array UInt32 :=
  Id.run do
    let startId :=
      (tokenIdByText? tok "<|startoftranscript|>").getD cfg.decoderStartTokenId.toUInt32
    let mut out : Array UInt32 := #[startId]
    let langTok := languageTokenString language
    if let some langId := tokenIdByText? tok langTok then
      out := out.push langId
    if let some transcribeId := tokenIdByText? tok "<|transcribe|>" then
      out := out.push transcribeId
    if noTimestamps then
      if let some noTsId := tokenIdByText? tok "<|notimestamps|>" then
        out := out.push noTsId
    out

private def decodeGeneratedText
    (tok : tokenizer.qwen3.QwenTokenizer)
    (ids : Array UInt32)
    : String :=
  let filtered := ids.filter (fun id => !tok.idToSpecial.contains id)
  (tokenizer.qwen3.decodeText tok filtered).trimAscii.toString

private def mergeChunkTexts (parts : Array String) : String :=
  Id.run do
    let mut out := ""
    for p in parts do
      let t := p.trimAscii.toString
      if !t.isEmpty then
        if out.isEmpty then
          out := t
        else
          out := out ++ " " ++ t
    out

private def splitWaveformFixed
    (wav : Array Float)
    (chunkSamples : Nat)
    : Array (Array Float) :=
  if chunkSamples == 0 || wav.size <= chunkSamples then
    #[wav]
  else
    Id.run do
      let mut out : Array (Array Float) := #[]
      let mut start : Nat := 0
      while start < wav.size do
        let stop := Nat.min wav.size (start + chunkSamples)
        out := out.push (wav.extract start stop)
        start := stop
      out

private def whisperMaxChunkSeconds
    (cfg : WhisperConfig)
    (pre : PreprocessorConfig)
    : Float :=
  let sr := if pre.samplingRate == 0 then sampleRate16k else pre.samplingRate
  let hop := if pre.hopLength == 0 then 160 else pre.hopLength
  let maxFrames := cfg.maxSourcePositions * 2
  (maxFrames * hop).toFloat / sr.toFloat

private partial def greedyDecodeLoop
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (eosTokenId : UInt64)
    (remaining : Nat)
    (allIds : Array UInt32)
    (generated : Array UInt32)
    : IO (Array UInt32 × Array UInt32) := do
  if remaining == 0 then
    pure (allIds, generated)
  else
    let seq : UInt64 := allIds.size.toUInt64
    if seq == 0 then
      pure (allIds, generated)
    else
      let idsVals : Array Int64 := allIds.map (fun id => Int64.ofNat id.toNat)
      let inputIdsCpu : T #[1, seq] := reshape (data.fromInt64Array idsVals) #[1, seq]
      let inputIds : T #[1, seq] :=
        if inputIdsCpu.device == encoderHidden.device then
          inputIdsCpu
        else
          inputIdsCpu.to encoderHidden.device
      let logits ← model.decode inputIds encoderHidden
      let last : T #[1, cfg.vocabSize] :=
        reshape (data.slice logits 1 (seq - 1) 1) #[1, cfg.vocabSize]
      let nextTokRaw : T #[1] := nn.argmax last 1
      let nextVals ← data.tensorToUInt64Array nextTokRaw
      let nextIdU64 := nextVals.getD 0 eosTokenId
      let nextId := nextIdU64.toUInt32
      let allIds' := allIds.push nextId
      if nextIdU64 == eosTokenId then
        pure (allIds', generated)
      else
        greedyDecodeLoop model encoderHidden eosTokenId (remaining - 1) allIds' (generated.push nextId)

def loadFromPretrainedDir (modelDir : String) : IO WhisperBundle := do
  let cfg ← WhisperConfig.loadFromPretrainedDir modelDir {}
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let preprocessor ←
    PreprocessorConfig.loadFromPretrainedDir modelDir {
      featureSize := cfg.numMelBins
      samplingRate := 16000
      hopLength := 160
      chunkLength := 30
      nFft := 400
      returnAttentionMask := true
      doNormalize := false
      dither := 0.0
    }
  let model ← WhisperForConditionalGeneration.loadSharded modelDir cfg
  pure { cfg, model, tok, preprocessor }

def transcribeWav
    {cfg : WhisperConfig}
    (model : WhisperForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : PreprocessorConfig)
    (wavPath : String)
    (language : String := "en")
    (maxNewTokens : UInt64 := 128)
    (noTimestamps : Bool := true)
    : IO WhisperTranscription := do
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  let frontendDevice := model.model.decoderTokenEmbedding.device
  let cfgCapSec := whisperMaxChunkSeconds cfg pre
  let preChunkSec :=
    if pre.chunkLength == 0 then
      cfgCapSec
    else
      pre.chunkLength.toFloat
  let chunkSeconds :=
    if preChunkSec <= 0.0 then
      cfgCapSec
    else
      if preChunkSec <= cfgCapSec then preChunkSec else cfgCapSec
  let sr := if pre.samplingRate == 0 then sampleRate16k else pre.samplingRate
  let chunkSamplesRaw := ((chunkSeconds * sr.toFloat) + 0.5).toUInt64.toNat
  let chunkSamples := if chunkSamplesRaw == 0 then wav16k.size else chunkSamplesRaw
  let chunks := splitWaveformFixed wav16k chunkSamples

  let mut texts : Array String := #[]
  let mut allTokenIds : Array UInt32 := #[]
  for chunk in chunks do
    let pack ←
      waveformToWhisperFeaturesDynamic
        pre
        chunk
        (minSeconds := 0.05)
        (maxSeconds := some chunkSeconds)
        (device := frontendDevice)
    match pack with
    | ⟨frames, out⟩ =>
      let inputFeatures : T #[1, cfg.numMelBins, frames] :=
        reshape (nn.eraseShape out.inputFeatures) #[1, cfg.numMelBins, frames]
      let encoderHidden : T #[1, WhisperConfig.conv2OutputSeq frames, cfg.dModel] ←
        model.encode (frames := frames) inputFeatures
      let promptIds := buildPromptIds cfg tok language noTimestamps
      let (_allIds, generated) ←
        greedyDecodeLoop
          (encSeq := WhisperConfig.conv2OutputSeq frames)
          model
          encoderHidden
          cfg.eosTokenId
          maxNewTokens.toNat
          promptIds
          #[]
      let text := decodeGeneratedText tok generated
      if !(text.trimAscii.toString).isEmpty then
        texts := texts.push text
      allTokenIds := allTokenIds ++ generated

  pure {
    language := normalizeLanguageCode language
    text := mergeChunkTexts texts
    tokenIds := allTokenIds
  }

end torch.whisper
