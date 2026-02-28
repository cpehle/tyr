/-  Tyr/Model/Qwen3ASR/Transcribe.lean

  Lean-native offline ASR orchestration:
  - build prompt
  - run frontend + generation
  - parse ASR output
  - optional forced-aligner timestamps
-/
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.Model
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.Processor
import Tyr.Model.Qwen3ASR.ForcedAligner
import Tyr.Model.Qwen3ASR.Streaming
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

/-- One offline ASR transcription result. -/
structure ASRTranscription where
  language : String
  text : String
  timeStamps : Option ForcedAlignResult := none
  deriving Repr, Inhabited

private def sampleRate16k : UInt64 := 16000
private def minAsrInputSeconds : Float := 0.5
private def maxAsrInputSeconds : Float := 1200.0
private def maxForceAlignInputSeconds : Float := 180.0

private def padRightWithZeros (xs : Array Float) (target : Nat) : Array Float :=
  if xs.size >= target then
    xs
  else
    Id.run do
      let mut out := xs
      let pad := target - xs.size
      for _ in [:pad] do
        out := out.push 0.0
      out

private def argminFloatArray (xs : Array Float) : Nat :=
  if xs.isEmpty then
    0
  else
    Id.run do
      let mut bestIdx : Nat := 0
      let mut bestVal := xs[0]!
      for i in [1:xs.size] do
        let v := xs[i]!
        if v < bestVal then
          bestVal := v
          bestIdx := i
      bestIdx

private def slidingWindowArgminAbs (xs : Array Float) (win : Nat) : Nat :=
  if xs.isEmpty || win == 0 || xs.size <= win then
    0
  else
    Id.run do
      let mut cur : Float := 0.0
      for i in [:win] do
        cur := cur + Float.abs (xs[i]!)
      let mut bestVal := cur
      let mut bestPos : Nat := 0
      let limit := xs.size - win + 1
      for pos in [1:limit] do
        let dropV := Float.abs (xs[pos - 1]!)
        let addV := Float.abs (xs[pos + win - 1]!)
        cur := cur - dropV + addV
        if cur < bestVal then
          bestVal := cur
          bestPos := pos
      bestPos

private def splitAudioIntoChunks
    (wav : Array Float)
    (sr : UInt64)
    (maxChunkSec : Float)
    (searchExpandSec : Float := 5.0)
    (minWindowMs : Float := 100.0)
    : Array (Array Float × Float) :=
  let wav := wav
  let srNat := if sr == 0 then sampleRate16k.toNat else sr.toNat
  if srNat == 0 then
    #[(wav, 0.0)]
  else
    let totalLen := wav.size
    let totalSec := totalLen.toFloat / srNat.toFloat
    if totalSec <= maxChunkSec then
      #[(wav, 0.0)]
    else
      Id.run do
        let maxLenRaw := ((maxChunkSec * srNat.toFloat) + 0.5).toUInt64.toNat
        let maxLen := if maxLenRaw == 0 then 1 else maxLenRaw
        let expandRaw := ((searchExpandSec * srNat.toFloat) + 0.5).toUInt64.toNat
        let expand := if expandRaw == 0 then 1 else expandRaw
        let winRaw := (((minWindowMs / 1000.0) * srNat.toFloat) + 0.5).toUInt64.toNat
        let win := Nat.max 4 winRaw

        let mut chunks : Array (Array Float × Float) := #[]
        let mut start : Nat := 0
        let mut offsetSec : Float := 0.0

        while (totalLen - start) > maxLen do
          let cut := start + maxLen
          let left := Nat.max start (cut - expand)
          let right := Nat.min totalLen (cut + expand)
          let boundary0 :=
            if right - left <= win then
              cut
            else
              Id.run do
                let seg := wav.extract left right
                let minPos := slidingWindowArgminAbs seg win
                let wstart := minPos
                let wend := minPos + win
                let localAbs := (seg.extract wstart wend).map (fun x => Float.abs x)
                pure (left + wstart + argminFloatArray localAbs)

          let boundary1 := Nat.max boundary0 (start + 1)
          let boundary := Nat.min boundary1 totalLen
          let chunk := wav.extract start boundary
          chunks := chunks.push (chunk, offsetSec)
          offsetSec := offsetSec + (boundary - start).toFloat / srNat.toFloat
          start := boundary

        let tail := wav.extract start totalLen
        chunks := chunks.push (tail, offsetSec)

        let minLenRaw := ((minAsrInputSeconds * srNat.toFloat) + 0.5).toUInt64.toNat
        if minLenRaw == 0 then
          chunks
        else
          chunks.map (fun (c, off) =>
            if c.size < minLenRaw then
              (padRightWithZeros c minLenRaw, off)
            else
              (c, off))

private def offsetForcedAlignResult (r : ForcedAlignResult) (offsetSec : Float) : ForcedAlignResult :=
  { items := r.items.map (fun it =>
      { it with
        startTime := it.startTime + offsetSec
        endTime := it.endTime + offsetSec
      }) }

private def mergeChunkTexts (parts : Array String) : String := Id.run do
  let mut out := ""
  for p in parts do
    out := out ++ p
  out

private def modelDevice {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg) : Device :=
  model.thinker.textModel.embed_tokens.device

private def validateForcedLanguage
    (supportedLanguages : Array String)
    (language : Option String)
    : IO (Option String) := do
  match language with
  | none => pure none
  | some l =>
    let s := l.trim
    if s.isEmpty then
      pure none
    else
      match normalizeLanguageName s with
      | .error e => throw <| IO.userError e
      | .ok ln =>
        match validateLanguage supportedLanguages ln with
        | .error e => throw <| IO.userError e
        | .ok _ => pure (some ln)

private def preparePromptInputIds
    {cfg : Qwen3ASRConfig}
    (tok : tokenizer.qwen3.QwenTokenizer)
    (prompt : String)
    (audioLen : UInt64)
    : IO (Sigma (fun seq => T #[1, seq])) := do
  let processor : Qwen3ASRProcessor := {}
  let promptExpanded ←
    match processor.replaceMultimodalSpecialTokens #[prompt] #[audioLen] with
    | .ok xs => pure (xs.getD 0 prompt)
    | .error e => throw <| IO.userError e

  let promptIds := tokenizer.qwen3.encodeText tok promptExpanded
  let idsVals : Array Int64 := promptIds.map (fun id => Int64.ofNat id.toNat)

  let seq : UInt64 := idsVals.size.toUInt64
  if seq == 0 then
    throw <| IO.userError "ASR prompt/input sequence is empty"
  let inputIds : T #[1, seq] := reshape (data.fromInt64Array idsVals) #[1, seq]
  pure ⟨seq, inputIds⟩

private def decodeGeneratedText
    (tok : tokenizer.qwen3.QwenTokenizer)
    {seq : UInt64}
    (outSeq : UInt64)
    (outIds : T #[1, outSeq])
    : IO String := do
  if outSeq <= seq then
    pure ""
  else
    let newSeq := outSeq - seq
    let newOnly : T #[1, newSeq] := data.slice outIds 1 seq newSeq
    let flat : T #[newSeq] := reshape newOnly #[newSeq]
    let idsU64 ← data.tensorToUInt64Array flat
    let ids := idsU64.map (fun x => x.toUInt32)
    pure (tokenizer.qwen3.decodeText tok ids)

private def extractAudioLenFromFeatureMask {cfg : Qwen3ASRConfig} {frames : UInt64}
    (featureAttentionMask : T #[1, frames])
    : IO UInt64 := do
  let validFramesTensor : T #[1] := nn.sumDim (data.toLong featureAttentionMask) 1 false
  let validFramesArr ← data.tensorToUInt64Array validFramesTensor
  let validFrames := validFramesArr.getD 0 0
  let audioLenRaw := AudioEncoderConfig.featExtractOutputLength validFrames
  let audioLenCap := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  pure (if audioLenRaw <= audioLenCap then audioLenRaw else audioLenCap)

private def maybeAlignTimestamps
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (text : String)
    (language : String)
    {melBins frames : UInt64}
    (frontendOut : WhisperFrontendOutput melBins frames)
    (audioLen : UInt64)
    (returnTimeStamps : Bool)
    : IO (Option ForcedAlignResult) := do
  if !returnTimeStamps then
    pure none
  else if text.trim.isEmpty then
    pure none
  else if !ThinkerConfig.isForcedAligner cfg.thinkerConfig then
    throw <| IO.userError
      "returnTimeStamps=true requires a forced-aligner thinker checkpoint (model_type contains \"forced_aligner\")"
  else
    let (wordList, alignRaw) := Qwen3ForceAlignProcessor.encodeTimestampText text language
    let processor : Qwen3ASRProcessor := {}
    let alignExpanded ←
      match processor.replaceMultimodalSpecialTokens #[alignRaw] #[audioLen] with
      | .ok xs => pure (xs.getD 0 alignRaw)
      | .error e => throw <| IO.userError e
    let alignIdsArr := tokenizer.qwen3.encodeText tok alignExpanded
    let seqAlign : UInt64 := alignIdsArr.size.toUInt64
    if seqAlign == 0 then
      pure none
    else
      let dev := modelDevice model
      let alignVals := alignIdsArr.map (fun id => Int64.ofNat id.toNat)
      let inputIdsAlignCpu : T #[1, seqAlign] := reshape (data.fromInt64Array alignVals) #[1, seqAlign]
      let inputIdsAlign : T #[1, seqAlign] := inputIdsAlignCpu.to dev
      let inputFeaturesDev : T #[1, melBins, frames] := frontendOut.inputFeatures.to dev
      let featureAttentionMaskDev : T #[1, frames] := frontendOut.featureAttentionMask.to dev
      let results ← model.alignPrepared
        inputIdsAlign
        #[wordList]
        (inputFeatures := some inputFeaturesDev)
        (featureAttentionMask := some featureAttentionMaskDev)
        (attentionMask := (none : Option (T #[1, seqAlign])))
        (timestampTokenId := cfg.timestampTokenId)
        (timestampSegmentTime := cfg.timestampSegmentTime)
      pure (results[0]?)

private def transcribeWithFrontend
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    {melBins frames : UInt64}
    (frontendOut : WhisperFrontendOutput melBins frames)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let forceLanguage ← validateForcedLanguage model.supportLanguages language
  let prompt := buildTextPrompt context forceLanguage
  let audioLen ← extractAudioLenFromFeatureMask (cfg := cfg) frontendOut.featureAttentionMask

  let promptPack ← preparePromptInputIds (cfg := cfg) tok prompt audioLen
  let dev := modelDevice model
  let seq := promptPack.1
  let inputIds : T #[1, seq] := promptPack.2.to dev
  let inputFeaturesDev : T #[1, melBins, frames] := frontendOut.inputFeatures.to dev
  let featureAttentionMaskDev : T #[1, frames] := frontendOut.featureAttentionMask.to dev
  let generated ←
    model.generateGreedy
      inputIds
      (inputFeatures := some inputFeaturesDev)
      (featureAttentionMask := some featureAttentionMaskDev)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)

  let outSeq := generated.1
  let outIds := generated.2
  let rawDecoded ← decodeGeneratedText tok (seq := seq) outSeq outIds
  let (lang, txt) := parseAsrOutput rawDecoded forceLanguage
  let timeStamps ← maybeAlignTimestamps
    model tok txt lang frontendOut audioLen returnTimeStamps
  pure { language := lang, text := txt, timeStamps := timeStamps }

/-- Offline ASR for one in-memory 16k mono waveform. -/
def transcribeWaveform
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio16k : Array Float)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let maxChunkSec := if returnTimeStamps then maxForceAlignInputSeconds else maxAsrInputSeconds
  let chunks := splitAudioIntoChunks audio16k sampleRate16k maxChunkSec

  let mut langs : Array String := #[]
  let mut texts : Array String := #[]
  let mut tsItems : Array ForcedAlignItem := #[]

  for (chunkWav, offsetSec) in chunks do
    let frontendPack ←
      waveformToWhisperFeaturesDynamic
        preprocessor
        chunkWav
        (minSeconds := minAsrInputSeconds)
        (maxSeconds := some maxChunkSec)
    let _frames := frontendPack.1
    let frontendOut := frontendPack.2
    let r ← transcribeWithFrontend
      model tok frontendOut context language returnTimeStamps maxNewTokens eosTokenIds
    langs := langs.push r.language
    texts := texts.push r.text
    if returnTimeStamps then
      match r.timeStamps with
      | some ts =>
        let shifted := offsetForcedAlignResult ts offsetSec
        tsItems := tsItems ++ shifted.items
      | none => pure ()

  pure {
    language := mergeLanguages langs
    text := mergeChunkTexts texts
    timeStamps :=
      if returnTimeStamps && !tsItems.isEmpty then
        some { items := tsItems }
      else
        none
  }

private def broadcastContexts (n : Nat) (contexts : Array String) : IO (Array String) := do
  if n == 0 then
    pure #[]
  else if contexts.isEmpty then
    pure (Array.replicate n "")
  else if contexts.size == 1 then
    pure (Array.replicate n (contexts[0]!))
  else if contexts.size == n then
    pure contexts
  else
    throw <| IO.userError s!"Batch size mismatch: audio={n}, context={contexts.size}"

private def broadcastLanguages (n : Nat) (languages : Array (Option String))
    : IO (Array (Option String)) := do
  if n == 0 then
    pure #[]
  else if languages.isEmpty then
    pure (Array.replicate n none)
  else if languages.size == 1 then
    pure (Array.replicate n (languages[0]!))
  else if languages.size == n then
    pure languages
  else
    throw <| IO.userError s!"Batch size mismatch: audio={n}, language={languages.size}"

/-- Offline ASR for a batch of in-memory 16k mono waveforms. -/
def transcribeWaveforms
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios16k : Array (Array Float))
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) := do
  let n := audios16k.size
  let ctxs ← broadcastContexts n contexts
  let langs ← broadcastLanguages n languages
  let mut out : Array ASRTranscription := Array.mkEmpty n
  for i in [:n] do
    let r ← transcribeWaveform
      model tok preprocessor (audios16k[i]!)
      (context := ctxs[i]!)
      (language := langs[i]!)
      (returnTimeStamps := returnTimeStamps)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
    out := out.push r
  pure out

/-- Offline ASR from one WAV path. WAV is normalized/resampled to 16k first. -/
def transcribeWav
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPath : String)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  transcribeWaveform
    model tok preprocessor wav16k context language returnTimeStamps maxNewTokens eosTokenIds

/-- Offline ASR from WAV paths. WAVs are normalized/resampled to 16k first. -/
def transcribeWavs
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPaths : Array String)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) := do
  let n := wavPaths.size
  let ctxs ← broadcastContexts n contexts
  let langs ← broadcastLanguages n languages
  let mut out : Array ASRTranscription := Array.mkEmpty n
  for i in [:n] do
    let r ← transcribeWav
      model
      tok
      preprocessor
      (wavPaths[i]!)
      (context := ctxs[i]!)
      (language := langs[i]!)
      (returnTimeStamps := returnTimeStamps)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)
    out := out.push r
  pure out

namespace Qwen3ASRForConditionalGeneration

/-- Method-style offline ASR for one waveform. -/
def transcribeWaveform
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio16k : Array Float)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  torch.qwen3asr.transcribeWaveform
    m tok preprocessor audio16k context language returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR for multiple waveforms. -/
def transcribeWaveforms
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios16k : Array (Array Float))
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  torch.qwen3asr.transcribeWaveforms
    m tok preprocessor audios16k contexts languages returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from one WAV path. -/
def transcribeWav
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPath : String)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  torch.qwen3asr.transcribeWav
    m tok preprocessor wavPath context language returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from WAV paths. -/
def transcribeWavs
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPaths : Array String)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  torch.qwen3asr.transcribeWavs
    m tok preprocessor wavPaths contexts languages returnTimeStamps maxNewTokens eosTokenIds

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
