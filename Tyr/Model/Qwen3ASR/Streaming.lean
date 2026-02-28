/- 
  Tyr/Model/Qwen3ASR/Streaming.lean

  Lean-native streaming helpers mirroring upstream Qwen3-ASR vLLM flow:
  - init_streaming_state
  - streaming_transcribe
  - finish_streaming_transcribe
-/
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.Model
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.Processor
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

private def asrTextTag : String := "<asr_text>"
private def languagePrefix : String := "language "
private def sampleRate16k : Float := 16000.0

/-- Streaming ASR state (single stream). -/
structure ASRStreamingState where
  unfixedChunkNum : Nat := 2
  unfixedTokenNum : Nat := 5
  chunkSizeSec : Float := 2.0
  chunkSizeSamples : Nat := 32000

  chunkId : Nat := 0
  buffer : Array Float := #[]
  audioAccum : Array Float := #[]

  promptRaw : String := ""
  context : String := ""
  forceLanguage : Option String := none

  language : String := ""
  text : String := ""
  rawDecoded : String := ""
  deriving Repr, Inhabited

/-- Streaming decode callback: `(prompt, accumulated_audio_16k) -> raw decoded text`. -/
abbrev StreamingDecodeFn := String → Array Float → IO String

private def toLower (s : String) : String :=
  String.ofList (s.toList.map Char.toLower)

private def startsWithStr (s pref : String) : Bool :=
  let ss := s.toList
  let pp := pref.toList
  pp.length ≤ ss.length && ss.take pp.length == pp

private def dropChars (s : String) (n : Nat) : String :=
  String.ofList (s.toList.drop n)

private def joinWith (sep : String) (parts : List String) : String :=
  match parts with
  | [] => ""
  | p :: ps => ps.foldl (fun acc x => acc ++ sep ++ x) p

private def containsSubstr (s pat : String) : Bool :=
  if pat.isEmpty then
    false
  else
    (s.splitOn pat).length > 1

private def splitFirst (s sep : String) : Option (String × String) :=
  match s.splitOn sep with
  | [] => none
  | [_] => none
  | x :: xs => some (x, joinWith sep xs)

/-- Normalize language as `Xxxxx...` (first upper, rest lower). -/
def normalizeLanguageName (language : String) : Except String String := do
  let s := language.trim
  if s.isEmpty then
    throw "language is empty"
  match s.toList with
  | [] => throw "language is empty"
  | c :: cs =>
    pure <| String.ofList (Char.toUpper c :: cs.map Char.toLower)

/-- Validate language against configured support list. -/
def validateLanguage (supported : Array String) (language : String) : Except String Unit := do
  if supported.contains language then
    pure ()
  else
    throw s!"Unsupported language: {language}. Supported: {supported}"

private def segmentEq (chars : Array Char) (a b len : Nat) : Bool := Id.run do
  if a + len > chars.size || b + len > chars.size then
    return false
  let mut ok := true
  for j in [:len] do
    if ok && chars[a + j]! != chars[b + j]! then
      ok := false
  return ok

private def segmentToString (chars : Array Char) (start len : Nat) : String :=
  String.ofList ((chars.extract start (start + len)).toList)

private def fixCharRepeats (s : String) (threshold : Nat) : String :=
  if threshold == 0 then
    s
  else
    Id.run do
      let chars := s.toList.toArray
      let mut out : Array Char := #[]
      let mut i : Nat := 0
      while i < chars.size do
        let mut count : Nat := 1
        while i + count < chars.size && chars[i + count]! == chars[i]! do
          count := count + 1
        if count > threshold then
          out := out.push chars[i]!
        else
          for j in [:count] do
            out := out.push chars[i + j]!
        i := i + count
      String.ofList out.toList

private partial def fixPatternRepeats (s : String) (threshold : Nat) (maxLen : Nat := 20) : String :=
  let chars := s.toList.toArray
  let n := chars.size
  let minRepeatChars := threshold * 2
  if threshold == 0 || n < minRepeatChars then
    s
  else
    Id.run do
      let mut i : Nat := 0
      let mut result := ""
      let mut found := false
      while i + minRepeatChars <= n && !found do
        let mut k : Nat := 1
        let mut matched := false
        while k <= maxLen && !matched do
          if i + k * threshold > n then
            k := maxLen + 1
          else
            let mut rep : Nat := 1
            let mut valid := true
            while rep < threshold && valid do
              let startIdx := i + rep * k
              if !(segmentEq chars i startIdx k) then
                valid := false
              rep := rep + 1
            if valid then
              let mut endIndex := i + threshold * k
              while endIndex + k <= n && segmentEq chars i endIndex k do
                endIndex := endIndex + k
              result := result ++ segmentToString chars i k
              let tail := segmentToString chars endIndex (n - endIndex)
              result := result ++ fixPatternRepeats tail threshold maxLen
              i := n
              found := true
              matched := true
            else
              k := k + 1
        if !found then
          result := result.push chars[i]!
          i := i + 1
      if !found then
        result := result ++ segmentToString chars i (n - i)
      result

/-- Port of upstream repetition cleanup used before parsing metadata. -/
def detectAndFixRepetitions (text : String) (threshold : Nat := 20) : String :=
  fixPatternRepeats (fixCharRepeats text threshold) threshold

/-- Parse raw ASR output into `(language, text)` with upstream semantics. -/
def parseAsrOutput (raw : String) (userLanguage : Option String := none) : String × String :=
  let s0 := raw.trim
  if s0.isEmpty then
    ("", "")
  else
    let s := detectAndFixRepetitions s0
    match userLanguage with
    | some forced =>
      let f := forced.trim
      if f.isEmpty then
        ("", s)
      else
        (f, s)
    | none =>
      match splitFirst s asrTextTag with
      | none => ("", s.trim)
      | some (metaPart, textPart) =>
        let metaLower := toLower metaPart
        if containsSubstr metaLower "language none" then
          let t := textPart.trim
          if t.isEmpty then
            ("", "")
          else
            ("", t)
        else
          Id.run do
            let mut lang := ""
            for line in metaPart.splitOn "\n" do
              if lang.isEmpty then
                let lineTrim := line.trim
                if !lineTrim.isEmpty then
                  let low := toLower lineTrim
                  if startsWithStr low languagePrefix then
                    let val := (dropChars lineTrim languagePrefix.length).trim
                    if !val.isEmpty then
                      match normalizeLanguageName val with
                      | .ok ln => lang := ln
                      | .error _ => lang := val
            (lang, textPart.trim)

private def joinArrayWith (sep : String) (xs : Array String) : String := Id.run do
  if xs.isEmpty then
    return ""
  let mut out := xs[0]!
  for i in [1:xs.size] do
    out := out ++ sep ++ xs[i]!
  out

/-- Merge per-chunk languages by removing empties and consecutive duplicates. -/
def mergeLanguages (langs : Array String) : String :=
  let merged : Array String := Id.run do
    let mut out : Array String := #[]
    let mut prev : Option String := none
    for l in langs do
      let x := l.trim
      if !x.isEmpty then
        match prev with
        | some p =>
          if x != p then
            out := out.push x
            prev := some x
        | none =>
          out := out.push x
          prev := some x
    out
  joinArrayWith "," merged

/-- Build base text prompt for streaming decode. -/
def buildTextPrompt (context : String) (forceLanguage : Option String := none) : String :=
  let base :=
    "<|im_start|>system\n" ++ context ++ "<|im_end|>\n" ++
    "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n" ++
    "<|im_start|>assistant\n"
  match forceLanguage with
  | some l => base ++ s!"language {l}{asrTextTag}"
  | none => base

private def chunkSizeSamplesFromSec (chunkSizeSec : Float) : Nat :=
  let rounded := ((chunkSizeSec * sampleRate16k) + 0.5).toUInt64.toNat
  if rounded == 0 then 1 else rounded

/-- Initialize streaming state (single stream, 16k PCM). -/
def initStreamingState
    (supportedLanguages : Array String)
    (context : String := "")
    (language : Option String := none)
    (unfixedChunkNum : Nat := 2)
    (unfixedTokenNum : Nat := 5)
    (chunkSizeSec : Float := 2.0)
    : IO ASRStreamingState := do
  if chunkSizeSec <= 0.0 then
    throw <| IO.userError s!"chunk_size_sec must be > 0, got: {chunkSizeSec}"

  let forceLanguage ←
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

  let chunkSizeSamples := chunkSizeSamplesFromSec chunkSizeSec
  let promptRaw := buildTextPrompt context forceLanguage
  pure {
    unfixedChunkNum
    unfixedTokenNum
    chunkSizeSec
    chunkSizeSamples
    chunkId := 0
    buffer := #[]
    audioAccum := #[]
    promptRaw
    context := context
    forceLanguage
    language := ""
    text := ""
    rawDecoded := ""
  }

private def containsReplacementChar (s : String) : Bool :=
  s.toList.any (fun c => c == Char.ofNat 0xFFFD)

private def rollbackPrefixForChunk
    (tok : tokenizer.qwen3.QwenTokenizer)
    (rawDecoded : String)
    (unfixedTokenNum : Nat)
    : String := Id.run do
  let curIds := tokenizer.qwen3.encodeText tok rawDecoded
  let mut k := unfixedTokenNum
  let mut pref := ""
  let mut done := false
  while !done do
    let endIdx := if curIds.size > k then curIds.size - k else 0
    pref :=
      if endIdx > 0 then
        tokenizer.qwen3.decodeText tok (curIds.extract 0 endIdx)
      else
        ""
    if !containsReplacementChar pref then
      done := true
    else if endIdx == 0 then
      pref := ""
      done := true
    else
      k := k + 1
  pref

private def rollbackPrefixForFinish
    (tok : tokenizer.qwen3.QwenTokenizer)
    (rawDecoded : String)
    (unfixedTokenNum : Nat)
    : String :=
  let curIds := tokenizer.qwen3.encodeText tok rawDecoded
  let endIdxRaw := Nat.max 1 (curIds.size - unfixedTokenNum)
  let endIdx := Nat.min endIdxRaw curIds.size
  tokenizer.qwen3.decodeText tok (curIds.extract 0 endIdx)

/-- Streaming decode step:
    append new PCM samples, consume full chunks, update state each decode. -/
def streamingTranscribe
    (tok : tokenizer.qwen3.QwenTokenizer)
    (decodeFn : StreamingDecodeFn)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    : IO ASRStreamingState := do
  if state.chunkSizeSamples == 0 then
    throw <| IO.userError "streaming state has invalid chunk_size_samples=0"

  let mut st := { state with buffer := state.buffer ++ pcm16k }

  while st.buffer.size >= st.chunkSizeSamples do
    let chunk := st.buffer.extract 0 st.chunkSizeSamples
    let rest := st.buffer.extract st.chunkSizeSamples st.buffer.size
    st := { st with buffer := rest, audioAccum := st.audioAccum ++ chunk }

    let pref :=
      if st.chunkId < st.unfixedChunkNum then
        ""
      else
        rollbackPrefixForChunk tok st.rawDecoded st.unfixedTokenNum
    let prompt := st.promptRaw ++ pref
    let genText ← decodeFn prompt st.audioAccum
    let rawDecoded := if pref.isEmpty then genText else pref ++ genText
    let (lang, txt) := parseAsrOutput rawDecoded st.forceLanguage
    st := { st with
      rawDecoded := rawDecoded
      language := lang
      text := txt
      chunkId := st.chunkId + 1
    }

  pure st

/-- Flush tail audio shorter than one chunk and decode once more. -/
def finishStreamingTranscribe
    (tok : tokenizer.qwen3.QwenTokenizer)
    (decodeFn : StreamingDecodeFn)
    (state : ASRStreamingState)
    : IO ASRStreamingState := do
  if state.buffer.isEmpty then
    pure state
  else
    let tail := state.buffer
    let mut st := { state with buffer := #[], audioAccum := state.audioAccum ++ tail }
    let pref :=
      if st.chunkId < st.unfixedChunkNum then
        ""
      else
        rollbackPrefixForFinish tok st.rawDecoded st.unfixedTokenNum
    let prompt := st.promptRaw ++ pref
    let genText ← decodeFn prompt st.audioAccum
    let rawDecoded := if pref.isEmpty then genText else pref ++ genText
    let (lang, txt) := parseAsrOutput rawDecoded st.forceLanguage
    st := { st with
      rawDecoded := rawDecoded
      language := lang
      text := txt
      chunkId := st.chunkId + 1
    }
    pure st

/-- Lean decode helper that runs one model generation from `(prompt, audio)` pair. -/
def decodeStreamingChunkWithModel
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (prompt : String)
    (audio16k : Array Float)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO String := do
  let frontendOut ← waveformToWhisperFeatures preprocessor audio16k
  let validFramesTensor : T #[1] := nn.sumDim (data.toLong frontendOut.featureAttentionMask) 1 false
  let validFramesArr ← data.tensorToUInt64Array validFramesTensor
  let validFrames := validFramesArr.getD 0 0
  let audioLenRaw := AudioEncoderConfig.featExtractOutputLength validFrames
  let audioLenCap :=
    AudioEncoderConfig.framesAfterConv3
      cfg.thinkerConfig.audioConfig
      (PreprocessorConfig.expectedFrames preprocessor)
  let audioLen := if audioLenRaw <= audioLenCap then audioLenRaw else audioLenCap

  let processor : Qwen3ASRProcessor := {}
  let promptExpanded ←
    match processor.replaceMultimodalSpecialTokens #[prompt] #[audioLen] with
    | .ok xs => pure (xs.getD 0 prompt)
    | .error e => throw <| IO.userError e

  let promptIds := tokenizer.qwen3.encodeText tok promptExpanded
  let idsVals : Array Int64 := promptIds.map (fun id => Int64.ofNat id.toNat)

  let seq : UInt64 := idsVals.size.toUInt64
  if seq == 0 then
    throw <| IO.userError "decodeStreamingChunkWithModel produced empty prompt/input sequence"
  let dev := model.thinker.textModel.embed_tokens.device
  let inputIdsCpu : T #[1, seq] := reshape (data.fromInt64Array idsVals) #[1, seq]
  let inputIds : T #[1, seq] := inputIdsCpu.to dev
  let inputFeaturesDev : T #[1, preprocessor.featureSize, PreprocessorConfig.expectedFrames preprocessor] :=
    frontendOut.inputFeatures.to dev
  let featureAttentionMaskDev : T #[1, PreprocessorConfig.expectedFrames preprocessor] :=
    frontendOut.featureAttentionMask.to dev
  let generated ←
    model.generateGreedy
      inputIds
      (inputFeatures := some inputFeaturesDev)
      (featureAttentionMask := some featureAttentionMaskDev)
      (maxNewTokens := maxNewTokens)
      (eosTokenIds := eosTokenIds)

  let outSeq := generated.1
  let outIds := generated.2
  if outSeq <= seq then
    pure ""
  else
    let newSeq := outSeq - seq
    let newOnly : T #[1, newSeq] := data.slice outIds 1 seq newSeq
    let flat : T #[newSeq] := reshape newOnly #[newSeq]
    let idsU64 ← data.tensorToUInt64Array flat
    let ids := idsU64.map (fun x => x.toUInt32)
    pure (tokenizer.qwen3.decodeText tok ids)

/-- Model-backed streaming decode step (no Python/vLLM bridge). -/
def streamingTranscribeWithModel
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState := do
  let decodeFn : StreamingDecodeFn := fun prompt audio =>
    decodeStreamingChunkWithModel model tok preprocessor prompt audio maxNewTokens eosTokenIds
  streamingTranscribe tok decodeFn pcm16k state

/-- Model-backed streaming finish step (flush tail audio). -/
def finishStreamingTranscribeWithModel
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState := do
  let decodeFn : StreamingDecodeFn := fun prompt audio =>
    decodeStreamingChunkWithModel model tok preprocessor prompt audio maxNewTokens eosTokenIds
  finishStreamingTranscribe tok decodeFn state

namespace Qwen3ASRForConditionalGeneration

/-- Method-style init wrapper over `torch.qwen3asr.initStreamingState`. -/
def initStreamingState
    (m : Qwen3ASRForConditionalGeneration cfg)
    (context : String := "")
    (language : Option String := none)
    (unfixedChunkNum : Nat := 2)
    (unfixedTokenNum : Nat := 5)
    (chunkSizeSec : Float := 2.0)
    : IO ASRStreamingState :=
  torch.qwen3asr.initStreamingState
    m.supportLanguages context language unfixedChunkNum unfixedTokenNum chunkSizeSec

/-- Method-style streaming step with Lean model backend. -/
def streamingTranscribe
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (pcm16k : Array Float)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState :=
  streamingTranscribeWithModel m tok preprocessor pcm16k state maxNewTokens eosTokenIds

/-- Method-style streaming flush with Lean model backend. -/
def finishStreamingTranscribe
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (state : ASRStreamingState)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRStreamingState :=
  finishStreamingTranscribeWithModel m tok preprocessor state maxNewTokens eosTokenIds

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
