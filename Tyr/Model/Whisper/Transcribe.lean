/-
  Tyr/Model/Whisper/Transcribe.lean

  Offline Whisper transcription pipeline in Tyr.
  Includes a Whisper-CLI-style decoding path:
  - beam-first decode at temperature 0
  - temperature fallback with best-of sampling
  - no-speech gating and decode-failure heuristics
  - rolling prompt-context carry between chunks
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

structure WhisperDecodeOptions where
  beamSize : UInt64 := 5
  bestOf : UInt64 := 5
  temperature : Float := 0.0
  temperatureInc : Float := 0.2
  maxTemperature : Float := 1.0
  topK : UInt64 := 0
  topP : Float := 1.0
  logprobThreshold : Float := -1.0
  noSpeechThreshold : Float := 0.6
  compressionRatioThreshold : Float := 2.4
  conditionOnPreviousText : Bool := true
  maxContextTokens : UInt64 := 0
  resetContextOnFallback : Bool := true
  noFallback : Bool := false
  deriving Repr, Inhabited

private def sampleRate16k : UInt64 := 16000

private structure DecodeState where
  allIds : Array UInt32
  generated : Array UInt32 := #[]
  sumLogprob : Float := 0.0
  tokenCount : Nat := 0
  noSpeechProb : Float := 0.0
  measuredNoSpeech : Bool := false
  deriving Inhabited

private structure BeamState where
  allIds : Array UInt32
  generated : Array UInt32 := #[]
  sumLogprob : Float := 0.0
  tokenCount : Nat := 0
  noSpeechProb : Float := 0.0
  measuredNoSpeech : Bool := false
  finished : Bool := false
  deriving Inhabited

private structure WhisperDecodeCandidate where
  generated : Array UInt32 := #[]
  text : String := ""
  avgLogprob : Float := -100.0
  noSpeechProb : Float := 0.0
  compressionRatio : Float := 0.0
  repetitionLoop : Bool := false
  temperature : Float := 0.0
  deriving Inhabited

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

private def takeLast (xs : Array α) (n : Nat) : Array α :=
  if n == 0 then
    #[]
  else if xs.size <= n then
    xs
  else
    xs.extract (xs.size - n) xs.size

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

private def decodeTemperatures (opts : WhisperDecodeOptions) : Array Float :=
  let t0 := if opts.temperature < 0.0 then 0.0 else opts.temperature
  if opts.noFallback || opts.temperatureInc <= 0.0 then
    #[t0]
  else
    Id.run do
      let maxT := if opts.maxTemperature < t0 then t0 else opts.maxTemperature
      let mut out : Array Float := #[t0]
      let mut t := t0
      let mut guard : Nat := 0
      while t + opts.temperatureInc <= maxT + (1e-6 : Float) && guard < 64 do
        t := t + opts.temperatureInc
        out := out.push t
        guard := guard + 1
      out

private def uniqueTokenCount (ids : Array UInt32) : Nat :=
  Id.run do
    let mut uniq : Array UInt32 := #[]
    for id in ids do
      if !uniq.contains id then
        uniq := uniq.push id
    uniq.size

private def approxCompressionRatio (ids : Array UInt32) : Float :=
  if ids.isEmpty then
    0.0
  else
    let uniq := uniqueTokenCount ids
    if uniq == 0 then
      0.0
    else
      ids.size.toFloat / uniq.toFloat

private def hasTailRepeatPattern (ids : Array UInt32) (patternLen repeats : Nat) : Bool :=
  if patternLen == 0 || repeats <= 1 then
    false
  else
    let need := patternLen * repeats
    if ids.size < need then
      false
    else
      let tail := ids.extract (ids.size - need) ids.size
      let refPat := tail.extract 0 patternLen
      Id.run do
        let mut ok := true
        for r in [:repeats] do
          if ok then
            let seg := tail.extract (r * patternLen) ((r + 1) * patternLen)
            if seg != refPat then
              ok := false
        ok

private def hasRepeatedSingleTail (ids : Array UInt32) (runLen : Nat) : Bool :=
  if runLen <= 1 || ids.size < runLen then
    false
  else
    let tail := ids.extract (ids.size - runLen) ids.size
    let first := tail[0]!
    Id.run do
      let mut ok := true
      for id in tail do
        if ok && id != first then
          ok := false
      ok

private def hasRepetitionLoop (ids : Array UInt32) : Bool :=
  hasRepeatedSingleTail ids 8 ||
  hasTailRepeatPattern ids 2 4 ||
  hasTailRepeatPattern ids 3 3

private def composePromptIds
    (basePrompt : Array UInt32)
    (rollingContext : Array UInt32)
    (promptMaxLen : Nat)
    (contextLimit : Nat)
    (conditionOnPreviousText : Bool)
    : Array UInt32 :=
  if promptMaxLen <= basePrompt.size then
    basePrompt.extract 0 promptMaxLen
  else if !conditionOnPreviousText || rollingContext.isEmpty || contextLimit == 0 then
    basePrompt
  else
    let ctxBudget := Nat.min contextLimit (promptMaxLen - basePrompt.size)
    basePrompt ++ takeLast rollingContext ctxBudget

private def candidateNeedsFallback (opts : WhisperDecodeOptions) (cand : WhisperDecodeCandidate) : Bool :=
  let badCompression :=
    opts.compressionRatioThreshold > 0.0 &&
      cand.compressionRatio > opts.compressionRatioThreshold
  let badLogprob :=
    cand.avgLogprob < opts.logprobThreshold &&
      cand.noSpeechProb < opts.noSpeechThreshold
  let likelyDecodeFailure := cand.generated.isEmpty && cand.noSpeechProb < opts.noSpeechThreshold
  cand.repetitionLoop || badCompression || badLogprob || likelyDecodeFailure

private def isNoSpeechChunk (opts : WhisperDecodeOptions) (cand : WhisperDecodeCandidate) : Bool :=
  cand.noSpeechProb > opts.noSpeechThreshold &&
    cand.avgLogprob < opts.logprobThreshold

private def buildInputIdsOnDevice
    (ids : Array UInt32)
    (device : Device)
    : IO (Sigma (fun seq => T #[1, seq])) := do
  let seq : UInt64 := ids.size.toUInt64
  if seq == 0 then
    throw <| IO.userError "Whisper decode prompt is empty"
  let idsVals : Array Int64 := ids.map (fun id => Int64.ofNat id.toNat)
  let inputIdsCpu : T #[1, seq] := reshape (data.fromInt64Array idsVals) #[1, seq]
  let inputIds : T #[1, seq] :=
    if inputIdsCpu.device == device then
      inputIdsCpu
    else
      inputIdsCpu.to device
  pure ⟨seq, inputIds⟩

private def logProbsFromLogitsArray (logits : Array Float) : Array Float :=
  if logits.isEmpty then
    #[]
  else
    let maxLogit := logits.foldl (fun acc x => if x > acc then x else acc) logits[0]!
    let sumExp := logits.foldl (fun acc x => acc + Float.exp (x - maxLogit)) 0.0
    let logDen :=
      if sumExp <= 0.0 then
        maxLogit
      else
        maxLogit + Float.log sumExp
    logits.map (fun x => x - logDen)

private def topKLogProbPairs (logProbs : Array Float) (k : Nat) : Array (Nat × Float) :=
  if k == 0 || logProbs.isEmpty then
    #[]
  else
    Id.run do
      let mut pairs : Array (Nat × Float) := #[]
      for i in [:logProbs.size] do
        pairs := pairs.push (i, logProbs[i]!)
      let sorted := pairs.qsort (fun a b => a.2 > b.2)
      sorted.extract 0 (Nat.min k sorted.size)

private def tokenLogProbFromLogits {vocab : UInt64}
    (logits : T #[1, vocab])
    (tokenId : UInt32)
    : IO Float := do
  let vals ← data.tensorToFloatArray' (nn.eraseShape logits)
  let logProbs := logProbsFromLogitsArray vals
  pure (logProbs.getD tokenId.toNat (-100.0))

private def tokenProbFromLogits {vocab : UInt64}
    (logits : T #[1, vocab])
    (tokenId : UInt32)
    : IO Float := do
  let vals ← data.tensorToFloatArray' (nn.eraseShape logits)
  let logProbs := logProbsFromLogitsArray vals
  pure (Float.exp (logProbs.getD tokenId.toNat (-100.0)))

private def clampTopK (topK vocab : UInt64) : UInt64 :=
  if topK == 0 then
    0
  else if topK > vocab then
    vocab
  else
    topK

private def filteredSamplingLogits {vocab : UInt64}
    (logits : T #[1, vocab])
    (temperature : Float)
    (topK : UInt64)
    (topP : Float)
    : T #[1, vocab] :=
  let scaled :=
    if temperature == 1.0 then
      logits
    else
      mul_scalar logits (1.0 / temperature)
  let topK' := clampTopK topK vocab
  let filtered :=
    if topK' == 0 then
      scaled
    else
      nn.topKFilter scaled topK'
  let topP' :=
    if topP <= 0.0 then
      (1e-6 : Float)
    else if topP > 1.0 then
      1.0
    else
      topP
  if topP' >= 1.0 then
    filtered
  else
    nn.topPFilter filtered topP'

private def argmaxTokenFromLogits {vocab : UInt64}
    (logits : T #[1, vocab])
    (fallback : UInt64)
    : IO UInt32 := do
  let nextTokRaw : T #[1] := nn.argmax logits 1
  let nextVals ← data.tensorToUInt64Array nextTokRaw
  pure (nextVals.getD 0 fallback).toUInt32

private def sampleTokenFromLogits {vocab : UInt64}
    (logits : T #[1, vocab])
    : IO UInt32 := do
  let vals ← data.tensorToFloatArray' (nn.eraseShape logits)
  let probsVals := (logProbsFromLogitsArray vals).map Float.exp
  let probs : T #[1, vocab] := reshape (data.fromFloatArray probsVals) #[1, vocab]
  let sampled ← nn.multinomial probs 1 false
  let sampledFlat : T #[1] := reshape (nn.eraseShape sampled) #[1]
  let sampledVals ← data.tensorToUInt64Array sampledFlat
  pure (sampledVals.getD 0 0).toUInt32

private partial def decodeLoop
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (temperature : Float)
    (topK : UInt64)
    (topP : Float)
    (sample : Bool)
    (remaining : Nat)
    (state : DecodeState)
    : IO DecodeState := do
  if remaining == 0 then
    pure state
  else
    match (← buildInputIdsOnDevice state.allIds encoderHidden.device) with
    | ⟨seq, inputIds⟩ =>
      let seqNat := seq.toNat
      if seqNat == 0 then
        pure state
      else
        let logits ← model.decode inputIds encoderHidden
        let last : T #[1, cfg.vocabSize] :=
          reshape (data.slice logits 1 (seq - 1) 1) #[1, cfg.vocabSize]
        let noSpeechProb ←
          if state.measuredNoSpeech then
            pure state.noSpeechProb
          else
            match noSpeechTokenId with
            | none => pure 0.0
            | some tid => tokenProbFromLogits last tid
        let distLogits :=
          if sample then
            filteredSamplingLogits last temperature topK topP
          else
            last
        let nextId ←
          if sample then
            sampleTokenFromLogits distLogits
          else
            argmaxTokenFromLogits last eosTokenId
        let tokLogprob ← tokenLogProbFromLogits distLogits nextId
        let nextIdU64 := nextId.toUInt64
        let generated' :=
          if nextIdU64 == eosTokenId then state.generated else state.generated.push nextId
        let tokenCount' :=
          if nextIdU64 == eosTokenId then state.tokenCount else state.tokenCount + 1
        let sumLogprob' :=
          if nextIdU64 == eosTokenId then state.sumLogprob else state.sumLogprob + tokLogprob
        let state' : DecodeState := {
          allIds := state.allIds.push nextId
          generated := generated'
          sumLogprob := sumLogprob'
          tokenCount := tokenCount'
          noSpeechProb := noSpeechProb
          measuredNoSpeech := true
        }
        if nextIdU64 == eosTokenId then
          pure state'
        else
          decodeLoop
            model
            encoderHidden
            noSpeechTokenId
            eosTokenId
            temperature
            topK
            topP
            sample
            (remaining - 1)
            state'

private def beamRankScore (beam : BeamState) : Float :=
  if beam.tokenCount == 0 then
    -100.0
  else
    beam.sumLogprob / beam.tokenCount.toFloat

private def chooseBetterBeam (a b : BeamState) : BeamState :=
  if beamRankScore b > beamRankScore a then b else a

private def expandBeam
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (beamWidth : UInt64)
    (beam : BeamState)
    : IO (Array BeamState) := do
  match (← buildInputIdsOnDevice beam.allIds encoderHidden.device) with
  | ⟨seq, inputIds⟩ =>
    let logits ← model.decode inputIds encoderHidden
    let last : T #[1, cfg.vocabSize] :=
      reshape (data.slice logits 1 (seq - 1) 1) #[1, cfg.vocabSize]
    let noSpeechProb ←
      if beam.measuredNoSpeech then
        pure beam.noSpeechProb
      else
        match noSpeechTokenId with
        | none => pure 0.0
        | some tid => tokenProbFromLogits last tid
    let kNat := Nat.max 1 (Nat.min beamWidth.toNat cfg.vocabSize.toNat)
    let logitsArr ← data.tensorToFloatArray' (nn.eraseShape last)
    let logProbs := logProbsFromLogitsArray logitsArr
    let topPairs := topKLogProbPairs logProbs kNat
    let mut out : Array BeamState := #[]
    for p in topPairs do
      let nextIdU64 := p.1.toUInt64
      let nextId := nextIdU64.toUInt32
      let lp := p.2
      let generated' :=
        if nextIdU64 == eosTokenId then beam.generated else beam.generated.push nextId
      let tokenCount' :=
        if nextIdU64 == eosTokenId then beam.tokenCount else beam.tokenCount + 1
      let sumLogprob' :=
        if nextIdU64 == eosTokenId then beam.sumLogprob else beam.sumLogprob + lp
      out := out.push {
        allIds := beam.allIds.push nextId
        generated := generated'
        sumLogprob := sumLogprob'
        tokenCount := tokenCount'
        noSpeechProb := noSpeechProb
        measuredNoSpeech := true
        finished := nextIdU64 == eosTokenId
      }
    pure out

private partial def beamDecodeLoop
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (beamWidth : UInt64)
    (remaining : Nat)
    (beams : Array BeamState)
    : IO (Array BeamState) := do
  if remaining == 0 then
    pure beams
  else
    let mut allFinished := true
    for beam in beams do
      if !beam.finished then
        allFinished := false
    if allFinished then
      pure beams
    else
      let mut expanded : Array BeamState := #[]
      for beam in beams do
        if beam.finished then
          expanded := expanded.push beam
        else
          expanded := expanded ++ (← expandBeam model encoderHidden noSpeechTokenId eosTokenId beamWidth beam)
      if expanded.isEmpty then
        pure beams
      else
        let sorted := expanded.qsort (fun a b => beamRankScore a > beamRankScore b)
        let keep := Nat.max 1 (Nat.min sorted.size beamWidth.toNat)
        beamDecodeLoop
          model
          encoderHidden
          noSpeechTokenId
          eosTokenId
          beamWidth
          (remaining - 1)
          (sorted.extract 0 keep)

private def decodeBeam
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (promptIds : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (beamWidth : UInt64)
    (remaining : Nat)
    : IO DecodeState := do
  let initBeam : BeamState := { allIds := promptIds }
  let beams ←
    beamDecodeLoop
      model
      encoderHidden
      noSpeechTokenId
      eosTokenId
      beamWidth
      remaining
      #[initBeam]
  if beams.isEmpty then
    pure { allIds := promptIds }
  else
    let mut best := beams[0]!
    for beam in beams do
      best := chooseBetterBeam best beam
    pure {
      allIds := best.allIds
      generated := best.generated
      sumLogprob := best.sumLogprob
      tokenCount := best.tokenCount
      noSpeechProb := best.noSpeechProb
      measuredNoSpeech := best.measuredNoSpeech
    }

private def toDecodeCandidate
    (tok : tokenizer.qwen3.QwenTokenizer)
    (st : DecodeState)
    (temperature : Float)
    : WhisperDecodeCandidate :=
  let avg :=
    if st.tokenCount == 0 then
      -100.0
    else
      st.sumLogprob / st.tokenCount.toFloat
  let text := decodeGeneratedText tok st.generated
  let ratio := approxCompressionRatio st.generated
  let repetition := hasRepetitionLoop st.generated
  {
    generated := st.generated
    text := text
    avgLogprob := avg
    noSpeechProb := st.noSpeechProb
    compressionRatio := ratio
    repetitionLoop := repetition
    temperature := temperature
  }

private def chooseBetterCandidate (a b : WhisperDecodeCandidate) : WhisperDecodeCandidate :=
  if b.avgLogprob > a.avgLogprob then b else a

private def decodeAtTemperature
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (promptIds : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (maxNewTokens : Nat)
    (opts : WhisperDecodeOptions)
    (temperature : Float)
    : IO WhisperDecodeCandidate := do
  if temperature <= 0.0 then
    let st ←
      if opts.beamSize > 1 then
        decodeBeam
          model
          encoderHidden
          promptIds
          noSpeechTokenId
          eosTokenId
          opts.beamSize
          maxNewTokens
      else
        decodeLoop
          model
          encoderHidden
          noSpeechTokenId
          eosTokenId
          0.0
          opts.topK
          opts.topP
          false
          maxNewTokens
          { allIds := promptIds }
    pure (toDecodeCandidate tok st temperature)
  else
    let runs := Nat.max 1 opts.bestOf.toNat
    let mut best? : Option WhisperDecodeCandidate := none
    for _ in [:runs] do
      let st ←
        decodeLoop
          model
          encoderHidden
          noSpeechTokenId
          eosTokenId
          temperature
          opts.topK
          opts.topP
          true
          maxNewTokens
          { allIds := promptIds }
      let cand := toDecodeCandidate tok st temperature
      best? :=
        match best? with
        | none => some cand
        | some cur => some (chooseBetterCandidate cur cand)
    pure ((best?).getD (default : WhisperDecodeCandidate))

private partial def decodeWithFallbackLoop
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (promptIds : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (maxNewTokens : Nat)
    (opts : WhisperDecodeOptions)
    (temperatures : Array Float)
    (idx : Nat)
    (usedFallback : Bool)
    : IO (WhisperDecodeCandidate × Bool) := do
  let t := temperatures.getD idx opts.temperature
  let cand ←
    decodeAtTemperature
      model
      tok
      encoderHidden
      promptIds
      noSpeechTokenId
      eosTokenId
      maxNewTokens
      opts
      t
  let isLast := idx + 1 >= temperatures.size
  if isLast then
    pure (cand, usedFallback)
  else if candidateNeedsFallback opts cand then
    decodeWithFallbackLoop
      model
      tok
      encoderHidden
      promptIds
      noSpeechTokenId
      eosTokenId
      maxNewTokens
      opts
      temperatures
      (idx + 1)
      true
  else
    pure (cand, usedFallback)

private def decodeWithFallback
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (encoderHidden : T #[1, encSeq, cfg.dModel])
    (promptIds : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (eosTokenId : UInt64)
    (maxNewTokens : Nat)
    (opts : WhisperDecodeOptions)
    : IO (WhisperDecodeCandidate × Bool) := do
  let temperatures := decodeTemperatures opts
  decodeWithFallbackLoop
    model
    tok
    encoderHidden
    promptIds
    noSpeechTokenId
    eosTokenId
    maxNewTokens
    opts
    temperatures
    0
    false

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
    (decodeOpts : WhisperDecodeOptions := {})
    : IO WhisperTranscription := do
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  let frontendDevice := model.model.decoderTokenEmbedding.device
  let basePrompt := buildPromptIds cfg tok language noTimestamps
  let maxTarget := cfg.maxTargetPositions.toNat
  if maxTarget <= basePrompt.size + 1 then
    throw <| IO.userError
      s!"Whisper max_target_positions={cfg.maxTargetPositions} is too small for base prompt size={basePrompt.size}"
  let maxNewCap := maxTarget - basePrompt.size - 1
  let maxNewNat := Nat.max 1 (Nat.min maxNewTokens.toNat maxNewCap)
  let promptMaxLen := maxTarget - maxNewNat - 1
  let modelContextLimit := maxTarget / 2
  let userContextLimit :=
    if decodeOpts.maxContextTokens == 0 then
      modelContextLimit
    else
      decodeOpts.maxContextTokens.toNat
  let contextBudget := if promptMaxLen > basePrompt.size then promptMaxLen - basePrompt.size else 0
  let contextLimit := Nat.min contextBudget userContextLimit
  let noSpeechTokenId := tokenIdByText? tok "<|nospeech|>"
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

  let mut rollingContext : Array UInt32 := #[]
  let mut texts : Array String := #[]
  let mut allTokenIds : Array UInt32 := #[]
  for i in [:chunks.size] do
    let chunk := chunks[i]!
    if i > 0 && i + 1 == chunks.size && chunk.size < (chunkSamples / 2) then
      -- Very short tail chunks tend to hallucinate with carried text context.
      rollingContext := #[]

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
      let promptIds :=
        composePromptIds
          basePrompt
          rollingContext
          promptMaxLen
          contextLimit
          decodeOpts.conditionOnPreviousText
      let (cand, usedFallback) ←
        decodeWithFallback
          model
          tok
          encoderHidden
          promptIds
          noSpeechTokenId
          cfg.eosTokenId
          maxNewNat
          decodeOpts
      let silent := isNoSpeechChunk decodeOpts cand
      if !silent then
        let text := cand.text.trimAscii.toString
        if !text.isEmpty then
          texts := texts.push text
        allTokenIds := allTokenIds ++ cand.generated
      if decodeOpts.conditionOnPreviousText then
        if silent then
          pure ()
        else if decodeOpts.resetContextOnFallback && (usedFallback || cand.temperature > 0.5) then
          rollingContext := #[]
        else
          rollingContext := takeLast (rollingContext ++ cand.generated) contextLimit

  pure {
    language := normalizeLanguageCode language
    text := mergeChunkTexts texts
    tokenIds := allTokenIds
  }

end torch.whisper
