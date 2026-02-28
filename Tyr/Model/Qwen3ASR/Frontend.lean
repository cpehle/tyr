/-
  Tyr/Model/Qwen3ASR/Frontend.lean

  Lean-native Qwen3-ASR audio frontend:
  - PCM16 WAV loading
  - resample to target sampling rate
  - Whisper-style log-mel extraction
  - feature attention-mask generation
-/
import Tyr.Torch
import Tyr.Model.Qwen3ASR.PreprocessorConfig

namespace torch.qwen3asr

structure WhisperFrontendOutput (melBins frames : UInt64) where
  inputFeatures : T #[1, melBins, frames]
  featureAttentionMask : T #[1, frames]

private def readU16LE (bytes : ByteArray) (offset : Nat) : Option UInt16 :=
  if offset + 2 > bytes.size then
    none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    some (b0.toUInt16 ||| (b1.toUInt16 <<< 8))

private def readU32LE (bytes : ByteArray) (offset : Nat) : Option UInt32 :=
  if offset + 4 > bytes.size then
    none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    let b2 := bytes[offset + 2]!
    let b3 := bytes[offset + 3]!
    some (b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24))

private def hasTag4 (bytes : ByteArray) (offset : Nat) (tag : String) : Bool :=
  offset + 4 <= bytes.size && bytes.extract offset (offset + 4) == tag.toUTF8

private def i16ToFloat (u : UInt16) : Float :=
  let s : Int64 :=
    if u.toNat < 32768 then
      Int64.ofNat u.toNat
    else
      (Int64.ofNat u.toNat) - (Int64.ofNat 65536)
  s.toFloat / 32768.0

/-- Parse PCM16 WAV data chunk(s) and return `(sampleRate, monoWaveform)` in `[-1, 1]`. -/
def loadMonoPcm16Wav (path : String) : IO (UInt64 × Array Float) := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size < 12 then
    throw <| IO.userError s!"Invalid WAV (too small): {path}"
  if !(hasTag4 bytes 0 "RIFF" && hasTag4 bytes 8 "WAVE") then
    throw <| IO.userError s!"Invalid WAV RIFF/WAVE header: {path}"

  let mut fmtOffset : Option Nat := none
  let mut fmtSize : Nat := 0
  let mut dataOffset : Option Nat := none
  let mut dataSize : Nat := 0

  let mut off : Nat := 12
  while off + 8 <= bytes.size do
    let csizeU32 ←
      match readU32LE bytes (off + 4) with
      | some x => pure x
      | none => throw <| IO.userError s!"Invalid WAV chunk header in {path}"
    let csize := csizeU32.toNat
    let body := off + 8
    if body + csize > bytes.size then
      throw <| IO.userError s!"Truncated WAV chunk payload in {path}"

    if hasTag4 bytes off "fmt " then
      fmtOffset := some body
      fmtSize := csize
    if hasTag4 bytes off "data" then
      dataOffset := some body
      dataSize := csize

    let next := body + csize
    off := if next % 2 == 1 then next + 1 else next

  let fmtOff ←
    match fmtOffset with
    | some x => pure x
    | none => throw <| IO.userError s!"WAV missing fmt chunk: {path}"
  let dataOff ←
    match dataOffset with
    | some x => pure x
    | none => throw <| IO.userError s!"WAV missing data chunk: {path}"

  if fmtSize < 16 then
    throw <| IO.userError s!"Unsupported fmt chunk size ({fmtSize}) in {path}"

  let readFmtU16 (rel : Nat) (name : String) : IO UInt16 := do
    match readU16LE bytes (fmtOff + rel) with
    | some x => pure x
    | none => throw <| IO.userError s!"Invalid fmt field `{name}` in {path}"

  let readFmtU32 (rel : Nat) (name : String) : IO UInt32 := do
    match readU32LE bytes (fmtOff + rel) with
    | some x => pure x
    | none => throw <| IO.userError s!"Invalid fmt field `{name}` in {path}"

  let audioFormat ← readFmtU16 0 "audio_format"
  let channels ← readFmtU16 2 "num_channels"
  let sampleRate ← readFmtU32 4 "sample_rate"
  let bitsPerSample ← readFmtU16 14 "bits_per_sample"

  if audioFormat != 1 then
    throw <| IO.userError s!"Only PCM WAV is supported, got format={audioFormat} in {path}"
  if bitsPerSample != 16 then
    throw <| IO.userError s!"Only 16-bit WAV is supported, got bits_per_sample={bitsPerSample} in {path}"
  if channels == 0 then
    throw <| IO.userError s!"Invalid WAV channels=0 in {path}"
  if dataSize % 2 != 0 then
    throw <| IO.userError s!"Invalid PCM16 data size (odd number of bytes) in {path}"

  let chNat := channels.toNat
  let sampleCountTotal := dataSize / 2
  if sampleCountTotal < chNat then
    throw <| IO.userError s!"WAV data is too short for channel count in {path}"

  let frameCount := sampleCountTotal / chNat
  let mut mono : Array Float := Array.mkEmpty frameCount
  for frame in [:frameCount] do
    let mut acc : Float := 0.0
    for ch in [:chNat] do
      let sampleIdx := frame * chNat + ch
      let byteOff := dataOff + sampleIdx * 2
      let u ←
        match readU16LE bytes byteOff with
        | some x => pure x
        | none => throw <| IO.userError s!"Invalid PCM sample in {path}"
      acc := acc + i16ToFloat u
    mono := mono.push (acc / channels.toFloat)

  pure (sampleRate.toUInt64, mono)

def resampleLinear (audio : Array Float) (srcRate targetRate : UInt64) : Array Float :=
  if audio.isEmpty || srcRate == targetRate || srcRate == 0 || targetRate == 0 then
    audio
  else
    let inSize := audio.size
    let outSize := (((inSize.toUInt64) * targetRate) / srcRate).toNat
    if outSize == 0 then
      #[]
    else
      Id.run do
        let mut out : Array Float := Array.mkEmpty outSize
        let srcNat := srcRate.toNat
        let dstNat := targetRate.toNat
        for i in [:outSize] do
          let srcNum := i * srcNat
          let idx0 := srcNum / dstNat
          let rem := srcNum % dstNat
          let i0 := if idx0 < inSize then idx0 else inSize - 1
          let i1 := if i0 + 1 < inSize then i0 + 1 else i0
          let w := rem.toFloat / targetRate.toFloat
          let v0 := audio[i0]!
          let v1 := audio[i1]!
          out := out.push ((1.0 - w) * v0 + w * v1)
        out

private def fmax (a b : Float) : Float := if a > b then a else b
private def fmin (a b : Float) : Float := if a < b then a else b

private def log10f (x : Float) : Float :=
  Float.log x / Float.log 10.0

private def hertzToMelSlaney (freq : Float) : Float :=
  let minLogHertz := 1000.0
  let minLogMel := 15.0
  let logstep := 27.0 / Float.log 6.4
  if freq >= minLogHertz then
    minLogMel + Float.log (freq / minLogHertz) * logstep
  else
    3.0 * freq / 200.0

private def melToHertzSlaney (mel : Float) : Float :=
  let minLogHertz := 1000.0
  let minLogMel := 15.0
  let logstep := Float.log 6.4 / 27.0
  if mel >= minLogMel then
    minLogHertz * Float.exp (logstep * (mel - minLogMel))
  else
    200.0 * mel / 3.0

private def linspace (start stop : Float) (n : Nat) : Array Float :=
  if n == 0 then
    #[]
  else if n == 1 then
    #[start]
  else
    Id.run do
      let mut out : Array Float := Array.mkEmpty n
      let step := (stop - start) / ((n - 1).toFloat)
      for i in [:n] do
        out := out.push (start + step * i.toFloat)
      out

private def buildSlaneyMelFilterBankFlat
    (numFrequencyBins numMelFilters samplingRate : Nat)
    : Array Float :=
  if numFrequencyBins < 2 || numMelFilters == 0 then
    #[]
  else
    let minHz : Float := 0.0
    let maxHz : Float := (samplingRate / 2).toFloat
    let melMin := hertzToMelSlaney minHz
    let melMax := hertzToMelSlaney maxHz
    let melFreqs := linspace melMin melMax (numMelFilters + 2)
    let filterFreqs := melFreqs.map melToHertzSlaney
    let fftFreqs := linspace 0.0 maxHz numFrequencyBins
    Id.run do
      let mut out : Array Float := Array.mkEmpty (numFrequencyBins * numMelFilters)
      for i in [:numFrequencyBins] do
        let fft := fftFreqs.getD i 0.0
        for j in [:numMelFilters] do
          let f0 := filterFreqs.getD j 0.0
          let f1 := filterFreqs.getD (j + 1) 0.0
          let f2 := filterFreqs.getD (j + 2) 0.0
          let downDen := f1 - f0
          let upDen := f2 - f1
          let down :=
            if downDen == 0.0 then 0.0 else (fft - f0) / downDen
          let up :=
            if upDen == 0.0 then 0.0 else (f2 - fft) / upDen
          let tri := fmax 0.0 (fmin down up)
          let enormDen := f2 - f0
          let enorm := if enormDen == 0.0 then 0.0 else 2.0 / enormDen
          out := out.push (tri * enorm)
      out

private def takePadRight (xs : Array Float) (target : Nat) (padValue : Float) : Array Float × Nat :=
  let taken := if xs.size > target then xs[:target].toArray else xs
  let validLen := Nat.min xs.size target
  if taken.size == target then
    (taken, validLen)
  else
    Id.run do
      let mut out := taken
      let padN := target - taken.size
      for _ in [:padN] do
        out := out.push padValue
      (out, validLen)

private def zeroMeanUnitVar (audio : Array Float) (validLen : Nat) (paddingValue : Float) : Array Float :=
  if audio.isEmpty || validLen == 0 then
    audio
  else
    Id.run do
      let useLen := Nat.min validLen audio.size
      let mut sum : Float := 0.0
      for i in [:useLen] do
        sum := sum + audio[i]!
      let mean := sum / useLen.toFloat

      let mut varAcc : Float := 0.0
      for i in [:useLen] do
        let d := audio[i]! - mean
        varAcc := varAcc + d * d
      let std := Float.sqrt (varAcc / useLen.toFloat + 1e-7)
      let denom := if std <= 0.0 then 1.0 else std

      let mut out : Array Float := Array.mkEmpty audio.size
      for i in [:audio.size] do
        if i < useLen then
          out := out.push ((audio[i]! - mean) / denom)
        else
          out := out.push paddingValue
      out

private def buildFeatureAttentionMask
    (validSamples targetSamples hopLength : Nat)
    (returnAttentionMask : Bool)
    : Array Int64 :=
  if hopLength == 0 then
    #[]
  else
    let frames := targetSamples / hopLength
    let validFrames := validSamples / hopLength
    Id.run do
      let mut out : Array Int64 := Array.mkEmpty frames
      for i in [:frames] do
        let v : Int64 :=
          if returnAttentionMask then
            if i < validFrames then 1 else 0
          else
            1
        out := out.push v
      out

private def adjustMaskLength (vals : Array Int64) (target : Nat) : Array Int64 :=
  if vals.size == target then
    vals
  else if vals.size > target then
    vals.extract 0 target
  else
    Id.run do
      let mut out := vals
      let pad := target - vals.size
      for _ in [:pad] do
        out := out.push 0
      out

/-- Faithful Whisper-style feature extraction on one waveform.
    This mirrors the reference path:
    Hann + STFT -> power -> Slaney mel -> log10 -> dynamic range clamp -> normalize. -/
def waveformToWhisperFeatures
    (cfg : PreprocessorConfig)
    (wave : Array Float)
    : IO (WhisperFrontendOutput cfg.featureSize (PreprocessorConfig.expectedFrames cfg)) := do
  let targetSamples := PreprocessorConfig.expectedSampleCount cfg
  let expectedFrames := PreprocessorConfig.expectedFrames cfg
  let nFft := if cfg.nFft == 0 then 400 else cfg.nFft
  let hop := if cfg.hopLength == 0 then 160 else cfg.hopLength
  let melBins := cfg.featureSize

  if melBins == 0 then
    throw <| IO.userError "Preprocessor config has feature_size=0"
  if targetSamples == 0 then
    throw <| IO.userError "Preprocessor config has n_samples=0"

  let (prepared0, validLenNat) := takePadRight wave targetSamples.toNat cfg.paddingValue
  let prepared :=
    if cfg.doNormalize then
      zeroMeanUnitVar prepared0 validLenNat cfg.paddingValue
    else
      prepared0

  let wav0 : T #[targetSamples] := reshape (data.fromFloatArray prepared) #[targetSamples]
  let wav ←
    if cfg.dither == 0.0 then
      pure wav0
    else
      let noise ← randn #[targetSamples]
      pure (wav0 + noise * cfg.dither)

  let window : T #[nFft] := signal.hannWindow nFft
  let stftDyn : T #[] := signal.stft1d (n := targetSamples) wav nFft hop nFft window true false
  let stftShape := stftDyn.runtimeShape
  if stftShape.size < 3 then
    throw <| IO.userError s!"Unexpected STFT rank {stftShape.size}, expected 3"

  let freqBins := stftShape.getD 0 0
  let stftFrames := stftShape.getD 1 0
  let packed := stftShape.getD 2 0
  if packed != 2 then
    throw <| IO.userError s!"Unexpected STFT packed size {packed}, expected 2"

  let droppedFrames := if stftFrames > 0 then stftFrames - 1 else 0
  if droppedFrames != expectedFrames then
    throw <| IO.userError
      s!"STFT frames mismatch: got={droppedFrames}, expected={expectedFrames} (stft_frames={stftFrames})"

  let stftVals ← data.tensorToFloatArray' stftDyn
  let freqNat := freqBins.toNat
  let stftFramesNat := stftFrames.toNat
  let framesNat := expectedFrames.toNat
  let packedNat := packed.toNat

  let mut powerFlat : Array Float := Array.mkEmpty (freqNat * framesNat)
  for f in [:freqNat] do
    for t in [:framesNat] do
      let base := ((f * stftFramesNat + t) * packedNat)
      let re := stftVals.getD base 0.0
      let im := stftVals.getD (base + 1) 0.0
      powerFlat := powerFlat.push (re * re + im * im)

  let melFilterFlat := buildSlaneyMelFilterBankFlat freqNat melBins.toNat cfg.samplingRate.toNat
  let powerSpec : T #[freqBins, expectedFrames] := reshape (data.fromFloatArray powerFlat) #[freqBins, expectedFrames]
  let melFilter : T #[freqBins, melBins] := reshape (data.fromFloatArray melFilterFlat) #[freqBins, melBins]
  let melPower : T #[melBins, expectedFrames] := nn.mm (nn.transpose2d melFilter) powerSpec
  let logSpec : T #[melBins, expectedFrames] := nn.log10 (clampFloat melPower (1e-10 : Float) (1e10 : Float))
  let maxVal := nn.item (nn.maxAll logSpec)
  let floored : T #[melBins, expectedFrames] := clampFloat logSpec (maxVal - (8.0 : Float)) (1e10 : Float)
  let normalized : T #[melBins, expectedFrames] := (floored + (4.0 : Float)) / (4.0 : Float)
  let inputFeatures : T #[1, melBins, expectedFrames] := reshape normalized #[1, melBins, expectedFrames]

  let maskVals := buildFeatureAttentionMask validLenNat targetSamples.toNat hop.toNat cfg.returnAttentionMask
  let featureAttentionMask : T #[1, expectedFrames] :=
    reshape (data.fromInt64Array maskVals) #[1, expectedFrames]

  pure { inputFeatures, featureAttentionMask }

/-- Dynamic-length Whisper-style feature extraction for full-wave inference.
    Unlike `waveformToWhisperFeatures`, this does not force 30s framing.
    It uses the full waveform length (with optional min/max duration constraints). -/
def waveformToWhisperFeaturesDynamic
    (cfg : PreprocessorConfig)
    (wave : Array Float)
    (minSeconds : Float := 0.5)
    (maxSeconds : Option Float := none)
    : IO (Sigma (fun frames => WhisperFrontendOutput cfg.featureSize frames)) := do
  let nFft := if cfg.nFft == 0 then 400 else cfg.nFft
  let hop := if cfg.hopLength == 0 then 160 else cfg.hopLength
  let melBins := cfg.featureSize

  if melBins == 0 then
    throw <| IO.userError "Preprocessor config has feature_size=0"
  if cfg.samplingRate == 0 then
    throw <| IO.userError "Preprocessor config has sampling_rate=0"

  let minSamplesRaw := ((minSeconds * cfg.samplingRate.toFloat) + 0.5).toUInt64
  let minSamples := if minSamplesRaw == 0 then 1 else minSamplesRaw
  let waveSamples := wave.size.toUInt64
  let targetSamples := if waveSamples >= minSamples then waveSamples else minSamples

  match maxSeconds with
  | none => pure ()
  | some sec =>
      if sec > 0.0 then
        let maxSamples := ((sec * cfg.samplingRate.toFloat) + 0.5).toUInt64
        if maxSamples > 0 && targetSamples > maxSamples then
          throw <| IO.userError
            s!"Audio duration exceeds maxSeconds={sec}: samples={targetSamples}, limit={maxSamples}"
      else
        pure ()

  let (prepared0, validLenNat) := takePadRight wave targetSamples.toNat cfg.paddingValue
  let prepared :=
    if cfg.doNormalize then
      zeroMeanUnitVar prepared0 validLenNat cfg.paddingValue
    else
      prepared0

  let wav0 : T #[targetSamples] := reshape (data.fromFloatArray prepared) #[targetSamples]
  let wav ←
    if cfg.dither == 0.0 then
      pure wav0
    else
      let noise ← randn #[targetSamples]
      pure (wav0 + noise * cfg.dither)

  let window : T #[nFft] := signal.hannWindow nFft
  let stftDyn : T #[] := signal.stft1d (n := targetSamples) wav nFft hop nFft window true false
  let stftShape := stftDyn.runtimeShape
  if stftShape.size < 3 then
    throw <| IO.userError s!"Unexpected STFT rank {stftShape.size}, expected 3"

  let freqBins := stftShape.getD 0 0
  let stftFrames := stftShape.getD 1 0
  let packed := stftShape.getD 2 0
  if packed != 2 then
    throw <| IO.userError s!"Unexpected STFT packed size {packed}, expected 2"

  let frames := if stftFrames > 0 then stftFrames - 1 else 0
  let stftVals ← data.tensorToFloatArray' stftDyn
  let freqNat := freqBins.toNat
  let stftFramesNat := stftFrames.toNat
  let framesNat := frames.toNat
  let packedNat := packed.toNat

  let mut powerFlat : Array Float := Array.mkEmpty (freqNat * framesNat)
  for f in [:freqNat] do
    for t in [:framesNat] do
      let base := ((f * stftFramesNat + t) * packedNat)
      let re := stftVals.getD base 0.0
      let im := stftVals.getD (base + 1) 0.0
      powerFlat := powerFlat.push (re * re + im * im)

  let melFilterFlat := buildSlaneyMelFilterBankFlat freqNat melBins.toNat cfg.samplingRate.toNat
  let powerSpec : T #[freqBins, frames] := reshape (data.fromFloatArray powerFlat) #[freqBins, frames]
  let melFilter : T #[freqBins, melBins] := reshape (data.fromFloatArray melFilterFlat) #[freqBins, melBins]
  let melPower : T #[melBins, frames] := nn.mm (nn.transpose2d melFilter) powerSpec
  let logSpec : T #[melBins, frames] := nn.log10 (clampFloat melPower (1e-10 : Float) (1e10 : Float))
  let maxVal := nn.item (nn.maxAll logSpec)
  let floored : T #[melBins, frames] := clampFloat logSpec (maxVal - (8.0 : Float)) (1e10 : Float)
  let normalized : T #[melBins, frames] := (floored + (4.0 : Float)) / (4.0 : Float)
  let inputFeatures : T #[1, melBins, frames] := reshape normalized #[1, melBins, frames]

  let maskVals0 := buildFeatureAttentionMask validLenNat targetSamples.toNat hop.toNat cfg.returnAttentionMask
  let maskVals := adjustMaskLength maskVals0 frames.toNat
  let featureAttentionMask : T #[1, frames] :=
    reshape (data.fromInt64Array maskVals) #[1, frames]

  pure ⟨frames, { inputFeatures, featureAttentionMask }⟩

/-- Load WAV, resample to preprocessor sampling rate, then extract Whisper-style features. -/
def wavToWhisperFeatures
    (cfg : PreprocessorConfig)
    (path : String)
    : IO (WhisperFrontendOutput cfg.featureSize (PreprocessorConfig.expectedFrames cfg)) := do
  let (sr, wav) ← loadMonoPcm16Wav path
  let wavTarget ← data.resampleSoxrHQ wav sr cfg.samplingRate
  waveformToWhisperFeatures cfg wavTarget

/-- Dynamic-length WAV frontend path.
    Uses full resampled waveform length (subject to optional min/max seconds). -/
def wavToWhisperFeaturesDynamic
    (cfg : PreprocessorConfig)
    (path : String)
    (minSeconds : Float := 0.5)
    (maxSeconds : Option Float := none)
    : IO (Sigma (fun frames => WhisperFrontendOutput cfg.featureSize frames)) := do
  let (sr, wav) ← loadMonoPcm16Wav path
  let wavTarget ← data.resampleSoxrHQ wav sr cfg.samplingRate
  waveformToWhisperFeaturesDynamic cfg wavTarget minSeconds maxSeconds

/-- Backward-compatible convenience wrapper.
    This now runs the faithful Whisper-style frontend instead of pseudo-mel. -/
def wavToPseudoMel {melBins frames : UInt64} (path : String) : IO (T #[1, melBins, frames]) := do
  let cfg : PreprocessorConfig := {
    featureSize := melBins
    samplingRate := 16000
    hopLength := 160
    nFft := 400
    nSamples := frames * 160
    nbMaxFrames := frames
    returnAttentionMask := true
    doNormalize := false
    dither := 0.0
  }
  let out ← wavToWhisperFeatures cfg path
  pure out.inputFeatures

def fullFeatureAttentionMask {frames : UInt64} : T #[1, frames] :=
  full_int #[1, frames] 1

def normalizeAudioTo16kFromWav (path : String) : IO (Array Float) := do
  let (sr, wav) ← loadMonoPcm16Wav path
  data.resampleSoxrHQ wav sr 16000

end torch.qwen3asr
