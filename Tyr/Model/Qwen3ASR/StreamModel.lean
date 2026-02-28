import Tyr.Model.Qwen3ASR.ConfigIO
import Tyr.Model.Qwen3ASR.Weights
import Tyr.Model.Qwen3ASR.Transcribe
import Tyr.Text.StreamingConsensus
import Tyr.Text.VADProvider
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

/-- Separate streaming-native model wrapper.
    This is intentionally parallel to the existing model API so baseline paths
    keep zero additional runtime overhead. -/
structure StreamModel where
  cfg : Qwen3ASRConfig
  model : Qwen3ASRForConditionalGeneration cfg
  tok : tokenizer.qwen3.QwenTokenizer
  preprocessor : PreprocessorConfig
  modelDir : String

/-- Streaming session mutable state.
    `encoderCachedFrames`/`decoderCachedTokens` are reserved for true
    incremental cache plumbing; current step uses overlap fallback while the
    cache-backed path is implemented. -/
structure StreamSession where
  chunkSec : Float := 2.0
  hopSec : Float := 0.5
  chunkSamples : Nat := 32000
  hopSamples : Nat := 8000
  ring : Array Float := #[]
  sinceLastDecode : Nat := 0
  context : String := ""
  language : Option String := none
  textConsensus : Tyr.Text.ConsensusState := {}
  sileroVAD : Option Tyr.Text.SileroProvider := none
  encoderCachedFrames : UInt64 := 0
  decoderCachedTokens : UInt64 := 0

structure StreamStepOutput where
  didDecode : Bool := false
  stableAppend : String := ""
  unstableText : String := ""
  fullText : String := ""
  mode : String := "fallback_overlap"

private def toSamples (sec : Float) : Nat :=
  let n := ((sec * 16000.0) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

private def tailSlice (xs : Array Float) (n : Nat) : Array Float :=
  if xs.size <= n then xs else xs.extract (xs.size - n) xs.size

def loadFromPretrained (modelDir : String) : IO StreamModel := do
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let preprocessor ← PreprocessorConfig.loadFromPretrainedDir modelDir
  let model ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  pure { cfg, model, tok, preprocessor, modelDir }

def newSession
    (_m : StreamModel)
    (chunkSec : Float := 2.0)
    (hopSec : Float := 0.5)
    (context : String := "")
    (language : Option String := none)
    (historyWindows : Nat := 6)
    (confirmWindows : Nat := 3)
    (rollbackTokens : Nat := 2)
    (freezeAfterSteps : Nat := 4)
    (mutableTailTokensWhileSpeech : Nat := 6)
    (sileroVADPath : Option String := none)
    : IO StreamSession := do
  if chunkSec <= 0.0 || hopSec <= 0.0 then
    throw <| IO.userError "chunkSec and hopSec must be > 0"
  let cCfg : Tyr.Text.ConsensusConfig := {
    historyWindows := historyWindows
    confirmWindows := confirmWindows
    rollbackTokens := rollbackTokens
    freezeAfterSteps := freezeAfterSteps
    mutableTailTokensWhileSpeech := mutableTailTokensWhileSpeech
  }
  let vad ←
    match sileroVADPath with
    | some p =>
      if p.isEmpty then
        pure none
      else
        pure (some (← Tyr.Text.initSileroProvider p))
    | none => pure none
  pure {
    chunkSec := chunkSec
    hopSec := hopSec
    chunkSamples := toSamples chunkSec
    hopSamples := toSamples hopSec
    context := context
    language := language
    textConsensus := { cfg := cCfg }
    sileroVAD := vad
  }

/-- Push PCM16k samples and optionally decode one step.
    This currently uses overlap decode fallback while true incremental
    encoder/KV cache path is being introduced in this separate codepath. -/
def pushAudio
    (m : StreamModel)
    (s : StreamSession)
    (pcm16k : Array Float)
    (maxNewTokens : UInt64 := 128)
    : IO (StreamSession × StreamStepOutput) := do
  let ring := s.ring ++ pcm16k
  let since := s.sinceLastDecode + pcm16k.size
  let mut s' := { s with ring := ring, sinceLastDecode := since }
  let vadSignal : Tyr.Text.VADSignal ←
    match s'.sileroVAD with
    | some p =>
      let (p', sig) ← Tyr.Text.stepSileroProvider p pcm16k
      s' := { s' with sileroVAD := some p' }
      pure sig
    | none =>
      pure { speechActive := true, boundary := false }

  if s'.ring.size < s'.chunkSamples || s'.sinceLastDecode < s'.hopSamples then
    pure (s', { didDecode := false, stableAppend := "", unstableText := "", fullText := "", mode := "fallback_overlap" })
  else
    let window := tailSlice s'.ring s'.chunkSamples
    let out ← m.model.transcribeWaveform
      m.tok
      m.preprocessor
      window
      (context := s'.context)
      (language := s'.language)
      (returnTimeStamps := false)
      (maxNewTokens := maxNewTokens)
    let ids := tokenizer.qwen3.encodeText m.tok out.text
    let (cs', delta) := Tyr.Text.updateWithSignals
      s'.textConsensus ids vadSignal.speechActive vadSignal.boundary (tokenizer.qwen3.decodeText m.tok)
    s' := {
      s' with
      textConsensus := cs'
      sinceLastDecode := 0
      encoderCachedFrames := s'.encoderCachedFrames + s'.hopSamples.toUInt64
    }
    pure (s', {
      didDecode := true
      stableAppend := delta.stableAppend
      unstableText := delta.unstableText
      fullText := delta.fullText
      mode := "fallback_overlap"
    })

def flush
    (m : StreamModel)
    (s : StreamSession)
    (maxNewTokens : UInt64 := 128)
    : IO (StreamSession × StreamStepOutput) := do
  if s.ring.isEmpty then
    pure (s, { didDecode := false, stableAppend := "", unstableText := "", fullText := "", mode := "fallback_overlap" })
  else
    let window := tailSlice s.ring s.chunkSamples
    let out ← m.model.transcribeWaveform
      m.tok
      m.preprocessor
      window
      (context := s.context)
      (language := s.language)
      (returnTimeStamps := false)
      (maxNewTokens := maxNewTokens)
    let ids := tokenizer.qwen3.encodeText m.tok out.text
    let (cs', delta) := Tyr.Text.updateWithSignals
      s.textConsensus ids false true (tokenizer.qwen3.decodeText m.tok)
    let s' := { s with textConsensus := cs' }
    pure (s', {
      didDecode := true
      stableAppend := delta.stableAppend
      unstableText := delta.unstableText
      fullText := delta.fullText
      mode := "fallback_overlap"
    })

end torch.qwen3asr
