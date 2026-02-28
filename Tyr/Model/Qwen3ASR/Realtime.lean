import Tyr.Model.Qwen3ASR.Transcribe
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

/-- Coherent transcript assembly state for overlap-window streaming.
    `stableText` is append-only; `unstableText` may change each hop. -/
structure RealtimeTranscriptState where
  rollbackChars : Nat := 12
  committedChars : Nat := 0
  stableText : String := ""
  unstableText : String := ""
  prevHypothesis : String := ""
  deriving Repr, Inhabited

/-- Delta emitted per decode step. -/
structure TranscriptDelta where
  stableAppend : String := ""
  unstableText : String := ""
  fullText : String := ""
  deriving Repr, Inhabited

private def toChars (s : String) : Array Char := s.toList.toArray
private def fromChars (xs : Array Char) : String := String.ofList xs.toList

private def commonPrefixLen (a b : String) : Nat :=
  let xa := toChars a
  let xb := toChars b
  let n := Nat.min xa.size xb.size
  Id.run do
    let mut i := 0
    while i < n && xa[i]! == xb[i]! do
      i := i + 1
    i

private def prefixChars (s : String) (n : Nat) : String :=
  fromChars <| (toChars s).extract 0 (Nat.min n (toChars s).size)

private def suffixChars (s : String) (start : Nat) : String :=
  let xs := toChars s
  if start >= xs.size then "" else fromChars (xs.extract start xs.size)

private def safeSub (a b : Nat) : Nat := if a >= b then a - b else 0

/-- Merge a fresh hypothesis into coherent `(stable, unstable)` form. -/
def updateTranscriptState
    (st : RealtimeTranscriptState)
    (hypothesis : String)
    : RealtimeTranscriptState × TranscriptDelta :=
  let lcpPrev := commonPrefixLen st.prevHypothesis hypothesis
  let commitCandidate := safeSub lcpPrev st.rollbackChars
  let commitChars := Nat.max st.committedChars commitCandidate
  let stable := prefixChars hypothesis commitChars
  let unstable := suffixChars hypothesis commitChars
  let oldStableLen := (toChars st.stableText).size
  let newStableLen := (toChars stable).size
  let stableAppend := if newStableLen > oldStableLen then suffixChars stable oldStableLen else ""
  let st' : RealtimeTranscriptState := {
    st with
    committedChars := commitChars
    stableText := stable
    unstableText := unstable
    prevHypothesis := hypothesis
  }
  let delta : TranscriptDelta := {
    stableAppend := stableAppend
    unstableText := unstable
    fullText := stable ++ unstable
  }
  (st', delta)

/-- Decode one overlap window and return coherent transcript deltas.
    This is complementary to existing streaming helpers and adds no overhead
    unless called by a realtime loop. -/
def decodeOverlapStep
    {cfg : Qwen3ASRConfig}
    (m : Qwen3ASRForConditionalGeneration cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio16kWindow : Array Float)
    (state : RealtimeTranscriptState)
    (context : String := "")
    (language : Option String := none)
    (maxNewTokens : UInt64 := 128)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (RealtimeTranscriptState × TranscriptDelta) := do
  let out ← m.transcribeWaveform
    tok
    preprocessor
    audio16kWindow
    (context := context)
    (language := language)
    (returnTimeStamps := false)
    (maxNewTokens := maxNewTokens)
    (eosTokenIds := eosTokenIds)
  pure (updateTranscriptState state out.text)

end torch.qwen3asr
