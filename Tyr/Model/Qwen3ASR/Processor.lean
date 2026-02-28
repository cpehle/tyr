/-
  Tyr/Model/Qwen3ASR/Processor.lean

  Lean-native port of the key algorithmic pieces from
  `processing_qwen3_asr.py` (without Python/HF dependencies):
  - multimodal special-token replacement
  - feature-length conversion helpers
  - chunk-index utilities
-/
import Tyr.Model.Qwen3ASR.Config

namespace torch.qwen3asr

/-- Default processor kwargs mirrored from reference processor defaults. -/
structure ProcessorKwargs where
  textPadding : Bool := false
  textPaddingSide : String := "left"
  audioSamplingRate : UInt64 := 16000
  audioPadding : Bool := true
  audioReturnAttentionMask : Bool := true
  deriving Repr, Inhabited

/-- Lightweight Lean processor config/state. -/
structure Qwen3ASRProcessor where
  audioToken : String := "<|audio_pad|>"
  audioBosToken : String := "<|audio_start|>"
  audioEosToken : String := "<|audio_end|>"
  kwargs : ProcessorKwargs := {}
  deriving Repr, Inhabited

namespace Qwen3ASRProcessor

private def repeatToken (tok : String) (n : Nat) : String :=
  Id.run do
    let mut out := ""
    for _ in [:n] do
      out := out ++ tok
    out

/-- Port of processor-side helper.
    Converts raw feature lengths to audio-token lengths after conv stack. -/
def featExtractOutputLengths (inputLens : Array UInt64) : Array UInt64 :=
  AudioEncoderConfig.featExtractOutputLengths inputLens

private def replaceInSample
    (audioToken : String)
    (sample : String)
    (audioLengths : Array UInt64)
    (startIdx : Nat)
    : Except String (String × Nat) := do
  let parts := (sample.splitOn audioToken).toArray
  if parts.size ≤ 1 then
    return (sample, startIdx)

  let mut idx := startIdx
  let mut out := parts[0]!
  for i in [1:parts.size] do
    if h : idx < audioLengths.size then
      let aLen := audioLengths[idx]!
      out := out ++ repeatToken audioToken aLen.toNat ++ parts[i]!
      idx := idx + 1
    else
      throw s!"Not enough audio lengths while replacing `{audioToken}` placeholders"
  return (out, idx)

/-- Port of `replace_multimodal_special_tokens` for audio placeholders.
    Each `audioToken` occurrence is replaced by repeating `audioToken`
    `audio_length` times for the corresponding audio in stream order. -/
def replaceMultimodalSpecialTokens
    (p : Qwen3ASRProcessor)
    (text : Array String)
    (audioLengths : Array UInt64)
    : Except String (Array String) := do
  let mut out : Array String := #[]
  let mut idx : Nat := 0
  for sample in text do
    let (sample', idx') ← replaceInSample p.audioToken sample audioLengths idx
    out := out.push sample'
    idx := idx'
  return out

/-- Port of processor chunking helper.
    Splits monotonically increasing token indices into value-range chunks. -/
def getChunkedIndex (tokenIndices : Array UInt64) (tokensPerChunk : UInt64) : Array (Nat × Nat) :=
  if tokensPerChunk == 0 then
    #[(0, tokenIndices.size)]
  else
    Id.run do
      let mut out : Array (Nat × Nat) := #[]
      let mut i : Nat := 0
      let mut startIdx : Nat := 0
      let mut currentChunk : UInt64 := 1
      while i < tokenIndices.size do
        if tokenIndices[i]! >= currentChunk * tokensPerChunk then
          out := out.push (startIdx, i)
          startIdx := i
          currentChunk := currentChunk + 1
        i := i + 1
      out.push (startIdx, tokenIndices.size)

/-- Equivalent of processor `model_input_names` merge.
    We expose the canonical multimodal names used by the Lean port. -/
def modelInputNames (_p : Qwen3ASRProcessor) : Array String :=
  #["input_ids", "attention_mask", "input_features", "feature_attention_mask"]

end Qwen3ASRProcessor

end torch.qwen3asr
