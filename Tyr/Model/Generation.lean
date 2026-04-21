/-
  Tyr/Model/Generation.lean

  Shared generation infrastructure for causal language models:
  sampling strategies, logits filtering, and token-selection helpers.
-/
import Tyr.Torch

namespace torch.Model

open torch

/-! ## Sampling Strategy -/

inductive SamplingStrategy where
  | greedy
  | multinomial (temperature : Float := 1.0) (topK : UInt64 := 0) (topP : Float := 1.0)
  deriving Repr, Inhabited

/-- Callback invoked on every generated token during streaming decode. -/
abbrev StreamCallback (batch : UInt64) := UInt64 → T #[batch] → IO Unit

/-! ## Token Selection -/

/-- Sample a batch of next tokens from logits according to a strategy. -/
def sampleFromLogits {batch vocab : UInt64}
    (logits : T #[batch, vocab])
    (strategy : SamplingStrategy)
    : IO (T #[batch]) := do
  match strategy with
  | .greedy =>
    pure (nn.argmax logits 1)
  | .multinomial temperature topK topP =>
    if temperature <= 0.0 then
      throw <| IO.userError s!"multinomial sampling requires temperature > 0, got {temperature}"
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

/-! ## EOS / Finish Helpers -/

/-- Return a boolean mask of tokens that belong to `values`. -/
def tokenInSet {n : UInt64}
    (tokens : T #[n])
    (values : Array UInt64)
    : T #[n] :=
  Id.run do
    let mut mask := falseMask (n := n) tokens.device
    for value in values do
      let hit : T #[n] := eq_scalar tokens (Int64.ofNat value.toNat)
      mask := logicalOr mask hit
    return mask

/-- Replace tokens with `eosToken` wherever `finished` is true. -/
def applyFinishedEos {n : UInt64}
    (tokens : T #[n])
    (finished : T #[n])
    (eosToken : UInt64)
    : T #[n] :=
  let eos : T #[n] := (full_int #[n] (Int64.ofNat eosToken.toNat)).to tokens.device
  where_ finished eos tokens

end torch.Model
