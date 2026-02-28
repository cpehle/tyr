/-
  Tyr/Model/Qwen3/Model.lean

  Standalone Qwen3 causal-LM built on shared Qwen transformer blocks.
  Includes:
  - LM head
  - masked/unmasked forward
  - incremental KV-cache decode
  - greedy generation (cached + uncached)
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen
import Tyr.Model.Qwen3.Config

namespace torch.qwen3

open torch

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

private def allInSet (xs : Array UInt64) (allow : Array UInt64) : Bool :=
  xs.all (fun x => allow.contains x)

/-- Causal-LM output container. -/
structure CausalLMOutput (cfg : Config) (batch seq : UInt64) where
  logits : T #[batch, seq, cfg.vocab_size]
  hiddenStates : Option (Array (T #[batch, seq, cfg.hidden_size])) := none

/-- Full standalone Qwen3 causal language model. -/
structure Qwen3ForCausalLM (cfg : Config) where
  model : qwen.Qwen3Model cfg
  lmHead : T #[cfg.vocab_size, cfg.hidden_size]
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Qwen3ForCausalLM

/-- Initialize model + LM head. By default, LM head is tied to token embeddings. -/
def init (cfg : Config) (tieWordEmbeddings : Bool := true) : IO (Qwen3ForCausalLM cfg) := do
  let model ← qwen.Qwen3Model.init cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initWeight #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model, lmHead, tieWordEmbeddings }

/-- Token embedding lookup. -/
def embedTokens {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  nn.embedding inputIds m.model.embed_tokens

/-- Forward from pre-computed input embeddings. -/
def forwardEmbeds {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocab_size] :=
  let (cos, sin) := rotary.computeFreqsPure seq cfg.head_dim cfg.rope_theta
  let hidden :=
    match attnMask with
    | some mask =>
      m.model.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.model.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  let hidden := m.model.norm.forward3d hidden
  linear3d hidden m.lmHead

/-- Standard forward pass from token IDs. -/
def forward {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocab_size] :=
  m.forwardEmbeds (m.embedTokens inputIds) attnMask

/-- Forward pass that also returns per-layer hidden states. -/
def forwardWithHiddenStates {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : CausalLMOutput cfg batch seq :=
  let (cos, sin) := rotary.computeFreqsPure seq cfg.head_dim cfg.rope_theta
  let hiddenStates :=
    match attnMask with
    | some mask => m.model.forwardWithHiddenStatesMasked cfg inputIds cos sin mask
    | none => m.model.forwardWithHiddenStates cfg inputIds cos sin
  let finalHidden : T #[batch, seq, cfg.hidden_size] :=
    hiddenStates.getD (hiddenStates.size - 1) (torch.zeros #[batch, seq, cfg.hidden_size])
  let logits := linear3d finalHidden m.lmHead
  { logits, hiddenStates := some hiddenStates }

/-- One-layer KV cache type for cached decoding. -/
abbrev LayerKVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch cfg.num_key_value_heads cfg.head_dim

private def initLayerKVCaches {batch : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache cfg batch) :=
  m.model.layers.map (fun _ =>
    qwen.QwenAttention.initKVCache
      maxLen
      (batch := batch)
      (num_kv_heads := cfg.num_key_value_heads)
      (head_dim := cfg.head_dim)
      device)

private def decodeStepFromEmbedWithCache {batch : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (tokenEmbed : T #[batch, 1, cfg.hidden_size])
    (position : UInt64)
    (caches : Array (LayerKVCache cfg batch))
    : IO (T #[batch, cfg.vocab_size] × Array (LayerKVCache cfg batch)) := do
  let freqLen := position + 1
  let (cosAll, sinAll) := rotary.computeFreqsPure freqLen cfg.head_dim cfg.rope_theta
  let cos : T #[1, cfg.head_dim / 2] := data.slice cosAll 0 position 1
  let sin : T #[1, cfg.head_dim / 2] := data.slice sinAll 0 position 1

  let mut hidden : T #[batch, 1, cfg.hidden_size] := tokenEmbed
  let mut nextCaches := caches

  for i in [:m.model.layers.size] do
    let layer ←
      match m.model.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Qwen3 layer at index {i}"
    let cache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Qwen3 KV cache at index {i}"
    let (hNext, cNext) := layer.forwardStep hidden cos sin cache
    hidden := hNext
    nextCaches := nextCaches.set! i cNext

  let hiddenNorm := m.model.norm.forward3d hidden
  let logits3 : T #[batch, 1, cfg.vocab_size] := linear3d hiddenNorm m.lmHead
  let logits2 : T #[batch, cfg.vocab_size] := reshape logits3 #[batch, cfg.vocab_size]
  pure (logits2, nextCaches)

private partial def prefillCachesFromEmbeds {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (caches : Array (LayerKVCache cfg batch))
    (position : Nat)
    (lastLogits : T #[batch, cfg.vocab_size])
    : IO (T #[batch, cfg.vocab_size] × Array (LayerKVCache cfg batch)) := do
  if position >= seq.toNat then
    pure (lastLogits, caches)
  else
    let tok : T #[batch, 1, cfg.hidden_size] := data.slice inputsEmbeds 1 position.toUInt64 1
    let (logits, caches') ← decodeStepFromEmbedWithCache m tok position.toUInt64 caches
    prefillCachesFromEmbeds m inputsEmbeds caches' (position + 1) logits

private partial def greedyLoopCached {batch : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (eosTokenIds : Array UInt64)
    (remaining : Nat)
    (caches : Array (LayerKVCache cfg batch))
    (lastLogits : T #[batch, cfg.vocab_size])
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTok : T #[batch] := nn.argmax lastLogits 1
  let nextVals ← data.tensorToUInt64Array nextTok
  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    let nextEmb : T #[batch, 1, cfg.hidden_size] := m.embedTokens nextCol
    let (nextLogits, caches') ← decodeStepFromEmbedWithCache m nextEmb curSeq caches
    greedyLoopCached m eosTokenIds (remaining - 1) caches' nextLogits appended

private partial def greedyLoopUncached {batch : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (eosTokenIds : Array UInt64)
    (remaining : Nat)
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"

  let logits := m.forward curIds none
  let lastPos := curSeq - 1
  let last3 : T #[batch, 1, cfg.vocab_size] :=
    reshape (data.slice logits 1 lastPos 1) #[batch, 1, cfg.vocab_size]
  let last2 : T #[batch, cfg.vocab_size] := reshape last3 #[batch, cfg.vocab_size]
  let nextTok : T #[batch] := nn.argmax last2 1
  let nextVals ← data.tensorToUInt64Array nextTok
  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    greedyLoopUncached m eosTokenIds (remaining - 1) appended

/-- Greedy generation with static per-layer KV caches. -/
def generateGreedy {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let inputsEmbeds := m.embedTokens inputIds
  let cacheDevice := inputsEmbeds.device
  let cacheMaxLen : UInt64 := seq + maxNewTokens
  let caches0 := initLayerKVCaches m cacheMaxLen cacheDevice
  let tok0 : T #[batch, 1, cfg.hidden_size] := data.slice inputsEmbeds 1 0 1
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m inputsEmbeds caches1 1 logits0
  greedyLoopCached m eosTokenIds maxNewTokens.toNat cachesPrefill lastLogits inputIds

/-- Reference greedy generation by full re-forward on each decode step. -/
def generateGreedyUncached {batch seq : UInt64}
    (m : Qwen3ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  greedyLoopUncached m eosTokenIds maxNewTokens.toNat inputIds

end Qwen3ForCausalLM

end torch.qwen3
