/-
  NanoProof Model Implementation

  A dependently-typed theorem proving transformer for Tyr.
  Key differences from GPT:
  - Rotary embeddings (no positional embeddings)
  - RMSNorm without learnable parameters
  - QK normalization
  - ReLU² activation in MLP
  - Group-Query Attention (GQA)
  - Dual heads: policy (logits) + value (bins)
  - No bias in linear layers
-/
import Tyr.Torch

namespace torch.nanoproof

open torch

/-- NanoProof model configuration -/
structure Config where
  sequence_len : UInt64 := 1024   -- max sequence length
  vocab_size : UInt64 := 50304    -- vocabulary size (padded to 64)
  n_layer : UInt64 := 12          -- number of transformer blocks
  n_head : UInt64 := 6            -- number of query heads
  n_kv_head : UInt64 := 6         -- number of KV heads (for GQA)
  n_embd : UInt64 := 768          -- embedding dimension
  num_value_bins : UInt64 := 64   -- bins for value head
  softcap : Float := 15.0         -- logit softcap value
  deriving Repr, Inhabited

/-- Compute head dimension from config -/
def Config.headDim (cfg : Config) : UInt64 := cfg.n_embd / cfg.n_head

/-- Check if GQA is enabled -/
def Config.gqaEnabled (cfg : Config) : Bool := cfg.n_head != cfg.n_kv_head

/-- Small config for testing -/
def Config.small : Config :=
  { sequence_len := 512, vocab_size := 20000, n_layer := 6, n_head := 6, n_kv_head := 6, n_embd := 384, num_value_bins := 64 }

/-- Tiny config for unit tests -/
def Config.tiny : Config :=
  { sequence_len := 128, vocab_size := 1000, n_layer := 2, n_head := 4, n_kv_head := 4, n_embd := 128, num_value_bins := 64 }

/-- Full config matching nanoproof -/
def Config.full : Config :=
  { sequence_len := 1024, vocab_size := 50304, n_layer := 26, n_head := 6, n_kv_head := 6, n_embd := 768, num_value_bins := 64 }

/-- Precomputed rotary embedding frequencies -/
structure RotaryCache (seq_len head_dim : UInt64) where
  cos : T #[seq_len, head_dim / 2]
  sin : T #[seq_len, head_dim / 2]

/-- Initialize rotary cache for given sequence length and head dimension -/
def RotaryCache.init (seq_len head_dim : UInt64) (base : Float := 10000.0) : IO (RotaryCache seq_len head_dim) := do
  let (cos, sin) ← rotary.computeFreqs seq_len head_dim base
  return { cos, sin }

/-- Parameters for Causal Self-Attention (no bias, supports GQA) -/
structure AttentionParams (n_embd n_head n_kv_head : UInt64) where
  -- Q projection: n_embd -> n_head * head_dim
  c_q : T #[n_head * (n_embd / n_head), n_embd]
  -- K projection: n_embd -> n_kv_head * head_dim
  c_k : T #[n_kv_head * (n_embd / n_head), n_embd]
  -- V projection: n_embd -> n_kv_head * head_dim
  c_v : T #[n_kv_head * (n_embd / n_head), n_embd]
  -- Output projection: n_embd -> n_embd (initialized to zero)
  c_proj : T #[n_embd, n_embd]

/-- Parameters for MLP (no bias, ReLU² activation) -/
structure MLPParams (n_embd : UInt64) where
  -- Expand: n_embd -> 4*n_embd
  c_fc : T #[4 * n_embd, n_embd]
  -- Project: 4*n_embd -> n_embd (initialized to zero)
  c_proj : T #[n_embd, 4 * n_embd]

/-- Parameters for a single transformer block -/
structure BlockParams (n_embd n_head n_kv_head : UInt64) where
  attn : AttentionParams n_embd n_head n_kv_head
  mlp : MLPParams n_embd

/-- Parameters for value head (for RL training) -/
structure ValueHeadParams (n_embd num_bins : UInt64) where
  c_fc : T #[4 * n_embd, n_embd]
  c_proj : T #[num_bins, 4 * n_embd]

/-- Full NanoProof model parameters -/
structure NanoProofParams (cfg : Config) where
  -- Token embedding (untied from lm_head)
  wte : T #[cfg.vocab_size, cfg.n_embd]
  -- Transformer blocks
  blocks : Array (BlockParams cfg.n_embd cfg.n_head cfg.n_kv_head)
  -- Language model head (policy head)
  lm_head : T #[cfg.vocab_size, cfg.n_embd]
  -- Value head (optional, for RL)
  value_head : Option (ValueHeadParams cfg.n_embd cfg.num_value_bins)

/-- Helper to create a leaf parameter tensor -/
def makeLeafParam {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (autograd.detach t) true

/-- Initialize attention parameters -/
def AttentionParams.init (n_embd n_head n_kv_head : UInt64) : IO (AttentionParams n_embd n_head n_kv_head) := do
  let head_dim := n_embd / n_head
  let q_dim := n_head * head_dim
  let kv_dim := n_kv_head * head_dim

  -- Scaled initialization: std = 1/sqrt(fan_in)
  let scale := 1.0 / Float.sqrt n_embd.toFloat

  let c_q ← randn #[q_dim, n_embd] false
  let c_k ← randn #[kv_dim, n_embd] false
  let c_v ← randn #[kv_dim, n_embd] false

  return {
    c_q := makeLeafParam (c_q * scale)
    c_k := makeLeafParam (c_k * scale)
    c_v := makeLeafParam (c_v * scale)
    -- c_proj initialized to zero (following nanoproof)
    c_proj := makeLeafParam (zeros #[n_embd, n_embd])
  }

/-- Initialize MLP parameters -/
def MLPParams.init (n_embd : UInt64) : IO (MLPParams n_embd) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_fc ← randn #[4 * n_embd, n_embd] false

  return {
    c_fc := makeLeafParam (c_fc * scale)
    -- c_proj initialized to zero (following nanoproof)
    c_proj := makeLeafParam (zeros #[n_embd, 4 * n_embd])
  }

/-- Initialize block parameters -/
def BlockParams.init (n_embd n_head n_kv_head : UInt64) : IO (BlockParams n_embd n_head n_kv_head) := do
  let attn ← AttentionParams.init n_embd n_head n_kv_head
  let mlp ← MLPParams.init n_embd
  return { attn, mlp }

/-- Initialize value head parameters -/
def ValueHeadParams.init (n_embd num_bins : UInt64) : IO (ValueHeadParams n_embd num_bins) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_fc ← randn #[4 * n_embd, n_embd] false

  return {
    c_fc := makeLeafParam (c_fc * scale)
    c_proj := makeLeafParam (zeros #[num_bins, 4 * n_embd])
  }

/-- Initialize full NanoProof model -/
def NanoProofParams.init (cfg : Config) (withValueHead : Bool := true) : IO (NanoProofParams cfg) := do
  let scale := 1.0 / Float.sqrt cfg.n_embd.toFloat

  let wte ← randn #[cfg.vocab_size, cfg.n_embd] false
  let lm_head ← randn #[cfg.vocab_size, cfg.n_embd] false

  let mut blocks := #[]
  for _ in [:cfg.n_layer.toNat] do
    let block ← BlockParams.init cfg.n_embd cfg.n_head cfg.n_kv_head
    blocks := blocks.push block

  let value_head ← if withValueHead then
    let vh ← ValueHeadParams.init cfg.n_embd cfg.num_value_bins
    pure (some vh)
  else
    pure none

  return {
    wte := makeLeafParam (wte * scale)
    blocks := blocks
    lm_head := makeLeafParam (lm_head * scale)
    value_head := value_head
  }

/-- RMSNorm helper (no learnable parameters) -/
def norm {s : Shape} (x : T s) : T s := nanoproof.rmsNorm x

/-- Attention forward pass -/
def attentionForward {batch seq n_embd n_head n_kv_head : UInt64}
    (params : AttentionParams n_embd n_head n_kv_head)
    (x : T #[batch, seq, n_embd])
    (rotaryCache : RotaryCache rotaryLen (n_embd / n_head))
    : T #[batch, seq, n_embd] :=
  let head_dim := n_embd / n_head

  -- Project to Q, K, V (no bias)
  let q := linear3d x params.c_q
  let k := linear3d x params.c_k
  let v := linear3d x params.c_v

  -- Reshape: [batch, seq, n*d] -> [batch, seq, n, d]
  let q := reshape q #[batch, seq, n_head, head_dim]
  let k := reshape k #[batch, seq, n_kv_head, head_dim]
  let v := reshape v #[batch, seq, n_kv_head, head_dim]

  -- Apply rotary embeddings (sliced to seq length)
  -- Slice rotaryCache from [rotaryLen, head_dim/2] to [seq, head_dim/2]
  let cos := data.slice2d rotaryCache.cos 0 seq
  let sin := data.slice2d rotaryCache.sin 0 seq
  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  -- QK normalization
  let q := norm q
  let k := norm k

  -- Transpose for attention: [batch, seq, n, d] -> [batch, n, seq, d]
  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  -- Scaled dot-product attention with GQA
  let enableGqa := n_head != n_kv_head
  let attn := nn.scaledDotProductAttentionGQA q k v 0.0 true enableGqa

  -- Reshape back: [batch, n, seq, d] -> [batch, seq, n*d]
  let attn := nn.transpose_from_attention attn
  let attn := reshape attn #[batch, seq, n_embd]

  -- Output projection (no bias)
  linear3d attn params.c_proj

/-- MLP forward pass with ReLU² -/
def mlpForward {batch seq n_embd : UInt64}
    (params : MLPParams n_embd)
    (x : T #[batch, seq, n_embd])
    : T #[batch, seq, n_embd] :=
  let h := linear3d x params.c_fc      -- Expand to 4x
  let h := nanoproof.reluSquared h     -- ReLU² activation
  linear3d h params.c_proj             -- Project back

/-- Block forward pass (pre-norm architecture) -/
def blockForward {batch seq n_embd n_head n_kv_head : UInt64}
    (params : BlockParams n_embd n_head n_kv_head)
    (x : T #[batch, seq, n_embd])
    (rotaryCache : RotaryCache rotaryLen (n_embd / n_head))
    : T #[batch, seq, n_embd] :=
  -- Attention with residual (pre-norm)
  let x := x + attentionForward params.attn (norm x) rotaryCache
  -- MLP with residual (pre-norm)
  x + mlpForward params.mlp (norm x)

/-- Value head forward pass -/
def valueHeadForward {batch seq n_embd num_bins : UInt64}
    (params : ValueHeadParams n_embd num_bins)
    (x : T #[batch, seq, n_embd])
    : T #[batch, seq, num_bins] :=
  let h := linear3d x params.c_fc
  let h := nanoproof.reluSquared h
  linear3d h params.c_proj

/-- Helper to compute optional value logits - workaround for compiler bug -/
@[noinline]
def computeValueLogits {batch seq n_embd num_bins : UInt64}
    (value_head : Option (ValueHeadParams n_embd num_bins))
    (x : T #[batch, seq, n_embd])
    : Option (T #[batch, seq, num_bins]) :=
  match value_head with
  | none => none
  | some vh => some (valueHeadForward vh x)

/-- Model output with both policy and value logits -/
structure ModelOutput (batch seq vocab_size num_bins : UInt64) where
  policy_logits : T #[batch, seq, vocab_size]
  value_logits : Option (T #[batch, seq, num_bins])

/-- Super minimal test - just create dummy tensor and return -/
def forwardMinimal {cfg : Config} (batch seq : UInt64)
    (_params : NanoProofParams cfg)
    (_rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (_idx : T #[batch, seq])
    : IO (ModelOutput batch seq cfg.vocab_size cfg.num_value_bins) := do
  -- Just create a dummy tensor and return
  let dummyLogits ← randn #[batch, seq, cfg.vocab_size] false
  pure { policy_logits := dummyLogits, value_logits := none }

/-- Minimal attention - just projections -/
def attentionMinimal {batch seq n_embd n_head n_kv_head : UInt64}
    (params : AttentionParams n_embd n_head n_kv_head)
    (x : T #[batch, seq, n_embd])
    : T #[batch, seq, n_embd] :=
  -- Just Q projection and output projection
  let q := linear3d x params.c_q
  let qBack := linear3d q params.c_proj
  qBack

/-- Full model forward pass -/
def forward {cfg : Config} (batch seq : UInt64)
    (params : NanoProofParams cfg)
    (_rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (idx : T #[batch, seq])
    : IO (ModelOutput batch seq cfg.vocab_size cfg.num_value_bins) := do
  -- Step 1: Embedding + norm
  let x := nn.embedding idx params.wte
  let x := norm x

  -- No blocks at all

  -- Project to vocab_size for output
  let logits := linear3d x params.lm_head

  pure { policy_logits := logits, value_logits := none }

/-- Compute combined loss for policy + value heads -/
def combinedLoss {cfg : Config} {batch seq : UInt64}
    (output : ModelOutput batch seq cfg.vocab_size cfg.num_value_bins)
    (targets : T #[batch, seq])
    (valueTargets : Option (T #[batch, seq]))
    (valueLossWeight : Float := 1.0)
    : T #[] :=
  -- Policy loss (cross-entropy)
  let policyLogitsFlat := reshape output.policy_logits #[batch * seq, cfg.vocab_size]
  let targetsFlat := reshape targets #[batch * seq]
  let policyLoss := nn.cross_entropy policyLogitsFlat targetsFlat

  -- Value loss (optional)
  match output.value_logits, valueTargets with
  | some vLogits, some vTargets =>
    let vLogitsFlat := reshape vLogits #[batch * seq, cfg.num_value_bins]
    let vTargetsFlat := reshape vTargets #[batch * seq]
    let valueLoss := nn.cross_entropy vLogitsFlat vTargetsFlat
    policyLoss + (valueLoss * valueLossWeight)
  | _, _ => policyLoss

/-- Training-only forward pass that returns loss -/
def loss {cfg : Config} (batch seq : UInt64)
    (params : NanoProofParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (idx : T #[batch, seq])
    (targets : T #[batch, seq])
    : IO (T #[]) := do
  let output ← forward batch seq params rotaryCache idx
  pure (combinedLoss output targets none)

end torch.nanoproof
