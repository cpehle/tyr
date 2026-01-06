/-
  Discrete Diffusion Transformer for Tyr

  A masked discrete diffusion model following tiny-diffusion architecture.
  Key differences from GPT:
  - Bidirectional attention (is_causal=false)
  - Time step conditioning via embeddings
  - RMSNorm without learnable parameters
  - ReLU² activation in MLP
  - Rotary positional embeddings
  - No bias in linear layers
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.NanoProof

namespace torch.diffusion

open torch
open torch.nanoproof (RotaryCache norm AttentionParams MLPParams BlockParams makeLeafParam)
open TensorStruct

/-- Diffusion model configuration -/
structure Config where
  seq_len : UInt64 := 256         -- sequence length for generation
  vocab_size : UInt64 := 128      -- ASCII vocab (0-127), where 0 = [MASK]
  mask_token_id : UInt64 := 0     -- NUL character as mask token
  n_layer : UInt64 := 4           -- transformer blocks (smaller for testing)
  n_head : UInt64 := 4            -- attention heads
  n_embd : UInt64 := 256          -- embedding dimension (smaller for testing)
  diffusion_steps : UInt64 := 64  -- timesteps for diffusion (smaller for testing)
  context_len : UInt64 := 16      -- prefix tokens never masked
  deriving Repr, Inhabited

/-- Compute head dimension from config -/
def Config.headDim (cfg : Config) : UInt64 := cfg.n_embd / cfg.n_head

/-- Tiny config for unit tests -/
def Config.tiny : Config :=
  { seq_len := 64, vocab_size := 128, n_layer := 2, n_head := 2, n_embd := 64, diffusion_steps := 16 }

/-- Small config for testing (no context for simpler sampling) -/
def Config.small : Config :=
  { seq_len := 128, vocab_size := 128, n_layer := 4, n_head := 4, n_embd := 256, diffusion_steps := 64, context_len := 0 }

/-- Full config matching tiny-diffusion paper -/
def Config.full : Config :=
  { seq_len := 256, vocab_size := 128, n_layer := 6, n_head := 6, n_embd := 384, diffusion_steps := 128 }

/-- Full diffusion model parameters -/
structure DiffusionParams (cfg : Config) where
  -- Token embedding
  token_emb : T #[cfg.vocab_size, cfg.n_embd]
  -- Time embedding for timesteps
  time_emb : T #[cfg.diffusion_steps, cfg.n_embd]
  -- Transformer blocks (reuse from NanoProof with n_head = n_kv_head)
  blocks : Array (BlockParams cfg.n_embd cfg.n_head cfg.n_head)
  -- Output head (NOT tied to token_emb)
  output_head : T #[cfg.vocab_size, cfg.n_embd]

-- Manual TensorStruct instance for DiffusionParams
instance {cfg : Config} : TensorStruct (DiffusionParams cfg) where
  map f p := {
    token_emb := f p.token_emb
    time_emb := f p.time_emb
    blocks := p.blocks.map (TensorStruct.map f)
    output_head := f p.output_head
  }
  mapM f p := do
    let token_emb ← f p.token_emb
    let time_emb ← f p.time_emb
    let blocks ← p.blocks.mapM (TensorStruct.mapM f)
    let output_head ← f p.output_head
    pure { token_emb, time_emb, blocks, output_head }
  zipWith f p1 p2 := {
    token_emb := f p1.token_emb p2.token_emb
    time_emb := f p1.time_emb p2.time_emb
    blocks := Array.zipWith (TensorStruct.zipWith f) p1.blocks p2.blocks
    output_head := f p1.output_head p2.output_head
  }
  fold f init p :=
    let acc := f p.token_emb init
    let acc := f p.time_emb acc
    let acc := p.blocks.foldl (fun acc block => TensorStruct.fold f acc block) acc
    f p.output_head acc

/-- Initialize diffusion model parameters -/
def DiffusionParams.init (cfg : Config) : IO (DiffusionParams cfg) := do
  let scale := 1.0 / Float.sqrt cfg.n_embd.toFloat

  -- Token and time embeddings
  let token_emb ← randn #[cfg.vocab_size, cfg.n_embd] false
  let time_emb ← randn #[cfg.diffusion_steps, cfg.n_embd] false

  -- Transformer blocks
  let mut blocks := #[]
  for _ in [:cfg.n_layer.toNat] do
    let block ← BlockParams.init cfg.n_embd cfg.n_head cfg.n_head
    blocks := blocks.push block

  -- Output head (zero-initialized like tiny-diffusion)
  let output_head := zeros #[cfg.vocab_size, cfg.n_embd]

  return {
    token_emb := makeLeafParam (token_emb * scale)
    time_emb := makeLeafParam time_emb  -- standard normal init for embeddings
    blocks := blocks
    output_head := makeLeafParam output_head
  }

/-- Bidirectional attention forward pass (is_causal=false) -/
def bidirectionalAttentionForward {batch seq n_embd n_head : UInt64}
    (params : AttentionParams n_embd n_head n_head)
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
  let k := reshape k #[batch, seq, n_head, head_dim]
  let v := reshape v #[batch, seq, n_head, head_dim]

  -- Apply rotary embeddings (use full cache - no slicing needed when rotaryLen == seq)
  let cos := rotaryCache.cos
  let sin := rotaryCache.sin
  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  -- QK normalization
  let q := norm q
  let k := norm k

  -- Transpose for attention: [batch, seq, n, d] -> [batch, n, seq, d]
  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  -- Bidirectional attention (is_causal=false)
  let attn := nn.scaled_dot_product_attention q k v 0.0 false

  -- Reshape back: [batch, n, seq, d] -> [batch, seq, n*d]
  let attn := nn.transpose_from_attention attn
  let attn := reshape attn #[batch, seq, n_embd]

  -- Output projection (no bias)
  linear3d attn params.c_proj

/-- MLP forward pass with ReLU² (reuse from NanoProof) -/
def mlpForward {batch seq n_embd : UInt64}
    (params : MLPParams n_embd)
    (x : T #[batch, seq, n_embd])
    : T #[batch, seq, n_embd] :=
  let h := linear3d x params.c_fc      -- Expand to 4x
  let h := nanoproof.reluSquared h     -- ReLU² activation
  linear3d h params.c_proj             -- Project back

/-- Block forward pass (pre-norm with bidirectional attention) -/
def blockForward {batch seq n_embd n_head : UInt64}
    (params : BlockParams n_embd n_head n_head)
    (x : T #[batch, seq, n_embd])
    (rotaryCache : RotaryCache rotaryLen (n_embd / n_head))
    : T #[batch, seq, n_embd] :=
  -- Attention with residual (pre-norm)
  let x := x + bidirectionalAttentionForward params.attn (norm x) rotaryCache
  -- MLP with residual (pre-norm)
  x + mlpForward params.mlp (norm x)

/-- Full diffusion model forward pass -/
def forward {cfg : Config} {batch seq : UInt64}
    (params : DiffusionParams cfg)
    (x_t : T #[batch, seq])              -- Noisy tokens (with masks)
    (t : T #[batch])                     -- Timestep indices
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    : T #[batch, seq, cfg.vocab_size] :=
  -- Token embedding
  let x := nn.embedding x_t params.token_emb  -- [batch, seq, n_embd]

  -- Time embedding: lookup and broadcast to all positions
  let t_emb := nn.embedding1d t params.time_emb  -- [batch, n_embd]
  let t_emb := nn.unsqueeze t_emb 1              -- [batch, 1, n_embd]
  let t_emb := nn.expand t_emb #[batch, seq, cfg.n_embd]  -- [batch, seq, n_embd]
  let x := x + t_emb
  -- Norm after combining embeddings (matching tiny-diffusion)
  let x := norm x

  -- Transformer blocks
  let x := params.blocks.foldl (fun x block =>
    blockForward block x rotaryCache
  ) x

  -- Final norm and output projection
  let x := norm x
  linear3d x params.output_head

/-- Compute diffusion loss (cross-entropy on all positions, masking done separately) -/
def loss {cfg : Config} {batch seq : UInt64}
    (logits : T #[batch, seq, cfg.vocab_size])
    (targets : T #[batch, seq])
    : T #[] :=
  let logitsFlat := reshape logits #[batch * seq, cfg.vocab_size]
  let targetsFlat := reshape targets #[batch * seq]
  nn.cross_entropy logitsFlat targetsFlat

/-- Helper to get mask of positions that are still masked -/
def getMaskedPositions' {batch seq : UInt64}
    (x : T #[batch, seq])
    (mask_token_id : UInt64)
    : T #[batch, seq] :=
  eq_scalar x mask_token_id.toInt64

/-- Sample using confidence-aware parallel decoding.
    At each step, decode all tokens whose confidence exceeds a threshold.
    Ensures at least one token is decoded per step if any remain masked.
-/
def sampleConfidence {cfg : Config} {batch : UInt64}
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (confidenceThreshold : Float := 0.5)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    (contextTokens : Option (T #[batch, cfg.context_len]) := none)
    : IO (T #[batch, cfg.seq_len]) := do
  -- Start with all mask tokens
  let mut x := full_int #[batch, cfg.seq_len] cfg.mask_token_id.toInt64

  -- If context tokens provided, set them
  if let some _ctx := contextTokens then
    -- Copy context tokens to first context_len positions
    -- For now, we'll skip this and just start from all masks
    -- TODO: implement proper context copying
    pure ()

  -- Track which positions are still masked
  let mut maskedPositions := getMaskedPositions' x cfg.mask_token_id

  for step in [:maxSteps] do
    -- Check if all tokens are decoded
    let anyMasked := any maskedPositions
    if !anyMasked then
      if step < 10 then
        IO.println s!"  All tokens decoded at step {step}"
      break

    -- Create timestep tensor (use step directly, matching tiny-diffusion)
    let stepClamped := min step (cfg.diffusion_steps.toNat - 1)
    let t := full_int #[batch] stepClamped.toUInt64.toInt64

    -- Predict tokens
    let logits := forward params x t rotaryCache

    -- Apply temperature and softmax
    let scaledLogits := logits / temperature
    let probs := nn.softmax scaledLogits  -- [batch, seq, vocab]

    -- Get max probability and predicted token per position
    let (confidences, predictedTokens) := max_dim probs 2  -- along vocab dim

    if step < 3 then
      -- Debug: show first few predicted tokens and their confidences
      let conf0 : T #[1] := data.slice1d (reshape confidences #[batch * cfg.seq_len]) 0 1
      let tok0 : T #[1] := data.slice1d (reshape predictedTokens #[batch * cfg.seq_len]) 0 1
      IO.println s!"  step {step}: conf[0]={nn.item conf0}, tok[0]={nn.itemInt tok0}"

    -- Select positions above threshold AND still masked
    let aboveThreshold := ge confidences (full #[batch, cfg.seq_len] confidenceThreshold)
    let selectMask := logical_and aboveThreshold maskedPositions

    -- Ensure at least one token decoded per step if any remain
    -- If no positions above threshold, decode the highest confidence masked token
    let anyAbove := any selectMask
    let selectMask := if anyAbove then
      selectMask
    else
      -- Fallback: decode all remaining masked positions
      maskedPositions

    -- Update x where selectMask is true
    let predictedLong := data.toLong predictedTokens
    x := where_ selectMask predictedLong x

    -- Update masked positions
    maskedPositions := getMaskedPositions' x cfg.mask_token_id

  return x

/-- Sample using top-K parallel decoding.
    At each step, decode exactly K tokens with highest confidence.
-/
def sampleTopK {cfg : Config} {batch : UInt64}
    (params : DiffusionParams cfg)
    (rotaryCache : RotaryCache rotaryLen cfg.headDim)
    (k : UInt64)
    (temperature : Float := 1.0)
    (maxSteps : Nat := 256)
    : IO (T #[batch, cfg.seq_len]) := do
  -- Start with all mask tokens
  let mut x := full_int #[batch, cfg.seq_len] cfg.mask_token_id.toInt64

  -- Track which positions are still masked
  let mut maskedPositions := getMaskedPositions' x cfg.mask_token_id

  for step in [:maxSteps] do
    -- Check if all tokens are decoded
    if !(any maskedPositions) then
      break

    -- Create timestep tensor
    let stepClamped := min step (cfg.diffusion_steps.toNat - 1)
    let t := full_int #[batch] stepClamped.toUInt64.toInt64

    -- Predict tokens
    let logits := forward params x t rotaryCache

    -- Apply temperature and softmax
    let scaledLogits := logits / temperature
    let probs := nn.softmax scaledLogits

    -- Get max probability and predicted token per position
    let (confidences, predictedTokens) := max_dim probs 2

    -- Mask out already-decoded positions (set confidence to -inf)
    let _maskFloat := toFloat' maskedPositions
    let negInf := full #[batch, cfg.seq_len] (-1e10)
    let maskedConfidences := where_ maskedPositions confidences negInf

    -- Get top-k positions (simplified: process first batch element)
    -- TODO: proper per-batch top-k selection
    let (_, _topkIndices) := topk maskedConfidences k 1  -- dim 1 = seq

    -- For now, just decode k tokens by selecting from top confidences
    -- This is a simplified version - full version needs scatter
    let selectMask := ge confidences (full #[batch, cfg.seq_len] 0.0)
    let selectMaskMasked := logical_and selectMask maskedPositions

    -- Update x
    let predictedLong := data.toLong predictedTokens
    x := where_ selectMaskMasked predictedLong x

    -- Update masked positions
    maskedPositions := getMaskedPositions' x cfg.mask_token_id

  return x

end torch.diffusion
