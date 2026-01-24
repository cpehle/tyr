/-
  Tyr/Model/Qwen/Attention.lean

  Grouped Query Attention (GQA) for Qwen3.
  Uses fewer key-value heads than query heads for efficiency.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Config

namespace torch.qwen

/-- Qwen attention with Grouped Query Attention (GQA).
    Query heads: num_attention_heads
    Key/Value heads: num_key_value_heads (typically 4 for Qwen3-4B)
    Each KV head is shared across multiple query heads. -/
structure QwenAttention (hidden_size num_heads num_kv_heads head_dim : UInt64) where
  /-- Query projection: [hidden_size, hidden_size] -/
  q_proj : T #[num_heads * head_dim, hidden_size]
  /-- Key projection: [kv_dim, hidden_size] -/
  k_proj : T #[num_kv_heads * head_dim, hidden_size]
  /-- Value projection: [kv_dim, hidden_size] -/
  v_proj : T #[num_kv_heads * head_dim, hidden_size]
  /-- Output projection: [hidden_size, hidden_size] -/
  o_proj : T #[hidden_size, num_heads * head_dim]
  /-- Optional Q norm (per-head normalization for Flux Klein text encoder) -/
  q_norm : Option (T #[head_dim]) := none
  /-- Optional K norm (per-head normalization for Flux Klein text encoder) -/
  k_norm : Option (T #[head_dim]) := none
  deriving TensorStruct

namespace QwenAttention

/-- Initialize attention layers with random weights -/
def init (hidden_size num_heads num_kv_heads head_dim : UInt64) : IO (QwenAttention hidden_size num_heads num_kv_heads head_dim) := do
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let q ← torch.randn #[num_heads * head_dim, hidden_size]
  let k ← torch.randn #[num_kv_heads * head_dim, hidden_size]
  let v ← torch.randn #[num_kv_heads * head_dim, hidden_size]
  let o ← torch.randn #[hidden_size, num_heads * head_dim]

  pure {
    q_proj := autograd.set_requires_grad (mul_scalar q std) true
    k_proj := autograd.set_requires_grad (mul_scalar k std) true
    v_proj := autograd.set_requires_grad (mul_scalar v std) true
    o_proj := autograd.set_requires_grad (mul_scalar o std) true
  }

/-- Apply RMS normalization along the last dimension (per-head norm).
    Input: [batch, seq, n_heads, head_dim], norm: [head_dim]
    Broadcasts norm across all heads. -/
private def applyHeadNorm {batch seq n_heads head_dim : UInt64}
    (x : T #[batch, seq, n_heads, head_dim])
    (norm : T #[head_dim])
    (eps : Float := 1e-6)
    : T #[batch, seq, n_heads, head_dim] :=
  -- RMSNorm: x / sqrt(mean(x^2) + eps) * scale
  -- We need to compute this per-head (over head_dim dimension)
  let x2 := x * x  -- element-wise square
  -- Flatten to [batch * seq * n_heads, head_dim] for mean computation
  let x2_flat := reshape x2 #[batch * seq * n_heads, head_dim]
  let x_flat := reshape x #[batch * seq * n_heads, head_dim]
  -- Mean over last dim (head_dim)
  let variance := nn.meanDim x2_flat 1 false  -- [batch*seq*n_heads]
  -- Add epsilon and compute rsqrt
  let variance_plus_eps := variance + eps
  let rsqrt_flat := nn.rsqrt variance_plus_eps  -- [batch*seq*n_heads]
  -- Expand rsqrt to match x_flat: [batch*seq*n_heads, 1]
  let rsqrt_expanded := nn.unsqueeze rsqrt_flat 1  -- [batch*seq*n_heads, 1]
  let rsqrt_broadcast := nn.expand rsqrt_expanded #[batch * seq * n_heads, head_dim]
  -- Normalize
  let normalized_flat := x_flat * rsqrt_broadcast
  let normalized := reshape normalized_flat #[batch, seq, n_heads, head_dim]
  -- Apply scale (broadcast norm across batch, seq, n_heads)
  let norm_4d := reshape norm #[1, 1, 1, head_dim]
  let norm_expanded := nn.expand norm_4d #[batch, seq, n_heads, head_dim]
  normalized * norm_expanded

/-- Forward pass for attention.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size] -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim : UInt64}
    (attn : QwenAttention hidden_size num_heads num_kv_heads head_dim)
    (x : T #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (is_causal : Bool := true)
    : T #[batch, seq, hidden_size] :=
  -- Project to Q, K, V
  let q := linear3d x attn.q_proj  -- [batch, seq, num_heads * head_dim]
  let k := linear3d x attn.k_proj  -- [batch, seq, num_kv_heads * head_dim]
  let v := linear3d x attn.v_proj  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := reshape q #[batch, seq, num_heads, head_dim]
  let k := reshape k #[batch, seq, num_kv_heads, head_dim]
  let v := reshape v #[batch, seq, num_kv_heads, head_dim]

  -- Apply Q/K norms if present (Flux Klein text encoder has these)
  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to Q and K
  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  -- Transpose to [batch, num_heads, seq, head_dim]
  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  -- Scaled dot-product attention with GQA
  let attn_out := nn.scaledDotProductAttentionGQA q k v 0.0 is_causal true

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := nn.transpose_from_attention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := reshape attn_out #[batch, seq, num_heads * head_dim]

  -- Output projection
  linear3d attn_out attn.o_proj

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim : UInt64}
    (attn : QwenAttention hidden_size num_heads num_kv_heads head_dim)
    (x : T #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (attn_mask : T #[batch, seq])
    (is_causal : Bool := true)
    : T #[batch, seq, hidden_size] :=
  -- Project to Q, K, V
  let q := linear3d x attn.q_proj  -- [batch, seq, num_heads * head_dim]
  let k := linear3d x attn.k_proj  -- [batch, seq, num_kv_heads * head_dim]
  let v := linear3d x attn.v_proj  -- [batch, seq, num_kv_heads * head_dim]

  -- Reshape to [batch, seq, num_heads, head_dim]
  let q := reshape q #[batch, seq, num_heads, head_dim]
  let k := reshape k #[batch, seq, num_kv_heads, head_dim]
  let v := reshape v #[batch, seq, num_kv_heads, head_dim]

  -- Apply Q/K norms if present (Flux Klein text encoder has these)
  let q := match attn.q_norm with
    | some qn => applyHeadNorm q qn
    | none => q
  let k := match attn.k_norm with
    | some kn => applyHeadNorm k kn
    | none => k

  -- Apply RoPE to Q and K
  let q := rotary.applyRotaryEmb q cos sin
  let k := rotary.applyRotaryEmb k cos sin

  -- Transpose to [batch, num_heads, seq, head_dim]
  let q := nn.transpose_for_attention q
  let k := nn.transpose_for_attention k
  let v := nn.transpose_for_attention v

  -- Scaled dot-product attention with GQA and padding mask
  let attn_out := nn.scaledDotProductAttentionGQAMask q k v attn_mask 0.0 is_causal true

  -- Transpose back to [batch, seq, num_heads, head_dim]
  let attn_out := nn.transpose_from_attention attn_out

  -- Reshape to [batch, seq, hidden_size]
  let attn_out := reshape attn_out #[batch, seq, num_heads * head_dim]

  -- Output projection
  linear3d attn_out attn.o_proj

end QwenAttention

end torch.qwen
