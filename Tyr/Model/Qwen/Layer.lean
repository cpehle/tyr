/-
  Tyr/Model/Qwen/Layer.lean

  Transformer layer for Qwen3.
  Pre-norm architecture with RMSNorm.
-/
import Tyr.Torch
import Tyr.Tensor
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen.MLP

namespace torch.qwen

/-- Qwen transformer layer.
    Uses pre-norm architecture with RMSNorm. -/
structure QwenLayer (hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64) where
  /-- Input layer norm (before attention) -/
  input_layernorm : RMSNorm hidden_size
  /-- Self-attention module -/
  self_attn : QwenAttention hidden_size num_heads num_kv_heads head_dim
  /-- Post-attention layer norm (before MLP) -/
  post_attention_layernorm : RMSNorm hidden_size
  /-- MLP module -/
  mlp : QwenMLP hidden_size intermediate_size
  deriving TensorStruct

namespace QwenLayer

/-- Initialize a transformer layer -/
def init (hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64)
    (eps : Float := 1e-6)
    : IO (QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size) := do
  let input_layernorm := RMSNorm.init hidden_size eps
  let self_attn ← QwenAttention.init hidden_size num_heads num_kv_heads head_dim
  let post_attention_layernorm := RMSNorm.init hidden_size eps
  let mlp ← QwenMLP.init hidden_size intermediate_size
  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Forward pass for a transformer layer.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size]

    Pre-norm: x + attn(norm1(x)) + mlp(norm2(x + attn(...))) -/
def forward {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : T #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (is_causal : Bool := true)
    : T #[batch, seq, hidden_size] :=
  -- Pre-norm attention
  let residual := x
  let x := layer.input_layernorm.forward3d x
  let x := layer.self_attn.forward x cos sin is_causal
  let x := residual + x

  -- Pre-norm MLP
  let residual := x
  let x := layer.post_attention_layernorm.forward3d x
  let x := layer.mlp.forward x
  residual + x

/-- Typed `forward` — preserves TensorMeta of the activation through
    the layer block. Body delegates to the legacy `forward`. -/
@[inline] def forwardT {tm : TensorMeta} {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : Tensor tm #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (is_causal : Bool := true)
    : Tensor tm #[batch, seq, hidden_size] :=
  Tensor.unsafeOfT tm (layer.forward (Tensor.toT x) cos sin is_causal)

/-- Incremental one-token transformer layer step with attention KV cache.
    Input/output: `[batch, 1, hidden_size]`. -/
def forwardStep {batch hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : T #[batch, 1, hidden_size])
    (cos : T #[1, head_dim / 2])
    (sin : T #[1, head_dim / 2])
    (cache : QwenAttention.KVCache batch num_kv_heads head_dim)
    : T #[batch, 1, hidden_size] × QwenAttention.KVCache batch num_kv_heads head_dim :=
  -- Pre-norm attention.
  let residual1 := x
  let x1 := layer.input_layernorm.forward3d x
  let (attnOut, cache') := layer.self_attn.forwardStep x1 cos sin cache
  let h1 := residual1 + attnOut

  -- Pre-norm MLP.
  let residual2 := h1
  let h2 := layer.post_attention_layernorm.forward3d h1
  let h3 := layer.mlp.forward h2
  (residual2 + h3, cache')

/-- Typed `forwardStep` — preserves TensorMeta of the activation
    through the cached single-token decode. Body delegates to legacy. -/
@[inline] def forwardStepT {tm : TensorMeta} {batch hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : Tensor tm #[batch, 1, hidden_size])
    (cos : T #[1, head_dim / 2])
    (sin : T #[1, head_dim / 2])
    (cache : QwenAttention.KVCache batch num_kv_heads head_dim)
    : Tensor tm #[batch, 1, hidden_size] × QwenAttention.KVCache batch num_kv_heads head_dim :=
  let (out, cache') := layer.forwardStep (Tensor.toT x) cos sin cache
  (Tensor.unsafeOfT tm out, cache')

def forwardMasked {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : T #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (attn_mask : T #[batch, seq])
    (is_causal : Bool := true)
    : T #[batch, seq, hidden_size] :=
  -- Pre-norm attention
  let residual := x
  let x := layer.input_layernorm.forward3d x
  let x := layer.self_attn.forwardMasked x cos sin attn_mask is_causal
  let x := residual + x

  -- Pre-norm MLP
  let residual := x
  let x := layer.post_attention_layernorm.forward3d x
  let x := layer.mlp.forward x
  residual + x

/-- Typed `forwardMasked` — preserves TensorMeta of the activation. -/
@[inline] def forwardMaskedT {tm : TensorMeta} {batch seq hidden_size num_heads num_kv_heads head_dim intermediate_size : UInt64}
    (layer : QwenLayer hidden_size num_heads num_kv_heads head_dim intermediate_size)
    (x : Tensor tm #[batch, seq, hidden_size])
    (cos : T #[seq, head_dim / 2])
    (sin : T #[seq, head_dim / 2])
    (attn_mask : T #[batch, seq])
    (is_causal : Bool := true)
    : Tensor tm #[batch, seq, hidden_size] :=
  Tensor.unsafeOfT tm (layer.forwardMasked (Tensor.toT x) cos sin attn_mask is_causal)

end QwenLayer

end torch.qwen
