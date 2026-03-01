/-
  Tyr/Model/Qwen35/Model.lean

  Standalone Qwen3.5 text causal-LM for Tyr.
  Features:
  - Hybrid layer schedule (`linear_attention` + `full_attention`)
  - Dense FFN and MoE FFN variants
  - Full-attention Q-gate + per-head Q/K RMSNorm
  - Linear GatedDeltaNet with depthwise causal conv and recurrent state cache
  - Cached and uncached generation
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen35.Config

namespace torch.qwen35

open torch

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

private def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

private def allInSet (xs : Array UInt64) (allow : Array UInt64) : Bool :=
  xs.all (fun x => allow.contains x)

private def softplus {s : Shape} (x : T s) : T s :=
  nn.log (add_scalar (nn.exp x) 1.0)

private def l2NormLast4d {b s h d : UInt64}
    (x : T #[b, s, h, d])
    (eps : Float := 1e-6)
    : T #[b, s, h, d] :=
  let flat : T #[b * s * h, d] := reshape x #[b * s * h, d]
  let n2 : T #[b * s * h, 1] := nn.sumDim (flat * flat) 1 true
  let inv : T #[b * s * h, 1] := nn.rsqrt (n2 + eps)
  let inv : T #[b * s * h, d] := nn.expand inv #[b * s * h, d]
  reshape (flat * inv) #[b, s, h, d]

private def applyMaskToPaddingStates {batch seq hidden : UInt64}
    (x : T #[batch, seq, hidden])
    (attentionMask : Option (T #[batch, seq]))
    : T #[batch, seq, hidden] :=
  match attentionMask with
  | none => x
  | some m =>
    let mf : T #[batch, seq] := toFloat' m
    let me : T #[batch, seq, hidden] := nn.expand (reshape mf #[batch, seq, 1]) #[batch, seq, hidden]
    x * me

/-- Qwen3.5 RMSNorm with `(1 + weight)` scaling. -/
structure Qwen35RMSNorm (dim : UInt64) where
  weight : T #[dim]
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35RMSNorm

def initZeroCentered (dim : UInt64) (eps : Float := 1e-6) : Qwen35RMSNorm dim :=
  { weight := autograd.set_requires_grad (torch.zeros #[dim]) true, eps }

def initOneCentered (dim : UInt64) (eps : Float := 1e-6) : Qwen35RMSNorm dim :=
  { weight := autograd.set_requires_grad (torch.ones #[dim]) true, eps }

def forward2d {n dim : UInt64}
    (m : Qwen35RMSNorm dim)
    (x : T #[n, dim])
    : T #[n, dim] :=
  let xf : T #[n, dim] := toFloat' x
  let var : T #[n, 1] := nn.meanDim (xf * xf) 1 true
  let inv : T #[n, 1] := nn.rsqrt (var + m.eps)
  let inv : T #[n, dim] := nn.expand inv #[n, dim]
  let normed : T #[n, dim] := xf * inv
  let scale1 : T #[dim] := add_scalar (toFloat' m.weight) 1.0
  let scale : T #[n, dim] := nn.expand (reshape scale1 #[1, dim]) #[n, dim]
  normed * scale

def forward3d {batch seq dim : UInt64}
    (m : Qwen35RMSNorm dim)
    (x : T #[batch, seq, dim])
    : T #[batch, seq, dim] :=
  let flat : T #[batch * seq, dim] := reshape x #[batch * seq, dim]
  reshape (forward2d m flat) #[batch, seq, dim]

end Qwen35RMSNorm

/-- Gated RMSNorm used at linear-attention output. -/
structure Qwen35RMSNormGated (dim : UInt64) where
  weight : T #[dim]
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35RMSNormGated

def init (dim : UInt64) (eps : Float := 1e-6) : Qwen35RMSNormGated dim :=
  { weight := autograd.set_requires_grad (torch.ones #[dim]) true, eps }

def forward2d {n dim : UInt64}
    (m : Qwen35RMSNormGated dim)
    (x : T #[n, dim])
    (gate : T #[n, dim])
    : T #[n, dim] :=
  let xf : T #[n, dim] := toFloat' x
  let var : T #[n, 1] := nn.meanDim (xf * xf) 1 true
  let inv : T #[n, 1] := nn.rsqrt (var + m.eps)
  let inv : T #[n, dim] := nn.expand inv #[n, dim]
  let normed : T #[n, dim] := xf * inv
  let scale : T #[n, dim] := nn.expand (reshape (toFloat' m.weight) #[1, dim]) #[n, dim]
  let g : T #[n, dim] := nn.silu (toFloat' gate)
  normed * scale * g

end Qwen35RMSNormGated

/-- SwiGLU MLP block (dense FFN). -/
structure Qwen35MLP (hidden_size intermediate_size : UInt64) where
  gate_proj : T #[intermediate_size, hidden_size]
  up_proj : T #[intermediate_size, hidden_size]
  down_proj : T #[hidden_size, intermediate_size]
  deriving TensorStruct

namespace Qwen35MLP

def init (hidden_size intermediate_size : UInt64) : IO (Qwen35MLP hidden_size intermediate_size) := do
  let gate ← initWeight #[intermediate_size, hidden_size] hidden_size
  let up ← initWeight #[intermediate_size, hidden_size] hidden_size
  let down ← initWeight #[hidden_size, intermediate_size] intermediate_size
  pure { gate_proj := gate, up_proj := up, down_proj := down }

def forward2d {tokens hidden inter : UInt64}
    (m : Qwen35MLP hidden inter)
    (x : T #[tokens, hidden])
    : T #[tokens, hidden] :=
  let gateDyn : T #[] := torch.einsum2 "oh,th->to" m.gate_proj x
  let upDyn : T #[] := torch.einsum2 "oh,th->to" m.up_proj x
  let gate : T #[tokens, inter] := reshape gateDyn #[tokens, inter]
  let up : T #[tokens, inter] := reshape upDyn #[tokens, inter]
  let hid : T #[tokens, inter] := nn.silu gate * up
  let outDyn : T #[] := torch.einsum2 "ho,to->th" m.down_proj hid
  reshape outDyn #[tokens, hidden]

def forward {batch seq hidden inter : UInt64}
    (m : Qwen35MLP hidden inter)
    (x : T #[batch, seq, hidden])
    : T #[batch, seq, hidden] :=
  linear3d (nn.silu (linear3d x m.gate_proj) * linear3d x m.up_proj) m.down_proj

end Qwen35MLP

/-- Top-k router for Qwen3.5 MoE. -/
structure Qwen35TopKRouter (cfg : Config) where
  weight : T #[cfg.num_experts, cfg.hidden_size]
  deriving TensorStruct

namespace Qwen35TopKRouter

def init (cfg : Config) : IO (Qwen35TopKRouter cfg) := do
  let w ← initWeight #[cfg.num_experts, cfg.hidden_size] cfg.hidden_size
  pure { weight := w }

def forward {tokens : UInt64}
    (cfg : Config)
    (m : Qwen35TopKRouter cfg)
    (hidden : T #[tokens, cfg.hidden_size])
    : T #[tokens, cfg.num_experts] × T #[tokens, cfg.num_experts_per_tok] × T #[tokens, cfg.num_experts_per_tok] :=
  let logitsDyn : T #[] := torch.einsum2 "eh,th->te" m.weight hidden
  let logits : T #[tokens, cfg.num_experts] := reshape logitsDyn #[tokens, cfg.num_experts]
  let probs : T #[tokens, cfg.num_experts] := nn.softmax logits (-1)
  let (topValsRaw, topIdx) := torch.topk_2d probs cfg.num_experts_per_tok 1
  let denom : T #[tokens, 1] := nn.sumDim topValsRaw 1 true
  let denom : T #[tokens, cfg.num_experts_per_tok] := nn.expand denom #[tokens, cfg.num_experts_per_tok]
  let topVals : T #[tokens, cfg.num_experts_per_tok] := nn.div topValsRaw denom
  (probs, topVals, topIdx)

end Qwen35TopKRouter

/-- Expert weight bank for Qwen3.5 MoE FFN. -/
structure Qwen35MoeExperts (cfg : Config) where
  gate_up_proj : T #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]
  down_proj : T #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]
  deriving TensorStruct

namespace Qwen35MoeExperts

def init (cfg : Config) : IO (Qwen35MoeExperts cfg) := do
  let gu ← initWeight #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size] cfg.hidden_size
  let down ← initWeight #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] cfg.moe_intermediate_size
  pure { gate_up_proj := gu, down_proj := down }

def forward2d {tokens : UInt64}
    (cfg : Config)
    (m : Qwen35MoeExperts cfg)
    (hidden : T #[tokens, cfg.hidden_size])
    (topVals : T #[tokens, cfg.num_experts_per_tok])
    (topIdx : T #[tokens, cfg.num_experts_per_tok])
    : T #[tokens, cfg.hidden_size] :=
  Id.run do
    let mut acc : T #[tokens, cfg.hidden_size] := torch.zeros #[tokens, cfg.hidden_size]

    for slot in [:cfg.num_experts_per_tok.toNat] do
      let idx2d : T #[tokens, 1] := data.slice topIdx 1 slot.toUInt64 1
      let srcOnes : T #[tokens, 1] := torch.ones #[tokens, 1]
      let base : T #[tokens, cfg.num_experts] := torch.zeros #[tokens, cfg.num_experts]
      let oneHot : T #[tokens, cfg.num_experts] := torch.scatter_2d base 1 idx2d srcOnes

      let tokGateUpDyn : T #[] := torch.einsum2 "te,eih->tih" oneHot m.gate_up_proj
      let tokGateUp : T #[tokens, 2 * cfg.moe_intermediate_size, cfg.hidden_size] :=
        reshape tokGateUpDyn #[tokens, 2 * cfg.moe_intermediate_size, cfg.hidden_size]

      let guDyn : T #[] := torch.einsum2 "tih,th->ti" tokGateUp hidden
      let gu : T #[tokens, 2 * cfg.moe_intermediate_size] := reshape guDyn #[tokens, 2 * cfg.moe_intermediate_size]

      let gate : T #[tokens, cfg.moe_intermediate_size] := data.slice gu 1 0 cfg.moe_intermediate_size
      let up : T #[tokens, cfg.moe_intermediate_size] :=
        data.slice gu 1 cfg.moe_intermediate_size cfg.moe_intermediate_size
      let inter : T #[tokens, cfg.moe_intermediate_size] := nn.silu gate * up

      let tokDownDyn : T #[] := torch.einsum2 "te,ehi->thi" oneHot m.down_proj
      let tokDown : T #[tokens, cfg.hidden_size, cfg.moe_intermediate_size] :=
        reshape tokDownDyn #[tokens, cfg.hidden_size, cfg.moe_intermediate_size]

      let outDyn : T #[] := torch.einsum2 "thi,ti->th" tokDown inter
      let out : T #[tokens, cfg.hidden_size] := reshape outDyn #[tokens, cfg.hidden_size]

      let w2d : T #[tokens, 1] := data.slice topVals 1 slot.toUInt64 1
      let w : T #[tokens, cfg.hidden_size] := nn.expand w2d #[tokens, cfg.hidden_size]
      acc := acc + out * w

    return acc

end Qwen35MoeExperts

/-- Sparse MoE block (routed experts + shared expert). -/
structure Qwen35SparseMoeBlock (cfg : Config) where
  router : Qwen35TopKRouter cfg
  experts : Qwen35MoeExperts cfg
  shared_expert : Qwen35MLP cfg.hidden_size cfg.shared_expert_intermediate_size
  shared_expert_gate : T #[1, cfg.hidden_size]
  deriving TensorStruct

namespace Qwen35SparseMoeBlock

def init (cfg : Config) : IO (Qwen35SparseMoeBlock cfg) := do
  let router ← Qwen35TopKRouter.init cfg
  let experts ← Qwen35MoeExperts.init cfg
  let sharedExpert ← Qwen35MLP.init cfg.hidden_size cfg.shared_expert_intermediate_size
  let sharedGate ← initWeight #[1, cfg.hidden_size] cfg.hidden_size
  pure {
    router := router
    experts := experts
    shared_expert := sharedExpert
    shared_expert_gate := sharedGate
  }

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35SparseMoeBlock cfg)
    (hidden : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.hidden_size] :=
  let tokens := batch * seq
  let hidden2d : T #[tokens, cfg.hidden_size] := reshape hidden #[tokens, cfg.hidden_size]

  let (_routerProbs, topVals, topIdx) := m.router.forward cfg hidden2d
  let expertOut : T #[tokens, cfg.hidden_size] :=
    m.experts.forward2d cfg hidden2d topVals topIdx

  let sharedOut : T #[tokens, cfg.hidden_size] :=
    m.shared_expert.forward2d hidden2d

  let gateDyn : T #[] := torch.einsum2 "oh,th->to" m.shared_expert_gate hidden2d
  let gate1 : T #[tokens, 1] := reshape gateDyn #[tokens, 1]
  let gate : T #[tokens, cfg.hidden_size] :=
    nn.expand (nn.sigmoid gate1) #[tokens, cfg.hidden_size]

  reshape (expertOut + sharedOut * gate) #[batch, seq, cfg.hidden_size]

end Qwen35SparseMoeBlock

/-- Full-attention token mixer for Qwen3.5 (Q-gate + Q/K RMSNorm + partial RoPE). -/
structure Qwen35Attention (cfg : Config) where
  q_proj : T #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size]
  k_proj : T #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]
  v_proj : T #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]
  o_proj : T #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim]
  q_norm : Qwen35RMSNorm cfg.head_dim
  k_norm : Qwen35RMSNorm cfg.head_dim
  deriving TensorStruct

namespace Qwen35Attention

abbrev KVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch cfg.num_key_value_heads cfg.head_dim

def init (cfg : Config) : IO (Qwen35Attention cfg) := do
  let q ← initWeight #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size] cfg.hidden_size
  let k ← initWeight #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let v ← initWeight #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let o ← initWeight #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] (cfg.num_attention_heads * cfg.head_dim)
  pure {
    q_proj := q
    k_proj := k
    v_proj := v
    o_proj := o
    q_norm := Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
    k_norm := Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
  }

private def applyHeadNorm {batch seq n_heads head_dim : UInt64}
    (norm : Qwen35RMSNorm head_dim)
    (x : T #[batch, seq, n_heads, head_dim])
    : T #[batch, seq, n_heads, head_dim] :=
  let flat : T #[batch * seq * n_heads, head_dim] := reshape x #[batch * seq * n_heads, head_dim]
  reshape (norm.forward2d flat) #[batch, seq, n_heads, head_dim]

private def applyRotaryPartial {batch seq n_head head_dim rotary_dim : UInt64}
    (x : T #[batch, seq, n_head, head_dim])
    (cos : T #[seq, rotary_dim / 2])
    (sin : T #[seq, rotary_dim / 2])
    : T #[batch, seq, n_head, head_dim] :=
  let xRot : T #[batch, seq, n_head, rotary_dim] := data.slice x 3 0 rotary_dim
  let xRot : T #[batch, seq, n_head, rotary_dim] := rotary.applyRotaryEmb xRot cos sin
  let xPassLen : UInt64 := head_dim - rotary_dim
  let xPass : T #[batch, seq, n_head, xPassLen] := data.slice x 3 rotary_dim xPassLen
  nn.cat xRot xPass 3

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, Config.rotaryHalfDim cfg])
    (sin : T #[seq, Config.rotaryHalfDim cfg])
    (is_causal : Bool := true)
    : T #[batch, seq, cfg.hidden_size] :=
  let qg : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2] := linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let qFlat : T #[batch, seq, qLen] := data.slice qg 2 0 qLen
  let gateFlat : T #[batch, seq, qLen] := data.slice qg 2 qLen qLen

  let kFlat : T #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] := linear3d x m.k_proj
  let vFlat : T #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] := linear3d x m.v_proj

  let q : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] :=
    reshape qFlat #[batch, seq, cfg.num_attention_heads, cfg.head_dim]
  let k : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] := nn.transpose_for_attention q
  let kh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention k
  let vh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention v

  let attn : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    nn.scaledDotProductAttentionGQA qh kh vh 0.0 is_causal true

  let attn : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] := nn.transpose_from_attention attn
  let attnFlat : T #[batch, seq, qLen] := reshape attn #[batch, seq, qLen]
  let gated : T #[batch, seq, qLen] := attnFlat * nn.sigmoid gateFlat
  linear3d gated m.o_proj

def forwardMasked {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, Config.rotaryHalfDim cfg])
    (sin : T #[seq, Config.rotaryHalfDim cfg])
    (attn_mask : T #[batch, seq])
    (is_causal : Bool := true)
    : T #[batch, seq, cfg.hidden_size] :=
  let qg : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2] := linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let qFlat : T #[batch, seq, qLen] := data.slice qg 2 0 qLen
  let gateFlat : T #[batch, seq, qLen] := data.slice qg 2 qLen qLen

  let kFlat : T #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] := linear3d x m.k_proj
  let vFlat : T #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] := linear3d x m.v_proj

  let q : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] :=
    reshape qFlat #[batch, seq, cfg.num_attention_heads, cfg.head_dim]
  let k : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] := nn.transpose_for_attention q
  let kh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention k
  let vh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention v

  let attn : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    nn.scaledDotProductAttentionGQAMask qh kh vh attn_mask 0.0 is_causal true

  let attn : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] := nn.transpose_from_attention attn
  let attnFlat : T #[batch, seq, qLen] := reshape attn #[batch, seq, qLen]
  let gated : T #[batch, seq, qLen] := attnFlat * nn.sigmoid gateFlat
  linear3d gated m.o_proj

def forwardStep {batch : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : T #[batch, 1, cfg.hidden_size])
    (cos : T #[1, Config.rotaryHalfDim cfg])
    (sin : T #[1, Config.rotaryHalfDim cfg])
    (cache : KVCache cfg batch)
    : T #[batch, 1, cfg.hidden_size] × KVCache cfg batch :=
  let qg : T #[batch, 1, cfg.num_attention_heads * cfg.head_dim * 2] := linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let qFlat : T #[batch, 1, qLen] := data.slice qg 2 0 qLen
  let gateFlat : T #[batch, 1, qLen] := data.slice qg 2 qLen qLen

  let kFlat : T #[batch, 1, cfg.num_key_value_heads * cfg.head_dim] := linear3d x m.k_proj
  let vFlat : T #[batch, 1, cfg.num_key_value_heads * cfg.head_dim] := linear3d x m.v_proj

  let q : T #[batch, 1, cfg.num_attention_heads, cfg.head_dim] :=
    reshape qFlat #[batch, 1, cfg.num_attention_heads, cfg.head_dim]
  let k : T #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape kFlat #[batch, 1, cfg.num_key_value_heads, cfg.head_dim]
  let v : T #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] :=
    reshape vFlat #[batch, 1, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : T #[batch, cfg.num_attention_heads, 1, cfg.head_dim] := nn.transpose_for_attention q
  let kNew : T #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] := nn.transpose_for_attention k
  let vNew : T #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] := nn.transpose_for_attention v

  let kStore : T #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] :=
    reshape cache.kStoreDyn #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim]
  let vStore : T #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] :=
    reshape cache.vStoreDyn #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim]

  if cache.seq < cache.maxLen then
    let kStore' : T #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] :=
      data.sliceScatter kStore 2 cache.seq kNew
    let vStore' : T #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] :=
      data.sliceScatter vStore 2 cache.seq vNew
    let kvLen : UInt64 := cache.seq + 1
    let kAll : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] := data.slice kStore' 2 0 kvLen
    let vAll : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] := data.slice vStore' 2 0 kvLen
    let attn : T #[batch, cfg.num_attention_heads, 1, cfg.head_dim] :=
      nn.scaledDotProductAttentionGQAQKV qh kAll vAll 0.0 false true
    let attn : T #[batch, 1, cfg.num_attention_heads, cfg.head_dim] := nn.transpose_from_attention attn
    let attnFlat : T #[batch, 1, qLen] := reshape attn #[batch, 1, qLen]
    let gated : T #[batch, 1, qLen] := attnFlat * nn.sigmoid gateFlat
    let out : T #[batch, 1, cfg.hidden_size] := linear3d gated m.o_proj
    let cache' : KVCache cfg batch := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : UInt64 := if cache.maxLen == 0 then 0 else cache.maxLen - 1
    let kStore' : T #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] :=
      data.sliceScatter kStore 2 writePos kNew
    let vStore' : T #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] :=
      data.sliceScatter vStore 2 writePos vNew
    let kvLen : UInt64 := cache.maxLen
    let kAll : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] := data.slice kStore' 2 0 kvLen
    let vAll : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] := data.slice vStore' 2 0 kvLen
    let attn : T #[batch, cfg.num_attention_heads, 1, cfg.head_dim] :=
      nn.scaledDotProductAttentionGQAQKV qh kAll vAll 0.0 false true
    let attn : T #[batch, 1, cfg.num_attention_heads, cfg.head_dim] := nn.transpose_from_attention attn
    let attnFlat : T #[batch, 1, qLen] := reshape attn #[batch, 1, qLen]
    let gated : T #[batch, 1, qLen] := attnFlat * nn.sigmoid gateFlat
    let out : T #[batch, 1, cfg.hidden_size] := linear3d gated m.o_proj
    let cache' : KVCache cfg batch := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

end Qwen35Attention

/-- Linear token mixer for Qwen3.5 (`GatedDeltaNet`). -/
structure Qwen35GatedDeltaNet (cfg : Config) where
  in_proj_qkv : T #[Config.linearConvDim cfg, cfg.hidden_size]
  in_proj_z : T #[Config.linearValueDim cfg, cfg.hidden_size]
  in_proj_b : T #[cfg.linear_num_value_heads, cfg.hidden_size]
  in_proj_a : T #[cfg.linear_num_value_heads, cfg.hidden_size]

  conv1d_weight : T #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim]
  conv1d_bias : T #[Config.linearConvDim cfg]

  dt_bias : T #[cfg.linear_num_value_heads]
  a_log : T #[cfg.linear_num_value_heads]

  norm : Qwen35RMSNormGated cfg.linear_value_head_dim
  out_proj : T #[cfg.hidden_size, Config.linearValueDim cfg]
  deriving TensorStruct

namespace Qwen35GatedDeltaNet

def init (cfg : Config) : IO (Qwen35GatedDeltaNet cfg) := do
  let inQKV ← initWeight #[Config.linearConvDim cfg, cfg.hidden_size] cfg.hidden_size
  let inZ ← initWeight #[Config.linearValueDim cfg, cfg.hidden_size] cfg.hidden_size
  let inB ← initWeight #[cfg.linear_num_value_heads, cfg.hidden_size] cfg.hidden_size
  let inA ← initWeight #[cfg.linear_num_value_heads, cfg.hidden_size] cfg.hidden_size

  let convW ← initWeight #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] cfg.linear_conv_kernel_dim
  let convB := initBias #[Config.linearConvDim cfg]

  let dtBias := autograd.set_requires_grad (torch.ones #[cfg.linear_num_value_heads]) true

  let aRand ← torch.rand #[cfg.linear_num_value_heads]
  let aScaled : T #[cfg.linear_num_value_heads] := add_scalar (mul_scalar aRand 16.0) 1e-6
  let aLog := autograd.set_requires_grad (nn.log aScaled) true

  let outProj ← initWeight #[cfg.hidden_size, Config.linearValueDim cfg] (Config.linearValueDim cfg)
  pure {
    in_proj_qkv := inQKV
    in_proj_z := inZ
    in_proj_b := inB
    in_proj_a := inA
    conv1d_weight := convW
    conv1d_bias := convB
    dt_bias := dtBias
    a_log := aLog
    norm := Qwen35RMSNormGated.init cfg.linear_value_head_dim cfg.rms_norm_eps
    out_proj := outProj
  }

private def repeatHeads {batch seq n_head head_dim rep : UInt64}
    (x : T #[batch, seq, n_head, head_dim])
    : T #[batch, seq, n_head * rep, head_dim] :=
  if rep == 1 then
    reshape x #[batch, seq, n_head * rep, head_dim]
  else
    let x5 : T #[batch, seq, n_head, 1, head_dim] := nn.unsqueeze x 3
    let x5 : T #[batch, seq, n_head, rep, head_dim] := nn.expand x5 #[batch, seq, n_head, rep, head_dim]
    reshape x5 #[batch, seq, n_head * rep, head_dim]

private def recurrentGatedDelta {batch seq n_head kdim vdim : UInt64}
    (query : T #[batch, seq, n_head, kdim])
    (key : T #[batch, seq, n_head, kdim])
    (value : T #[batch, seq, n_head, vdim])
    (g : T #[batch, seq, n_head])
    (beta : T #[batch, seq, n_head])
    (initialState : Option (T #[batch, n_head, kdim, vdim]) := none)
    : T #[batch, seq, n_head, vdim] × T #[batch, n_head, kdim, vdim] :=
  Id.run do
    let q : T #[batch, seq, n_head, kdim] := l2NormLast4d query
    let k : T #[batch, seq, n_head, kdim] := l2NormLast4d key

    let scale := 1.0 / Float.sqrt kdim.toFloat
    let q : T #[batch, seq, n_head, kdim] := q * scale

    let qh : T #[batch, n_head, seq, kdim] := nn.transpose_for_attention q
    let kh : T #[batch, n_head, seq, kdim] := nn.transpose_for_attention k
    let vh : T #[batch, n_head, seq, vdim] := nn.transpose_for_attention value
    let gh : T #[batch, n_head, seq] := nn.transpose3d_12 g
    let bh : T #[batch, n_head, seq] := nn.transpose3d_12 beta

    let mut state : T #[batch, n_head, kdim, vdim] :=
      match initialState with
      | some s => s
      | none => torch.zeros #[batch, n_head, kdim, vdim]

    let mut out : T #[batch, n_head, seq, vdim] := torch.zeros #[batch, n_head, seq, vdim]

    for t in [:seq.toNat] do
      let q_t4 : T #[batch, n_head, 1, kdim] := data.slice qh 2 t.toUInt64 1
      let k_t4 : T #[batch, n_head, 1, kdim] := data.slice kh 2 t.toUInt64 1
      let v_t4 : T #[batch, n_head, 1, vdim] := data.slice vh 2 t.toUInt64 1
      let g_t3 : T #[batch, n_head, 1] := data.slice gh 2 t.toUInt64 1
      let b_t3 : T #[batch, n_head, 1] := data.slice bh 2 t.toUInt64 1

      let q_t : T #[batch, n_head, kdim] := reshape q_t4 #[batch, n_head, kdim]
      let k_t : T #[batch, n_head, kdim] := reshape k_t4 #[batch, n_head, kdim]
      let v_t : T #[batch, n_head, vdim] := reshape v_t4 #[batch, n_head, vdim]
      let g_t : T #[batch, n_head] := reshape g_t3 #[batch, n_head]
      let b_t : T #[batch, n_head] := reshape b_t3 #[batch, n_head]

      let gExp : T #[batch, n_head, kdim, vdim] :=
        nn.expand (reshape (nn.exp g_t) #[batch, n_head, 1, 1]) #[batch, n_head, kdim, vdim]
      state := state * gExp

      let kExp : T #[batch, n_head, kdim, vdim] :=
        nn.expand (reshape k_t #[batch, n_head, kdim, 1]) #[batch, n_head, kdim, vdim]
      let kvMem : T #[batch, n_head, vdim] := nn.sumDim (state * kExp) 2 false

      let bExp : T #[batch, n_head, vdim] :=
        nn.expand (reshape b_t #[batch, n_head, 1]) #[batch, n_head, vdim]
      let delta : T #[batch, n_head, vdim] := (v_t - kvMem) * bExp
      let deltaExp : T #[batch, n_head, kdim, vdim] :=
        nn.expand (reshape delta #[batch, n_head, 1, vdim]) #[batch, n_head, kdim, vdim]
      state := state + (kExp * deltaExp)

      let qExp : T #[batch, n_head, kdim, vdim] :=
        nn.expand (reshape q_t #[batch, n_head, kdim, 1]) #[batch, n_head, kdim, vdim]
      let out_t : T #[batch, n_head, vdim] := nn.sumDim (state * qExp) 2 false
      let out_t4 : T #[batch, n_head, 1, vdim] := reshape out_t #[batch, n_head, 1, vdim]
      out := data.sliceScatter out 2 t.toUInt64 out_t4

    (nn.transpose_from_attention out, state)

/-- Run linear token mixer. If `cacheConv`/`cacheRec` are provided and
    `usePrecomputedState=true`, performs incremental one-step update. -/
def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35GatedDeltaNet cfg)
    (hidden : T #[batch, seq, cfg.hidden_size])
    (attentionMask : Option (T #[batch, seq]) := none)
    (cacheConv : Option (T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim]) := none)
    (cacheRec : Option (T #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim]) := none)
    (usePrecomputedState : Bool := false)
    : T #[batch, seq, cfg.hidden_size]
      × Option (T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim])
      × Option (T #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim]) :=
  let hidden := applyMaskToPaddingStates hidden attentionMask

  let mixedQKV : T #[batch, seq, Config.linearConvDim cfg] := linear3d hidden m.in_proj_qkv
  let mixedQKVc : T #[batch, Config.linearConvDim cfg, seq] := nn.transpose3d_12 mixedQKV

  let zFlat : T #[batch, seq, Config.linearValueDim cfg] := linear3d hidden m.in_proj_z
  let z : T #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim] :=
    reshape zFlat #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim]

  let b : T #[batch, seq, cfg.linear_num_value_heads] := linear3d hidden m.in_proj_b
  let a : T #[batch, seq, cfg.linear_num_value_heads] := linear3d hidden m.in_proj_a

  let (convOut, nextConvState) :
      T #[batch, Config.linearConvDim cfg, seq]
        × Option (T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim]) :=
    if usePrecomputedState && seq == 1 then
      match cacheConv with
      | some st =>
        let catIn : T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim + 1] := nn.cat st mixedQKVc 2
        let stNext : T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] :=
          data.slice catIn 2 1 cfg.linear_conv_kernel_dim
        let raw : T #[batch, Config.linearConvDim cfg, 2] :=
          nn.conv1d_group_bias catIn m.conv1d_weight m.conv1d_bias 1 0 1 (Config.linearConvDim cfg)
        let y : T #[batch, Config.linearConvDim cfg, 1] := data.slice raw 2 1 1
        (nn.silu y, some stNext)
      | none =>
        let raw : T #[batch, Config.linearConvDim cfg, seq + cfg.linear_conv_kernel_dim - 1] :=
          nn.conv1d_group_bias mixedQKVc m.conv1d_weight m.conv1d_bias 1 (cfg.linear_conv_kernel_dim - 1) 1 (Config.linearConvDim cfg)
        let y : T #[batch, Config.linearConvDim cfg, seq] := data.slice raw 2 0 seq
        let nextState :=
          if seq >= cfg.linear_conv_kernel_dim then
            some (data.slice mixedQKVc 2 (seq - cfg.linear_conv_kernel_dim) cfg.linear_conv_kernel_dim)
          else
            let padLen : UInt64 := cfg.linear_conv_kernel_dim - seq
            let z0 : T #[batch, Config.linearConvDim cfg, padLen] := torch.zeros #[batch, Config.linearConvDim cfg, padLen]
            some (nn.cat z0 mixedQKVc 2)
        (nn.silu y, nextState)
    else
      let raw : T #[batch, Config.linearConvDim cfg, seq + cfg.linear_conv_kernel_dim - 1] :=
        nn.conv1d_group_bias mixedQKVc m.conv1d_weight m.conv1d_bias 1 (cfg.linear_conv_kernel_dim - 1) 1 (Config.linearConvDim cfg)
      let y : T #[batch, Config.linearConvDim cfg, seq] := data.slice raw 2 0 seq
      let nextState :=
        if seq >= cfg.linear_conv_kernel_dim then
          some (data.slice mixedQKVc 2 (seq - cfg.linear_conv_kernel_dim) cfg.linear_conv_kernel_dim)
        else
          let padLen : UInt64 := cfg.linear_conv_kernel_dim - seq
          let z0 : T #[batch, Config.linearConvDim cfg, padLen] := torch.zeros #[batch, Config.linearConvDim cfg, padLen]
          some (nn.cat z0 mixedQKVc 2)
      (nn.silu y, nextState)

  let mixed : T #[batch, seq, Config.linearConvDim cfg] := nn.transpose3d_12 convOut

  let keyDim : UInt64 := Config.linearKeyDim cfg
  let valDim : UInt64 := Config.linearValueDim cfg

  let qRaw : T #[batch, seq, keyDim] := data.slice mixed 2 0 keyDim
  let kRaw : T #[batch, seq, keyDim] := data.slice mixed 2 keyDim keyDim
  let vRaw : T #[batch, seq, valDim] := data.slice mixed 2 (keyDim * 2) valDim

  let q0 : T #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim] :=
    reshape qRaw #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim]
  let k0 : T #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim] :=
    reshape kRaw #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim]
  let v : T #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim] :=
    reshape vRaw #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim]

  let beta : T #[batch, seq, cfg.linear_num_value_heads] := nn.sigmoid b

  let dt : T #[batch, seq, cfg.linear_num_value_heads] :=
    nn.expand (reshape m.dt_bias #[1, 1, cfg.linear_num_value_heads]) #[batch, seq, cfg.linear_num_value_heads]
  let aExp : T #[batch, seq, cfg.linear_num_value_heads] :=
    nn.expand (reshape (nn.exp (toFloat' m.a_log)) #[1, 1, cfg.linear_num_value_heads]) #[batch, seq, cfg.linear_num_value_heads]
  let gPos : T #[batch, seq, cfg.linear_num_value_heads] := softplus (toFloat' (a + dt))
  let g : T #[batch, seq, cfg.linear_num_value_heads] := mul_scalar (gPos * aExp) (-1.0)

  let rep := Config.linearKVRepeat cfg
  let q : T #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim] :=
    repeatHeads (rep := rep) q0
  let k : T #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim] :=
    repeatHeads (rep := rep) k0

  let (core, finalState) :=
    recurrentGatedDelta
      q
      k
      v
      g
      beta
      (if usePrecomputedState then cacheRec else none)

  let core2d : T #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim] :=
    reshape core #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim]
  let z2d : T #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim] :=
    reshape z #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim]
  let core2d := m.norm.forward2d core2d z2d
  let core3d : T #[batch, seq, valDim] := reshape core2d #[batch, seq, valDim]

  let out : T #[batch, seq, cfg.hidden_size] := linear3d core3d m.out_proj
  (out, nextConvState, some finalState)

end Qwen35GatedDeltaNet

/-- Layer-local cache bundle for hybrid token mixers. -/
structure HybridCache (cfg : Config) (batch : UInt64) where
  attnCaches : Array (Option (Qwen35Attention.KVCache cfg batch))
  convStates : Array (Option (T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim]))
  recurrentStates : Array (Option (T #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim]))
  lastLinearLayer : Option Nat := none

namespace HybridCache

def hasPreviousState (c : HybridCache cfg batch) : Bool :=
  match c.lastLinearLayer with
  | none => false
  | some idx =>
    match c.convStates[idx]? with
    | some (some _) => true
    | _ => false

end HybridCache

/-- One Qwen3.5 decoder layer (hybrid token mixer + dense/MoE FFN). -/
structure Qwen35Layer (cfg : Config) where
  layerType : LayerType
  input_layernorm : Qwen35RMSNorm cfg.hidden_size
  full_attn : Option (Qwen35Attention cfg) := none
  linear_attn : Option (Qwen35GatedDeltaNet cfg) := none
  post_attention_layernorm : Qwen35RMSNorm cfg.hidden_size
  dense_mlp : Option (Qwen35MLP cfg.hidden_size cfg.intermediate_size) := none
  sparse_moe : Option (Qwen35SparseMoeBlock cfg) := none

namespace Qwen35Layer

def init (cfg : Config) (layerType : LayerType) : IO (Qwen35Layer cfg) := do
  let inputNorm := Qwen35RMSNorm.initZeroCentered cfg.hidden_size cfg.rms_norm_eps
  let postNorm := Qwen35RMSNorm.initZeroCentered cfg.hidden_size cfg.rms_norm_eps

  let fullAttn ←
    match layerType with
    | .fullAttention =>
      let m ← Qwen35Attention.init cfg
      pure (some m)
    | .linearAttention => pure none

  let linearAttn ←
    match layerType with
    | .linearAttention =>
      let m ← Qwen35GatedDeltaNet.init cfg
      pure (some m)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← Qwen35MLP.init cfg.hidden_size cfg.intermediate_size
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← Qwen35SparseMoeBlock.init cfg
      pure (some m)
    else
      pure none

  pure {
    layerType := layerType
    input_layernorm := inputNorm
    full_attn := fullAttn
    linear_attn := linearAttn
    post_attention_layernorm := postNorm
    dense_mlp := denseMlp
    sparse_moe := sparseMoe
  }

def forward {batch seq : UInt64}
    (cfg : Config)
    (layer : Qwen35Layer cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, Config.rotaryHalfDim cfg])
    (sin : T #[seq, Config.rotaryHalfDim cfg])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.hidden_size] :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let mixed :=
    match layer.layerType, layer.full_attn, layer.linear_attn, attnMask with
    | .fullAttention, some a, _, some m => a.forwardMasked cfg h1 cos sin m true
    | .fullAttention, some a, _, none => a.forward cfg h1 cos sin true
    | .linearAttention, _, some l, _ =>
      let (y, _, _) := l.forward cfg h1 attnMask none none false
      y
    | _, _, _, _ => h1
  let h2 := residual1 + mixed

  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  residual2 + ffn

def forwardStep {batch : UInt64}
    (cfg : Config)
    (layer : Qwen35Layer cfg)
    (x : T #[batch, 1, cfg.hidden_size])
    (cos : T #[1, Config.rotaryHalfDim cfg])
    (sin : T #[1, Config.rotaryHalfDim cfg])
    (cache : HybridCache cfg batch)
    (layerIdx : Nat)
    : T #[batch, 1, cfg.hidden_size] × HybridCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x

  let (mixed, cache1) :=
    match layer.layerType with
    | .fullAttention =>
      match layer.full_attn, cache.attnCaches[layerIdx]? with
      | some a, some (some kv) =>
        let (out, kv') := a.forwardStep cfg h1 cos sin kv
        let c' := { cache with attnCaches := cache.attnCaches.set! layerIdx (some kv') }
        (out, c')
      | _, _ =>
        (h1, cache)
    | .linearAttention =>
      match layer.linear_attn with
      | some l =>
        let convOpt := cache.convStates[layerIdx]?.getD none
        let recOpt := cache.recurrentStates[layerIdx]?.getD none
        let usePrev := cache.hasPreviousState && convOpt.isSome && recOpt.isSome
        let (out, conv', rec') := l.forward cfg h1 none convOpt recOpt usePrev
        let c' := {
          cache with
          convStates := cache.convStates.set! layerIdx conv'
          recurrentStates := cache.recurrentStates.set! layerIdx rec'
        }
        (out, c')
      | none => (h1, cache)

  let h2 := residual1 + mixed
  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  (residual2 + ffn, cache1)

end Qwen35Layer

/-- Text-only Qwen3.5 base model. -/
structure Qwen35Model (cfg : Config) where
  embed_tokens : T #[cfg.vocab_size, cfg.hidden_size]
  layers : Array (Qwen35Layer cfg)
  norm : Qwen35RMSNorm cfg.hidden_size

namespace Qwen35Model

def init (cfg : Config) : IO (Qwen35Model cfg) := do
  let embRaw ← torch.randn #[cfg.vocab_size, cfg.hidden_size]
  let embedTokens := autograd.set_requires_grad (mul_scalar embRaw 0.02) true

  let layerTypes := Config.normalizedLayerTypes cfg
  let mut layers : Array (Qwen35Layer cfg) := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let lt := layerTypes.getD i .linearAttention
    let l ← Qwen35Layer.init cfg lt
    layers := layers.push l

  let norm := Qwen35RMSNorm.initZeroCentered cfg.hidden_size cfg.rms_norm_eps
  pure { embed_tokens := embedTokens, layers := layers, norm := norm }

def initCache {batch : UInt64}
    (cfg : Config)
    (m : Qwen35Model cfg)
    (maxLen : UInt64)
    (device : Device)
    : HybridCache cfg batch :=
  Id.run do
    let layerTypes := Config.normalizedLayerTypes cfg
    let mut attn : Array (Option (Qwen35Attention.KVCache cfg batch)) := #[]
    let mut conv : Array (Option (T #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim])) := #[]
    let mut recStates : Array (Option (T #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim])) := #[]
    let mut lastLinear : Option Nat := none

    for i in [:m.layers.size] do
      let lt := layerTypes.getD i .linearAttention
      match lt with
      | .fullAttention =>
        let kv := qwen.QwenAttention.initKVCache
          maxLen
          (batch := batch)
          (num_kv_heads := cfg.num_key_value_heads)
          (head_dim := cfg.head_dim)
          device
        attn := attn.push (some kv)
        conv := conv.push none
        recStates := recStates.push none
      | .linearAttention =>
        attn := attn.push none
        conv := conv.push none
        recStates := recStates.push none
        lastLinear := some i

    return {
      attnCaches := attn
      convStates := conv
      recurrentStates := recStates
      lastLinearLayer := lastLinear
    }

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Model cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.hidden_size] :=
  let x0 : T #[batch, seq, cfg.hidden_size] := nn.embedding inputIds m.embed_tokens
  let (cos, sin) := rotary.computeFreqsPure seq (Config.rotaryDim cfg) cfg.rope_theta
  let h :=
    m.layers.foldl
      (fun h layer => layer.forward cfg h cos sin attnMask)
      x0
  m.norm.forward3d h

end Qwen35Model

/-- Causal-LM output container. -/
structure CausalLMOutput (cfg : Config) (batch seq : UInt64) where
  logits : T #[batch, seq, cfg.vocab_size]

/-- Full standalone Qwen3.5 causal language model (dense or MoE depending on config). -/
structure Qwen35ForCausalLM (cfg : Config) where
  model : Qwen35Model cfg
  lmHead : T #[cfg.vocab_size, cfg.hidden_size]
  tieWordEmbeddings : Bool := true

namespace Qwen35ForCausalLM

def init (cfg : Config) (tieWordEmbeddings : Bool := true) : IO (Qwen35ForCausalLM cfg) := do
  let model ← Qwen35Model.init cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initWeight #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

def embedTokens {batch seq : UInt64}
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  nn.embedding inputIds m.model.embed_tokens

def forwardEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocab_size] :=
  let (cos, sin) := rotary.computeFreqsPure seq (Config.rotaryDim cfg) cfg.rope_theta
  let hidden := m.model.layers.foldl (fun h layer => layer.forward cfg h cos sin attnMask) inputsEmbeds
  let hidden := m.model.norm.forward3d hidden
  linear3d hidden m.lmHead

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.vocab_size] :=
  let embeds := m.embedTokens inputIds
  m.forwardEmbeds cfg embeds attnMask

private def decodeStepFromEmbedWithCache {batch : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (tokenEmbed : T #[batch, 1, cfg.hidden_size])
    (position : UInt64)
    (cache : HybridCache cfg batch)
    : IO (T #[batch, cfg.vocab_size] × HybridCache cfg batch) := do
  let freqLen := position + 1
  let (cosAll, sinAll) := rotary.computeFreqsPure freqLen (Config.rotaryDim cfg) cfg.rope_theta
  let cos : T #[1, Config.rotaryHalfDim cfg] := data.slice cosAll 0 position 1
  let sin : T #[1, Config.rotaryHalfDim cfg] := data.slice sinAll 0 position 1

  let mut hidden : T #[batch, 1, cfg.hidden_size] := tokenEmbed
  let mut cache' := cache

  for i in [:m.model.layers.size] do
    let layer ←
      match m.model.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Qwen35 layer at index {i}"
    let (hNext, cNext) := layer.forwardStep cfg hidden cos sin cache' i
    hidden := hNext
    cache' := cNext

  let hiddenNorm := m.model.norm.forward3d hidden
  let logits3 : T #[batch, 1, cfg.vocab_size] := linear3d hiddenNorm m.lmHead
  let logits2 : T #[batch, cfg.vocab_size] := reshape logits3 #[batch, cfg.vocab_size]
  pure (logits2, cache')

private partial def prefillCachesFromEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (cache : HybridCache cfg batch)
    (position : Nat)
    (lastLogits : T #[batch, cfg.vocab_size])
    : IO (T #[batch, cfg.vocab_size] × HybridCache cfg batch) := do
  if position >= seq.toNat then
    pure (lastLogits, cache)
  else
    let tok : T #[batch, 1, cfg.hidden_size] := data.slice inputsEmbeds 1 position.toUInt64 1
    let (logits, cache') ← decodeStepFromEmbedWithCache cfg m tok position.toUInt64 cache
    prefillCachesFromEmbeds cfg m inputsEmbeds cache' (position + 1) logits

inductive SamplingStrategy where
  | greedy
  | multinomial (temperature : Float := 1.0) (topK : UInt64 := 0) (topP : Float := 1.0)
  deriving Repr, Inhabited

abbrev StreamCallback (batch : UInt64) := UInt64 → T #[batch] → IO Unit

private def sampleFromLogits {batch vocab : UInt64}
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

private partial def decodeLoopCached {batch : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (remaining : Nat)
    (cache : HybridCache cfg batch)
    (lastLogits : T #[batch, cfg.vocab_size])
    (onStep : Option (StreamCallback batch))
    (generatedSoFar : UInt64)
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTok ← sampleFromLogits lastLogits strategy
  match onStep with
  | some cb => cb generatedSoFar nextTok
  | none => pure ()
  let nextVals ← data.tensorToUInt64Array nextTok
  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    let nextEmb : T #[batch, 1, cfg.hidden_size] := m.embedTokens nextCol
    let (nextLogits, cache') ← decodeStepFromEmbedWithCache cfg m nextEmb curSeq cache
    decodeLoopCached cfg m strategy eosTokenIds (remaining - 1) cache' nextLogits onStep (generatedSoFar + 1) appended

private partial def decodeLoopUncached {batch : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (remaining : Nat)
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"

  let logits := m.forward cfg curIds none
  let lastPos := curSeq - 1
  let last3 : T #[batch, 1, cfg.vocab_size] := reshape (data.slice logits 1 lastPos 1) #[batch, 1, cfg.vocab_size]
  let last2 : T #[batch, cfg.vocab_size] := reshape last3 #[batch, cfg.vocab_size]

  let nextTok ← sampleFromLogits last2 strategy
  let nextVals ← data.tensorToUInt64Array nextTok
  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    decodeLoopUncached cfg m strategy eosTokenIds (remaining - 1) appended

/-- Generic generation entry point (cached decode). -/
private def generateFromEmbedsCore {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let cacheDevice := inputsEmbeds.device
  let cacheMaxLen : UInt64 := seq + maxNewTokens
  let cache0 := m.model.initCache cfg cacheMaxLen cacheDevice

  let tok0 : T #[batch, 1, cfg.hidden_size] := data.slice inputsEmbeds 1 0 1
  let (logits0, cache1) ← decodeStepFromEmbedWithCache cfg m tok0 0 cache0
  let (lastLogits, cachePrefill) ← prefillCachesFromEmbeds cfg m inputsEmbeds cache1 1 logits0

  decodeLoopCached cfg m strategy eosTokenIds maxNewTokens.toNat cachePrefill lastLogits onStep 0 inputIds

/-- Cached generation from explicit input embeddings.
    Useful for multimodal wrappers that inject vision features into token embeddings. -/
def generateFromEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

/-- Cached generation from explicit input embeddings with per-step token callback. -/
def generateFromEmbedsStream {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

/-- Generic generation entry point (cached decode). -/
def generate {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let inputsEmbeds := m.embedTokens inputIds
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

/-- Cached generation with per-step token callback (streaming). -/
def generateStream {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let inputsEmbeds := m.embedTokens inputIds
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

/-- Reference generation path by full re-forward each decode step. -/
def generateUncached {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  decodeLoopUncached cfg m strategy eosTokenIds maxNewTokens.toNat inputIds

/-- Convenience wrapper for greedy generation. -/
def generateGreedy {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generate cfg m inputIds maxNewTokens .greedy eosTokenIds

end Qwen35ForCausalLM

end torch.qwen35
