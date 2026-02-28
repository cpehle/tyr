/-
  Tyr/Model/Qwen35/Weights.lean

  Pretrained weight loading for standalone Qwen3.5 text causal-LM.
  Supports dense and MoE checkpoints from HuggingFace SafeTensors
  (single-file and sharded).
-/
import Tyr.Torch
import Tyr.Model.Qwen35.Model

namespace torch.qwen35

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (toFloat' t) false

private def centeredScale {dim : UInt64} (w : T #[dim]) : T #[dim] :=
  sub_scalar (toFloat' w) 1.0

private def pushUnique (xs : Array String) (x : String) : Array String :=
  if xs.contains x then xs else xs.push x

private def nameCandidates (name : String) : Array String :=
  let out : Array String := #[]
  let out := pushUnique out name
  let out := pushUnique out s!"language_model.{name}"
  let out := pushUnique out s!"model.language_model.{name}"
  let out :=
    if name.startsWith "model." then
      let suffix := name.drop 6
      let out := pushUnique out s!"model.language_model.{suffix}"
      pushUnique out s!"language_model.{suffix}"
    else
      out
  out

private def tryLoadTensorSharded (modelDir : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensorSharded modelDir name s
    pure (some t)
  catch _ =>
    pure none

private def tryLoadTensor (path : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensor path name s
    pure (some t)
  catch _ =>
    pure none

private def loadTensorShardedCandidates (modelDir : String) (names : Array String) (s : Shape)
    : IO (T s) := do
  for n in names do
    if let some t ← tryLoadTensorSharded modelDir n s then
      return t
  throw <| IO.userError s!"Failed to load tensor (sharded): {names}"

private def loadTensorCandidates (path : String) (names : Array String) (s : Shape)
    : IO (T s) := do
  for n in names do
    if let some t ← tryLoadTensor path n s then
      return t
  throw <| IO.userError s!"Failed to load tensor: {names}"

private def tryLoadTensorShardedCandidates (modelDir : String) (names : Array String) (s : Shape)
    : IO (Option (T s)) := do
  for n in names do
    if let some t ← tryLoadTensorSharded modelDir n s then
      return some t
  pure none

private def tryLoadTensorCandidates (path : String) (names : Array String) (s : Shape)
    : IO (Option (T s)) := do
  for n in names do
    if let some t ← tryLoadTensor path n s then
      return some t
  pure none

/-- Dequantize FP8 weights with blockwise inverse scales (128x128 blocks). -/
private def dequantizeFP8 {outDim inDim : UInt64}
    (weight : T #[outDim, inDim])
    (scaleInv : T #[outDim / 128, inDim / 128])
    : T #[outDim, inDim] :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := toFloat' weight
  let s := toFloat' scaleInv
  let w := reshape w #[outBlocks, 128, inBlocks, 128]
  let s := reshape s #[outBlocks, 1, inBlocks, 1]
  let s := nn.expand s #[outBlocks, 128, inBlocks, 128]
  let w := w * s
  reshape w #[outDim, inDim]

/-- Dequantize FP8 expert weights with blockwise inverse scales per expert.
    weight: [experts, out, in], scale_inv: [experts, out/128, in/128]. -/
private def dequantizeFP8Experts {experts outDim inDim : UInt64}
    (weight : T #[experts, outDim, inDim])
    (scaleInv : T #[experts, outDim / 128, inDim / 128])
    : T #[experts, outDim, inDim] :=
  let outBlocks := outDim / 128
  let inBlocks := inDim / 128
  let w := toFloat' weight
  let s := toFloat' scaleInv
  let w := reshape w #[experts, outBlocks, 128, inBlocks, 128]
  let s := reshape s #[experts, outBlocks, 1, inBlocks, 1]
  let s := nn.expand s #[experts, outBlocks, 128, inBlocks, 128]
  let w := w * s
  reshape w #[experts, outDim, inDim]

private def loadLinearWeightSharded (modelDir : String) (baseName : String) (outDim inDim : UInt64)
    : IO (T #[outDim, inDim]) := do
  let bases := nameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some w ← tryLoadTensorSharded modelDir wName #[outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensorSharded modelDir scaleName #[outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some s => dequantizeFP8 w s
        | none => w
  throw <| IO.userError s!"Failed to load linear weight (sharded): {baseName}"

private def loadLinearWeight (path : String) (baseName : String) (outDim inDim : UInt64)
    : IO (T #[outDim, inDim]) := do
  let bases := nameCandidates baseName
  for b in bases do
    let wName := s!"{b}.weight"
    if let some w ← tryLoadTensor path wName #[outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensor path scaleName #[outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some s => dequantizeFP8 w s
        | none => w
  throw <| IO.userError s!"Failed to load linear weight: {baseName}"

private def loadMoeExpertWeightSharded
    (modelDir : String)
    (baseName : String)
    (experts outDim inDim : UInt64)
    : IO (T #[experts, outDim, inDim]) := do
  let bases := nameCandidates baseName
  for b in bases do
    -- BF16-style checkpoints often store experts tensors directly (no ".weight").
    if let some w ← tryLoadTensorSharded modelDir b #[experts, outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensorSharded modelDir scaleName #[experts, outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some s => dequantizeFP8Experts w s
        | none => w
    -- FP8 checkpoints store them as "<base>.weight" + optional weight_scale_inv.
    let wName := s!"{b}.weight"
    if let some w ← tryLoadTensorSharded modelDir wName #[experts, outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensorSharded modelDir scaleName #[experts, outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some s => dequantizeFP8Experts w s
        | none => w
  throw <| IO.userError s!"Failed to load MoE expert weight (sharded): {baseName}"

private def loadMoeExpertWeight
    (path : String)
    (baseName : String)
    (experts outDim inDim : UInt64)
    : IO (T #[experts, outDim, inDim]) := do
  let bases := nameCandidates baseName
  for b in bases do
    if let some w ← tryLoadTensor path b #[experts, outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensor path scaleName #[experts, outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some s => dequantizeFP8Experts w s
        | none => w
    let wName := s!"{b}.weight"
    if let some w ← tryLoadTensor path wName #[experts, outDim, inDim] then
      let scaleName := s!"{b}.weight_scale_inv"
      let scaleInvOpt ← tryLoadTensor path scaleName #[experts, outDim / 128, inDim / 128]
      return match scaleInvOpt with
        | some s => dequantizeFP8Experts w s
        | none => w
  throw <| IO.userError s!"Failed to load MoE expert weight: {baseName}"

private def loadRMSNormSharded (modelDir : String) (baseName : String) (dim : UInt64)
    : IO (Qwen35RMSNorm dim) := do
  let w ← loadTensorShardedCandidates modelDir (nameCandidates s!"{baseName}.weight") #[dim]
  pure { weight := reqGradFalse (centeredScale w), eps := 1e-6 }

private def loadRMSNorm (path : String) (baseName : String) (dim : UInt64)
    : IO (Qwen35RMSNorm dim) := do
  let w ← loadTensorCandidates path (nameCandidates s!"{baseName}.weight") #[dim]
  pure { weight := reqGradFalse (centeredScale w), eps := 1e-6 }

private def loadRMSNormGatedSharded (modelDir : String) (baseName : String) (dim : UInt64)
    : IO (Qwen35RMSNormGated dim) := do
  let w ← loadTensorShardedCandidates modelDir (nameCandidates s!"{baseName}.weight") #[dim]
  pure { weight := reqGradFalse w, eps := 1e-6 }

private def loadRMSNormGated (path : String) (baseName : String) (dim : UInt64)
    : IO (Qwen35RMSNormGated dim) := do
  let w ← loadTensorCandidates path (nameCandidates s!"{baseName}.weight") #[dim]
  pure { weight := reqGradFalse w, eps := 1e-6 }

private def loadAttentionSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35Attention cfg) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeightSharded modelDir s!"{p}.q_proj" (cfg.num_attention_heads * cfg.head_dim * 2) cfg.hidden_size
  let kProj ← loadLinearWeightSharded modelDir s!"{p}.k_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size
  let vProj ← loadLinearWeightSharded modelDir s!"{p}.v_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size
  let oProj ← loadLinearWeightSharded modelDir s!"{p}.o_proj" cfg.hidden_size (cfg.num_attention_heads * cfg.head_dim)

  let qNorm ← loadRMSNormSharded modelDir s!"{p}.q_norm" cfg.head_dim
  let kNorm ← loadRMSNormSharded modelDir s!"{p}.k_norm" cfg.head_dim

  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadAttention (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35Attention cfg) := do
  let p := s!"model.layers.{layerIdx}.self_attn"
  let qProj ← loadLinearWeight path s!"{p}.q_proj" (cfg.num_attention_heads * cfg.head_dim * 2) cfg.hidden_size
  let kProj ← loadLinearWeight path s!"{p}.k_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size
  let vProj ← loadLinearWeight path s!"{p}.v_proj" (cfg.num_key_value_heads * cfg.head_dim) cfg.hidden_size
  let oProj ← loadLinearWeight path s!"{p}.o_proj" cfg.hidden_size (cfg.num_attention_heads * cfg.head_dim)

  let qNorm ← loadRMSNorm path s!"{p}.q_norm" cfg.head_dim
  let kNorm ← loadRMSNorm path s!"{p}.k_norm" cfg.head_dim

  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm
    k_norm := kNorm
  }

private def loadLinearMixerSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35GatedDeltaNet cfg) := do
  let p := s!"model.layers.{layerIdx}.linear_attn"

  let inQKV ← loadLinearWeightSharded modelDir s!"{p}.in_proj_qkv" (Config.linearConvDim cfg) cfg.hidden_size
  let inZ ← loadLinearWeightSharded modelDir s!"{p}.in_proj_z" (Config.linearValueDim cfg) cfg.hidden_size
  let inB ← loadLinearWeightSharded modelDir s!"{p}.in_proj_b" cfg.linear_num_value_heads cfg.hidden_size
  let inA ← loadLinearWeightSharded modelDir s!"{p}.in_proj_a" cfg.linear_num_value_heads cfg.hidden_size

  let convW ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.conv1d.weight") #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim]
  let convBOpt ← tryLoadTensorShardedCandidates modelDir (nameCandidates s!"{p}.conv1d.bias") #[Config.linearConvDim cfg]
  let convB ←
    match convBOpt with
    | some b => pure b
    | none => pure (torch.zeros #[Config.linearConvDim cfg])

  let dtBias ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.dt_bias") #[cfg.linear_num_value_heads]
  let aLog ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.A_log") #[cfg.linear_num_value_heads]

  let norm ← loadRMSNormGatedSharded modelDir s!"{p}.norm" cfg.linear_value_head_dim
  let outProj ← loadLinearWeightSharded modelDir s!"{p}.out_proj" cfg.hidden_size (Config.linearValueDim cfg)

  pure {
    in_proj_qkv := reqGradFalse inQKV
    in_proj_z := reqGradFalse inZ
    in_proj_b := reqGradFalse inB
    in_proj_a := reqGradFalse inA
    conv1d_weight := reqGradFalse convW
    conv1d_bias := reqGradFalse convB
    dt_bias := reqGradFalse dtBias
    a_log := reqGradFalse aLog
    norm := norm
    out_proj := reqGradFalse outProj
  }

private def loadLinearMixer (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35GatedDeltaNet cfg) := do
  let p := s!"model.layers.{layerIdx}.linear_attn"

  let inQKV ← loadLinearWeight path s!"{p}.in_proj_qkv" (Config.linearConvDim cfg) cfg.hidden_size
  let inZ ← loadLinearWeight path s!"{p}.in_proj_z" (Config.linearValueDim cfg) cfg.hidden_size
  let inB ← loadLinearWeight path s!"{p}.in_proj_b" cfg.linear_num_value_heads cfg.hidden_size
  let inA ← loadLinearWeight path s!"{p}.in_proj_a" cfg.linear_num_value_heads cfg.hidden_size

  let convW ← loadTensorCandidates path (nameCandidates s!"{p}.conv1d.weight") #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim]
  let convBOpt ← tryLoadTensorCandidates path (nameCandidates s!"{p}.conv1d.bias") #[Config.linearConvDim cfg]
  let convB ←
    match convBOpt with
    | some b => pure b
    | none => pure (torch.zeros #[Config.linearConvDim cfg])

  let dtBias ← loadTensorCandidates path (nameCandidates s!"{p}.dt_bias") #[cfg.linear_num_value_heads]
  let aLog ← loadTensorCandidates path (nameCandidates s!"{p}.A_log") #[cfg.linear_num_value_heads]

  let norm ← loadRMSNormGated path s!"{p}.norm" cfg.linear_value_head_dim
  let outProj ← loadLinearWeight path s!"{p}.out_proj" cfg.hidden_size (Config.linearValueDim cfg)

  pure {
    in_proj_qkv := reqGradFalse inQKV
    in_proj_z := reqGradFalse inZ
    in_proj_b := reqGradFalse inB
    in_proj_a := reqGradFalse inA
    conv1d_weight := reqGradFalse convW
    conv1d_bias := reqGradFalse convB
    dt_bias := reqGradFalse dtBias
    a_log := reqGradFalse aLog
    norm := norm
    out_proj := reqGradFalse outProj
  }

private def loadDenseMLPSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35MLP cfg.hidden_size cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeightSharded modelDir s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size
  let up ← loadLinearWeightSharded modelDir s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size
  let down ← loadLinearWeightSharded modelDir s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadDenseMLP (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35MLP cfg.hidden_size cfg.intermediate_size) := do
  let p := s!"model.layers.{layerIdx}.mlp"
  let gate ← loadLinearWeight path s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size
  let up ← loadLinearWeight path s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size
  let down ← loadLinearWeight path s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size
  pure {
    gate_proj := reqGradFalse gate
    up_proj := reqGradFalse up
    down_proj := reqGradFalse down
  }

private def loadSparseMoeSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35SparseMoeBlock cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"

  let routerW ← loadTensorShardedCandidates modelDir (nameCandidates s!"{p}.gate.weight") #[cfg.num_experts, cfg.hidden_size]
  let gu ← loadMoeExpertWeightSharded
    modelDir
    s!"{p}.experts.gate_up_proj"
    cfg.num_experts
    (2 * cfg.moe_intermediate_size)
    cfg.hidden_size
  let down ← loadMoeExpertWeightSharded
    modelDir
    s!"{p}.experts.down_proj"
    cfg.num_experts
    cfg.hidden_size
    cfg.moe_intermediate_size

  let sharedGateProj ← loadLinearWeightSharded modelDir s!"{p}.shared_expert.gate_proj" cfg.shared_expert_intermediate_size cfg.hidden_size
  let sharedUpProj ← loadLinearWeightSharded modelDir s!"{p}.shared_expert.up_proj" cfg.shared_expert_intermediate_size cfg.hidden_size
  let sharedDownProj ← loadLinearWeightSharded modelDir s!"{p}.shared_expert.down_proj" cfg.hidden_size cfg.shared_expert_intermediate_size
  let sharedExpertGate ← loadLinearWeightSharded modelDir s!"{p}.shared_expert_gate" 1 cfg.hidden_size

  let router : Qwen35TopKRouter cfg := { weight := reqGradFalse routerW }
  let experts : Qwen35MoeExperts cfg := {
    gate_up_proj := reqGradFalse gu
    down_proj := reqGradFalse down
  }
  let sharedExpert : Qwen35MLP cfg.hidden_size cfg.shared_expert_intermediate_size := {
    gate_proj := reqGradFalse sharedGateProj
    up_proj := reqGradFalse sharedUpProj
    down_proj := reqGradFalse sharedDownProj
  }

  pure {
    router := router
    experts := experts
    shared_expert := sharedExpert
    shared_expert_gate := reqGradFalse sharedExpertGate
  }

private def loadSparseMoe (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35SparseMoeBlock cfg) := do
  let p := s!"model.layers.{layerIdx}.mlp"

  let routerW ← loadTensorCandidates path (nameCandidates s!"{p}.gate.weight") #[cfg.num_experts, cfg.hidden_size]
  let gu ← loadMoeExpertWeight
    path
    s!"{p}.experts.gate_up_proj"
    cfg.num_experts
    (2 * cfg.moe_intermediate_size)
    cfg.hidden_size
  let down ← loadMoeExpertWeight
    path
    s!"{p}.experts.down_proj"
    cfg.num_experts
    cfg.hidden_size
    cfg.moe_intermediate_size

  let sharedGateProj ← loadLinearWeight path s!"{p}.shared_expert.gate_proj" cfg.shared_expert_intermediate_size cfg.hidden_size
  let sharedUpProj ← loadLinearWeight path s!"{p}.shared_expert.up_proj" cfg.shared_expert_intermediate_size cfg.hidden_size
  let sharedDownProj ← loadLinearWeight path s!"{p}.shared_expert.down_proj" cfg.hidden_size cfg.shared_expert_intermediate_size
  let sharedExpertGate ← loadLinearWeight path s!"{p}.shared_expert_gate" 1 cfg.hidden_size

  let router : Qwen35TopKRouter cfg := { weight := reqGradFalse routerW }
  let experts : Qwen35MoeExperts cfg := {
    gate_up_proj := reqGradFalse gu
    down_proj := reqGradFalse down
  }
  let sharedExpert : Qwen35MLP cfg.hidden_size cfg.shared_expert_intermediate_size := {
    gate_proj := reqGradFalse sharedGateProj
    up_proj := reqGradFalse sharedUpProj
    down_proj := reqGradFalse sharedDownProj
  }

  pure {
    router := router
    experts := experts
    shared_expert := sharedExpert
    shared_expert_gate := reqGradFalse sharedExpertGate
  }

private def loadLayerSharded (modelDir : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35Layer cfg) := do
  let p := s!"model.layers.{layerIdx}"
  let layerTypes := Config.normalizedLayerTypes cfg
  let lt := layerTypes.getD layerIdx.toNat .linearAttention

  let inputNorm ← loadRMSNormSharded modelDir s!"{p}.input_layernorm" cfg.hidden_size
  let postNorm ← loadRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cfg.hidden_size

  let fullAttn ←
    match lt with
    | .fullAttention =>
      let a ← loadAttentionSharded modelDir cfg layerIdx
      pure (some a)
    | .linearAttention => pure none

  let linearAttn ←
    match lt with
    | .linearAttention =>
      let l ← loadLinearMixerSharded modelDir cfg layerIdx
      pure (some l)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← loadDenseMLPSharded modelDir cfg layerIdx
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← loadSparseMoeSharded modelDir cfg layerIdx
      pure (some m)
    else
      pure none

  pure {
    layerType := lt
    input_layernorm := inputNorm
    full_attn := fullAttn
    linear_attn := linearAttn
    post_attention_layernorm := postNorm
    dense_mlp := denseMlp
    sparse_moe := sparseMoe
  }

private def loadLayer (path : String) (cfg : Config) (layerIdx : UInt64)
    : IO (Qwen35Layer cfg) := do
  let p := s!"model.layers.{layerIdx}"
  let layerTypes := Config.normalizedLayerTypes cfg
  let lt := layerTypes.getD layerIdx.toNat .linearAttention

  let inputNorm ← loadRMSNorm path s!"{p}.input_layernorm" cfg.hidden_size
  let postNorm ← loadRMSNorm path s!"{p}.post_attention_layernorm" cfg.hidden_size

  let fullAttn ←
    match lt with
    | .fullAttention =>
      let a ← loadAttention path cfg layerIdx
      pure (some a)
    | .linearAttention => pure none

  let linearAttn ←
    match lt with
    | .linearAttention =>
      let l ← loadLinearMixer path cfg layerIdx
      pure (some l)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← loadDenseMLP path cfg layerIdx
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← loadSparseMoe path cfg layerIdx
      pure (some m)
    else
      pure none

  pure {
    layerType := lt
    input_layernorm := inputNorm
    full_attn := fullAttn
    linear_attn := linearAttn
    post_attention_layernorm := postNorm
    dense_mlp := denseMlp
    sparse_moe := sparseMoe
  }

namespace Qwen35ForCausalLM

/-- Load Qwen3.5 model from sharded HF SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : Config := Config.qwen35_9B)
    : IO (Qwen35ForCausalLM cfg) := do
  IO.println s!"Loading Qwen35ForCausalLM from {modelDir}..."

  let embedTokens ← loadTensorShardedCandidates modelDir (nameCandidates "model.embed_tokens.weight") #[cfg.vocab_size, cfg.hidden_size]

  let mut layers : Array (Qwen35Layer cfg) := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayerSharded modelDir cfg i.toUInt64
    layers := layers.push layer
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers.toNat then
      IO.println s!"  loaded layers {i + 1}/{cfg.num_hidden_layers.toNat}"

  let norm ← loadRMSNormSharded modelDir "model.norm" cfg.hidden_size

  let model : Qwen35Model cfg := {
    embed_tokens := reqGradFalse embedTokens
    layers := layers
    norm := norm
  }

  let lmHeadOpt0 ← tryLoadTensorSharded modelDir "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let lmHeadOpt ←
    match lmHeadOpt0 with
    | some t => pure (some t)
    | none =>
      -- try multimodal-style prefixed key
      tryLoadTensorSharded modelDir "language_model.lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]

  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      IO.println "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  IO.println "Loaded Qwen35ForCausalLM weights."
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

/-- Load Qwen3.5 model from a single HF SafeTensors file. -/
def load (path : String) (cfg : Config := Config.qwen35_9B)
    : IO (Qwen35ForCausalLM cfg) := do
  IO.println s!"Loading Qwen35ForCausalLM from {path}..."

  let embedTokens ← loadTensorCandidates path (nameCandidates "model.embed_tokens.weight") #[cfg.vocab_size, cfg.hidden_size]

  let mut layers : Array (Qwen35Layer cfg) := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayer path cfg i.toUInt64
    layers := layers.push layer
    if (i + 1) % 8 == 0 || i + 1 == cfg.num_hidden_layers.toNat then
      IO.println s!"  loaded layers {i + 1}/{cfg.num_hidden_layers.toNat}"

  let norm ← loadRMSNorm path "model.norm" cfg.hidden_size

  let model : Qwen35Model cfg := {
    embed_tokens := reqGradFalse embedTokens
    layers := layers
    norm := norm
  }

  let lmHeadOpt0 ← tryLoadTensor path "lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]
  let lmHeadOpt ←
    match lmHeadOpt0 with
    | some t => pure (some t)
    | none => tryLoadTensor path "language_model.lm_head.weight" #[cfg.vocab_size, cfg.hidden_size]

  let (lmHead, tieWordEmbeddings) ←
    match lmHeadOpt with
    | some w => pure (reqGradFalse w, false)
    | none => do
      IO.println "  lm_head.weight not found; tying to embeddings."
      pure (reqGradFalse model.embed_tokens, true)

  IO.println "Loaded Qwen35ForCausalLM weights."
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

end Qwen35ForCausalLM

end torch.qwen35
