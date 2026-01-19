/-
  Tyr/Model/Qwen/Weights.lean

  Weight loading for Qwen3-4B from SafeTensors format.
  Maps HuggingFace weight names to Tyr structure.
  Supports both single-file and sharded HuggingFace models.
-/
import Tyr.Torch
import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Embedder

namespace torch.qwen

/-- Helper to try loading a tensor, returning none on failure.
    Used for optional weights like Q/K norms. -/
private def tryLoadTensorSharded (modelDir : String) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    let t ← safetensors.loadTensorSharded modelDir name s
    pure (some t)
  catch _ =>
    pure none

/-- Load a single attention layer from sharded SafeTensors.
    Optionally loads Q/K norms if present (for Flux Klein text encoder). -/
def loadAttentionSharded (modelDir : String) (layerIdx : UInt64)
    (hidden_size num_heads num_kv_heads head_dim : UInt64)
    (loadQKNorms : Bool := false)
    : IO (QwenAttention hidden_size num_heads num_kv_heads head_dim) := do
  let layerPrefix := s!"model.layers.{layerIdx}.self_attn"

  -- Load separate projections
  let q_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.q_proj.weight" #[num_heads * head_dim, hidden_size]
  let k_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.k_proj.weight" #[num_kv_heads * head_dim, hidden_size]
  let v_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.v_proj.weight" #[num_kv_heads * head_dim, hidden_size]
  let o_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.o_proj.weight" #[hidden_size, num_heads * head_dim]

  -- Optionally load Q/K norms (Flux Klein text encoder has these)
  let q_norm ← if loadQKNorms then
    tryLoadTensorSharded modelDir s!"{layerPrefix}.q_norm.weight" #[head_dim]
  else
    pure none

  let k_norm ← if loadQKNorms then
    tryLoadTensorSharded modelDir s!"{layerPrefix}.k_norm.weight" #[head_dim]
  else
    pure none

  pure {
    q_proj := autograd.set_requires_grad q_proj false
    k_proj := autograd.set_requires_grad k_proj false
    v_proj := autograd.set_requires_grad v_proj false
    o_proj := autograd.set_requires_grad o_proj false
    q_norm := q_norm.map (autograd.set_requires_grad · false)
    k_norm := k_norm.map (autograd.set_requires_grad · false)
  }

/-- Load a single MLP layer from sharded SafeTensors -/
def loadMLPSharded (modelDir : String) (layerIdx : UInt64)
    (hidden_size intermediate_size : UInt64)
    : IO (QwenMLP hidden_size intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}.mlp"

  let gate_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.gate_proj.weight" #[intermediate_size, hidden_size]
  let up_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.up_proj.weight" #[intermediate_size, hidden_size]
  let down_proj ← safetensors.loadTensorSharded modelDir s!"{layerPrefix}.down_proj.weight" #[hidden_size, intermediate_size]

  pure {
    gate_proj := autograd.set_requires_grad gate_proj false
    up_proj := autograd.set_requires_grad up_proj false
    down_proj := autograd.set_requires_grad down_proj false
  }

/-- Load RMSNorm weights from sharded SafeTensors -/
def loadRMSNormSharded (modelDir : String) (name : String) (dim : UInt64)
    : IO (RMSNorm dim) := do
  let scale ← safetensors.loadTensorSharded modelDir s!"{name}.weight" #[dim]
  pure { scale := autograd.set_requires_grad scale false, eps := ⟨1e-6⟩ }

/-- Load a single transformer layer from sharded SafeTensors -/
def loadLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : QwenConfig)
    (loadQKNorms : Bool := false)
    : IO (QwenLayer cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}"

  let input_layernorm ← loadRMSNormSharded modelDir s!"{layerPrefix}.input_layernorm" cfg.hidden_size
  let self_attn ← loadAttentionSharded modelDir layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim loadQKNorms
  let post_attention_layernorm ← loadRMSNormSharded modelDir s!"{layerPrefix}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLPSharded modelDir layerIdx cfg.hidden_size cfg.intermediate_size

  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Load full Qwen3 model from sharded SafeTensors directory.
    modelDir should contain model.safetensors.index.json for sharded models,
    or model.safetensors for single-file models. -/
def loadQwen3ModelSharded (modelDir : String) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    (loadQKNorms : Bool := false)
    : IO (Qwen3Model cfg) := do
  IO.println s!"Loading Qwen3 model from {modelDir}..."

  -- Token embeddings
  let embed_tokens ← safetensors.loadTensorSharded modelDir "model.embed_tokens.weight" #[cfg.vocab_size, cfg.hidden_size]
  let embed_tokens := autograd.set_requires_grad embed_tokens false
  IO.println "  Loaded embed_tokens"

  -- Layers
  let mut layers := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayerSharded modelDir i.toUInt64 cfg loadQKNorms
    layers := layers.push layer
    if (i + 1) % 7 == 0 then
      IO.println s!"  Loaded {i + 1}/{cfg.num_hidden_layers.toNat} layers"

  -- Final norm
  let norm ← loadRMSNormSharded modelDir "model.norm" cfg.hidden_size
  IO.println "  Loaded final norm"

  IO.println "Qwen3 model loaded successfully!"
  pure { embed_tokens, layers, norm }

-- Legacy functions for backwards compatibility with single-file models

/-- Load a single attention layer from SafeTensors (single file) -/
def loadAttention (path : String) (layerIdx : UInt64)
    (hidden_size num_heads num_kv_heads head_dim : UInt64)
    : IO (QwenAttention hidden_size num_heads num_kv_heads head_dim) := do
  let layerPrefix := s!"model.layers.{layerIdx}.self_attn"

  -- Qwen uses fused QKV in some versions, separate in others
  -- We'll load separate projections
  let q_proj ← safetensors.loadTensor path s!"{layerPrefix}.q_proj.weight" #[num_heads * head_dim, hidden_size]
  let k_proj ← safetensors.loadTensor path s!"{layerPrefix}.k_proj.weight" #[num_kv_heads * head_dim, hidden_size]
  let v_proj ← safetensors.loadTensor path s!"{layerPrefix}.v_proj.weight" #[num_kv_heads * head_dim, hidden_size]
  let o_proj ← safetensors.loadTensor path s!"{layerPrefix}.o_proj.weight" #[hidden_size, num_heads * head_dim]

  pure {
    q_proj := autograd.set_requires_grad q_proj false
    k_proj := autograd.set_requires_grad k_proj false
    v_proj := autograd.set_requires_grad v_proj false
    o_proj := autograd.set_requires_grad o_proj false
  }

/-- Load a single MLP layer from SafeTensors (single file) -/
def loadMLP (path : String) (layerIdx : UInt64)
    (hidden_size intermediate_size : UInt64)
    : IO (QwenMLP hidden_size intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}.mlp"

  let gate_proj ← safetensors.loadTensor path s!"{layerPrefix}.gate_proj.weight" #[intermediate_size, hidden_size]
  let up_proj ← safetensors.loadTensor path s!"{layerPrefix}.up_proj.weight" #[intermediate_size, hidden_size]
  let down_proj ← safetensors.loadTensor path s!"{layerPrefix}.down_proj.weight" #[hidden_size, intermediate_size]

  pure {
    gate_proj := autograd.set_requires_grad gate_proj false
    up_proj := autograd.set_requires_grad up_proj false
    down_proj := autograd.set_requires_grad down_proj false
  }

/-- Load RMSNorm weights from SafeTensors (single file) -/
def loadRMSNorm (path : String) (name : String) (dim : UInt64)
    : IO (RMSNorm dim) := do
  let scale ← safetensors.loadTensor path s!"{name}.weight" #[dim]
  pure { scale := autograd.set_requires_grad scale false, eps := ⟨1e-6⟩ }

/-- Load a single transformer layer from SafeTensors (single file) -/
def loadLayer (path : String) (layerIdx : UInt64) (cfg : QwenConfig)
    : IO (QwenLayer cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim cfg.intermediate_size) := do
  let layerPrefix := s!"model.layers.{layerIdx}"

  let input_layernorm ← loadRMSNorm path s!"{layerPrefix}.input_layernorm" cfg.hidden_size
  let self_attn ← loadAttention path layerIdx cfg.hidden_size cfg.num_attention_heads cfg.num_key_value_heads cfg.head_dim
  let post_attention_layernorm ← loadRMSNorm path s!"{layerPrefix}.post_attention_layernorm" cfg.hidden_size
  let mlp ← loadMLP path layerIdx cfg.hidden_size cfg.intermediate_size

  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

/-- Load full Qwen3 model from SafeTensors file (single file, legacy) -/
def loadQwen3Model (path : String) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    : IO (Qwen3Model cfg) := do
  IO.println s!"Loading Qwen3 model from {path}..."

  -- Token embeddings
  let embed_tokens ← safetensors.loadTensor path "model.embed_tokens.weight" #[cfg.vocab_size, cfg.hidden_size]
  let embed_tokens := autograd.set_requires_grad embed_tokens false
  IO.println "  Loaded embed_tokens"

  -- Layers
  let mut layers := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let layer ← loadLayer path i.toUInt64 cfg
    layers := layers.push layer
    if (i + 1) % 7 == 0 then
      IO.println s!"  Loaded {i + 1}/{cfg.num_hidden_layers.toNat} layers"

  -- Final norm
  let norm ← loadRMSNorm path "model.norm" cfg.hidden_size
  IO.println "  Loaded final norm"

  IO.println "Qwen3 model loaded successfully!"
  pure { embed_tokens, layers, norm }

/-- Load Qwen model as Flux text embedder (single file, legacy) -/
def loadQwenFluxEmbedder (path : String) (cfg : QwenConfig := QwenConfig.qwen3_4B)
    (max_seq : UInt64 := 512)
    (outputLayers : Array UInt64 := #[8, 17, 26])
    : IO (QwenFluxEmbedder cfg max_seq) := do
  let model ← loadQwen3Model path cfg
  QwenFluxEmbedder.fromModel cfg max_seq model outputLayers

/-- Load Qwen model as Flux text embedder from sharded directory.
    This is the preferred method for HuggingFace models with sharded weights. -/
def loadQwenFluxEmbedderSharded (modelDir : String) (cfg : QwenConfig := QwenConfig.fluxKleinTextEncoder)
    (max_seq : UInt64 := 512)
    (outputLayers : Array UInt64 := #[8, 17, 26])  -- Flux2 reference uses these layers
    : IO (QwenFluxEmbedder cfg max_seq) := do
  -- Flux Klein text encoder has Q/K norms
  let model ← loadQwen3ModelSharded modelDir cfg true
  QwenFluxEmbedder.fromModel cfg max_seq model outputLayers

end torch.qwen
