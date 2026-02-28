/-
  Tyr/Model/Qwen35/VLWeights.lean

  Pretrained weight loading for Qwen3.5 multimodal wrapper.
-/
import Tyr.Torch
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.Multimodal

namespace torch.qwen35

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (toFloat' t) false

private def visionNameCandidates (name : String) : Array String :=
  let out : Array String := #[name]
  let out :=
    if name.startsWith "model.visual." then
      out.push s!"visual.{name.drop 13}"
    else if name.startsWith "visual." then
      out.push s!"model.{name}"
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

private def loadVisionLayerNormSharded (modelDir : String) (base : String) (dim : UInt64)
    : IO (Qwen35VisionLayerNorm dim) := do
  let w ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{base}.weight") #[dim]
  let b ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{base}.bias") #[dim]
  pure { weight := reqGradFalse w, bias := reqGradFalse b, eps := 1e-6 }

private def loadVisionLayerNorm (path : String) (base : String) (dim : UInt64)
    : IO (Qwen35VisionLayerNorm dim) := do
  let w ← loadTensorCandidates path (visionNameCandidates s!"{base}.weight") #[dim]
  let b ← loadTensorCandidates path (visionNameCandidates s!"{base}.bias") #[dim]
  pure { weight := reqGradFalse w, bias := reqGradFalse b, eps := 1e-6 }

private def loadVisionMLPSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Qwen35VisionMLP cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.mlp"
  let w1 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc1.weight") #[cfg.intermediate_size, cfg.hidden_size]
  let b1 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc1.bias") #[cfg.intermediate_size]
  let w2 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc2.weight") #[cfg.hidden_size, cfg.intermediate_size]
  let b2 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc2.bias") #[cfg.hidden_size]
  pure {
    linear_fc1_weight := reqGradFalse w1
    linear_fc1_bias := reqGradFalse b1
    linear_fc2_weight := reqGradFalse w2
    linear_fc2_bias := reqGradFalse b2
  }

private def loadVisionMLP (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Qwen35VisionMLP cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.mlp"
  let w1 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc1.weight") #[cfg.intermediate_size, cfg.hidden_size]
  let b1 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc1.bias") #[cfg.intermediate_size]
  let w2 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc2.weight") #[cfg.hidden_size, cfg.intermediate_size]
  let b2 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc2.bias") #[cfg.hidden_size]
  pure {
    linear_fc1_weight := reqGradFalse w1
    linear_fc1_bias := reqGradFalse b1
    linear_fc2_weight := reqGradFalse w2
    linear_fc2_bias := reqGradFalse b2
  }

private def loadVisionAttentionSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Qwen35VisionAttention cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.attn"
  let qkvW ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.qkv.weight") #[cfg.hidden_size * 3, cfg.hidden_size]
  let qkvB ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.qkv.bias") #[cfg.hidden_size * 3]
  let projW ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.proj.weight") #[cfg.hidden_size, cfg.hidden_size]
  let projB ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.proj.bias") #[cfg.hidden_size]
  pure {
    qkv_weight := reqGradFalse qkvW
    qkv_bias := reqGradFalse qkvB
    proj_weight := reqGradFalse projW
    proj_bias := reqGradFalse projB
  }

private def loadVisionAttention (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Qwen35VisionAttention cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.attn"
  let qkvW ← loadTensorCandidates path (visionNameCandidates s!"{p}.qkv.weight") #[cfg.hidden_size * 3, cfg.hidden_size]
  let qkvB ← loadTensorCandidates path (visionNameCandidates s!"{p}.qkv.bias") #[cfg.hidden_size * 3]
  let projW ← loadTensorCandidates path (visionNameCandidates s!"{p}.proj.weight") #[cfg.hidden_size, cfg.hidden_size]
  let projB ← loadTensorCandidates path (visionNameCandidates s!"{p}.proj.bias") #[cfg.hidden_size]
  pure {
    qkv_weight := reqGradFalse qkvW
    qkv_bias := reqGradFalse qkvB
    proj_weight := reqGradFalse projW
    proj_bias := reqGradFalse projB
  }

private def loadVisionBlockSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Qwen35VisionBlock cfg) := do
  let norm1 ← loadVisionLayerNormSharded modelDir s!"model.visual.blocks.{layerIdx}.norm1" cfg.hidden_size
  let norm2 ← loadVisionLayerNormSharded modelDir s!"model.visual.blocks.{layerIdx}.norm2" cfg.hidden_size
  let attn ← loadVisionAttentionSharded modelDir cfg layerIdx
  let mlp ← loadVisionMLPSharded modelDir cfg layerIdx
  pure { norm1 := norm1, norm2 := norm2, attn := attn, mlp := mlp }

private def loadVisionBlock (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Qwen35VisionBlock cfg) := do
  let norm1 ← loadVisionLayerNorm path s!"model.visual.blocks.{layerIdx}.norm1" cfg.hidden_size
  let norm2 ← loadVisionLayerNorm path s!"model.visual.blocks.{layerIdx}.norm2" cfg.hidden_size
  let attn ← loadVisionAttention path cfg layerIdx
  let mlp ← loadVisionMLP path cfg layerIdx
  pure { norm1 := norm1, norm2 := norm2, attn := attn, mlp := mlp }

private def loadVisionModelSharded (modelDir : String) (cfg : VisionConfig)
    : IO (Qwen35VisionModel cfg) := do
  let convW ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.patch_embed.proj.weight")
    #[cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
  let convB ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.patch_embed.proj.bias")
    #[cfg.hidden_size]
  let patchW : T #[cfg.hidden_size, VisionConfig.patchDim cfg] :=
    reshape convW #[cfg.hidden_size, VisionConfig.patchDim cfg]
  let patchEmbed : Qwen35VisionPatchEmbed cfg := {
    weight := reqGradFalse patchW
    bias := reqGradFalse convB
  }

  let posEmbed ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.pos_embed.weight")
    #[cfg.num_position_embeddings, cfg.hidden_size]

  let mut blocks : Array (Qwen35VisionBlock cfg) := #[]
  for i in [:cfg.depth.toNat] do
    blocks := blocks.push (← loadVisionBlockSharded modelDir cfg i.toUInt64)

  let mergeUnit := VisionConfig.mergeUnit cfg
  let mergedHidden := cfg.hidden_size * mergeUnit
  let mergerNorm ← loadVisionLayerNormSharded modelDir "model.visual.merger.norm" cfg.hidden_size
  let fc1W ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc1.weight")
    #[mergedHidden, mergedHidden]
  let fc1B ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc1.bias")
    #[mergedHidden]
  let fc2W ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc2.weight")
    #[cfg.out_hidden_size, mergedHidden]
  let fc2B ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc2.bias")
    #[cfg.out_hidden_size]

  let merger : Qwen35VisionPatchMerger cfg := {
    norm := mergerNorm
    linear_fc1_weight := reqGradFalse fc1W
    linear_fc1_bias := reqGradFalse fc1B
    linear_fc2_weight := reqGradFalse fc2W
    linear_fc2_bias := reqGradFalse fc2B
  }

  pure {
    patch_embed := patchEmbed
    pos_embed := reqGradFalse posEmbed
    blocks := blocks
    merger := merger
  }

private def loadVisionModel (path : String) (cfg : VisionConfig)
    : IO (Qwen35VisionModel cfg) := do
  let convW ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.patch_embed.proj.weight")
    #[cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
  let convB ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.patch_embed.proj.bias")
    #[cfg.hidden_size]
  let patchW : T #[cfg.hidden_size, VisionConfig.patchDim cfg] :=
    reshape convW #[cfg.hidden_size, VisionConfig.patchDim cfg]
  let patchEmbed : Qwen35VisionPatchEmbed cfg := {
    weight := reqGradFalse patchW
    bias := reqGradFalse convB
  }

  let posEmbed ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.pos_embed.weight")
    #[cfg.num_position_embeddings, cfg.hidden_size]

  let mut blocks : Array (Qwen35VisionBlock cfg) := #[]
  for i in [:cfg.depth.toNat] do
    blocks := blocks.push (← loadVisionBlock path cfg i.toUInt64)

  let mergeUnit := VisionConfig.mergeUnit cfg
  let mergedHidden := cfg.hidden_size * mergeUnit
  let mergerNorm ← loadVisionLayerNorm path "model.visual.merger.norm" cfg.hidden_size
  let fc1W ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc1.weight")
    #[mergedHidden, mergedHidden]
  let fc1B ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc1.bias")
    #[mergedHidden]
  let fc2W ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc2.weight")
    #[cfg.out_hidden_size, mergedHidden]
  let fc2B ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc2.bias")
    #[cfg.out_hidden_size]

  let merger : Qwen35VisionPatchMerger cfg := {
    norm := mergerNorm
    linear_fc1_weight := reqGradFalse fc1W
    linear_fc1_bias := reqGradFalse fc1B
    linear_fc2_weight := reqGradFalse fc2W
    linear_fc2_bias := reqGradFalse fc2B
  }

  pure {
    patch_embed := patchEmbed
    pos_embed := reqGradFalse posEmbed
    blocks := blocks
    merger := merger
  }

namespace Qwen35ForConditionalGeneration

/-- Load Qwen3.5 multimodal model from sharded HF SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : VLConfig := {})
    : IO (Qwen35ForConditionalGeneration cfg) := do
  IO.println s!"Loading Qwen35ForConditionalGeneration from {modelDir}..."
  if cfg.vision_config.out_hidden_size != cfg.text_config.hidden_size then
    throw <| IO.userError
      s!"vision out_hidden_size ({cfg.vision_config.out_hidden_size}) must match text hidden_size ({cfg.text_config.hidden_size})"

  let visual ← loadVisionModelSharded modelDir cfg.vision_config
  let languageModel ← Qwen35ForCausalLM.loadSharded modelDir cfg.text_config
  IO.println "Loaded Qwen35ForConditionalGeneration weights."
  pure { visual := visual, language_model := languageModel }

/-- Load Qwen3.5 multimodal model from a single HF SafeTensors file. -/
def load (path : String) (cfg : VLConfig := {})
    : IO (Qwen35ForConditionalGeneration cfg) := do
  IO.println s!"Loading Qwen35ForConditionalGeneration from {path}..."
  if cfg.vision_config.out_hidden_size != cfg.text_config.hidden_size then
    throw <| IO.userError
      s!"vision out_hidden_size ({cfg.vision_config.out_hidden_size}) must match text hidden_size ({cfg.text_config.hidden_size})"

  let visual ← loadVisionModel path cfg.vision_config
  let languageModel ← Qwen35ForCausalLM.load path cfg.text_config
  IO.println "Loaded Qwen35ForConditionalGeneration weights."
  pure { visual := visual, language_model := languageModel }

end Qwen35ForConditionalGeneration

end torch.qwen35
