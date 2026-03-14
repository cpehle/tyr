import Tyr.Model.Qwen35

open torch
open torch.qwen35

def summarizeText (cfg : Config) : String :=
  s!"text(hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}, kvHeads={cfg.num_key_value_heads}, headDim={cfg.head_dim}, maxPos={cfg.max_position_embeddings})"

def summarizeVision (cfg : VisionConfig) : String :=
  s!"vision(hidden={cfg.hidden_size}, depth={cfg.depth}, heads={cfg.num_heads}, patch={cfg.patch_size}, merge={cfg.spatial_merge_size}, temporalPatch={cfg.temporal_patch_size}, outHidden={cfg.out_hidden_size})"

def repos : Array String := #[
  "Qwen/Qwen3.5-0.8B",
  "Qwen/Qwen3.5-0.8B-Base"
]

def main : IO Unit := do
  let opts : hub.DownloadOptions := {
    cacheDir := ".model-cache/qwen35"
    revision := "main"
    includeTokenizer := true
  }
  for repo in repos do
    let dir ← hub.resolvePretrainedDir repo opts
    let textCfg ← Config.loadFromPretrainedDir dir Config.qwen35_0_8B
    let vlCfg ← VLConfig.loadFromPretrainedDir dir VLConfig.qwen35_0_8B
    IO.println s!"repo={repo}"
    IO.println s!"dir={dir}"
    IO.println (summarizeText textCfg)
    IO.println (summarizeVision vlCfg.vision_config)
    IO.println s!"tokens(image={vlCfg.image_token_id}, video={vlCfg.video_token_id}, visionStart={vlCfg.vision_start_token_id}, visionEnd={vlCfg.vision_end_token_id})"
    IO.println ""
