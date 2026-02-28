/-
  Tyr/Model/Qwen35/VLConfig.lean

  Multimodal (vision + text) configuration for standalone Qwen3.5.
-/
import Tyr.Model.Qwen35.Config

namespace torch.qwen35

/-- Vision backbone config used by Qwen3.5 multimodal checkpoints. -/
structure Qwen35VisionConfig where
  depth : UInt64 := 27
  hidden_size : UInt64 := 1152
  hidden_act : String := "gelu_pytorch_tanh"
  intermediate_size : UInt64 := 4304
  num_heads : UInt64 := 16
  in_channels : UInt64 := 3
  patch_size : UInt64 := 16
  spatial_merge_size : UInt64 := 2
  temporal_patch_size : UInt64 := 2
  out_hidden_size : UInt64 := 3584
  num_position_embeddings : UInt64 := 2304
  initializer_range : Float := 0.02
  deriving Repr, Inhabited

namespace Qwen35VisionConfig

def patchDim (cfg : Qwen35VisionConfig) : UInt64 :=
  cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size

def mergeUnit (cfg : Qwen35VisionConfig) : UInt64 :=
  cfg.spatial_merge_size * cfg.spatial_merge_size

def headDim (cfg : Qwen35VisionConfig) : UInt64 :=
  if cfg.num_heads == 0 then cfg.hidden_size else cfg.hidden_size / cfg.num_heads

def mergedTokenCount (cfg : Qwen35VisionConfig) (nPatches : UInt64) : UInt64 :=
  let u := mergeUnit cfg
  if u == 0 then nPatches else nPatches / u

end Qwen35VisionConfig

/-- Composite Qwen3.5 multimodal config (`text_config` + `vision_config`). -/
structure Qwen35VLConfig where
  text_config : Config := Config.qwen35_9B
  vision_config : Qwen35VisionConfig := {}
  image_token_id : UInt64 := 248056
  video_token_id : UInt64 := 248057
  vision_start_token_id : UInt64 := 248053
  vision_end_token_id : UInt64 := 248054
  tie_word_embeddings : Bool := false
  deriving Repr, Inhabited

namespace Qwen35VLConfig

def normalize (cfg : Qwen35VLConfig) : Qwen35VLConfig :=
  { cfg with text_config := Config.normalize cfg.text_config }

end Qwen35VLConfig

abbrev VisionConfig := Qwen35VisionConfig
abbrev VLConfig := Qwen35VLConfig

namespace VisionConfig

def patchDim (cfg : VisionConfig) : UInt64 := Qwen35VisionConfig.patchDim cfg
def mergeUnit (cfg : VisionConfig) : UInt64 := Qwen35VisionConfig.mergeUnit cfg
def headDim (cfg : VisionConfig) : UInt64 := Qwen35VisionConfig.headDim cfg
def mergedTokenCount (cfg : VisionConfig) (nPatches : UInt64) : UInt64 :=
  Qwen35VisionConfig.mergedTokenCount cfg nPatches

end VisionConfig

namespace VLConfig

def normalize (cfg : VLConfig) : VLConfig := Qwen35VLConfig.normalize cfg

end VLConfig

end torch.qwen35
