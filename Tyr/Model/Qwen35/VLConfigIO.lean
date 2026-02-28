/-
  Tyr/Model/Qwen35/VLConfigIO.lean

  HuggingFace `config.json` loader for Qwen3.5 multimodal checkpoints.
-/
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.VLConfig
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen35

open Lean

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def fromJson? {α} [FromJson α] (j : Json) : Option α :=
  match (FromJson.fromJson? j : Except String α) with
  | .ok v => some v
  | .error _ => none

private def getNat? (j : Json) : Option Nat := fromJson? j
private def getBool? (j : Json) : Option Bool := fromJson? j
private def getString? (j : Json) : Option String := fromJson? j

private def getFloat? (j : Json) : Option Float :=
  match (fromJson? (α := Float) j) with
  | some x => some x
  | none => (getNat? j).map (·.toFloat)

private def getNatFieldD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getNat? with
  | some n => n.toUInt64
  | none => d

private def getBoolFieldD (j : Json) (key : String) (d : Bool) : Bool :=
  match getObjVal? j key >>= getBool? with
  | some b => b
  | none => d

private def getStringFieldD (j : Json) (key : String) (d : String) : String :=
  match getObjVal? j key >>= getString? with
  | some s => s
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def parseVisionConfig
    (j : Json)
    (d : VisionConfig := {})
    : VisionConfig :=
  {
    depth := getNatFieldD j "depth" d.depth
    hidden_size := getNatFieldD j "hidden_size" d.hidden_size
    hidden_act := getStringFieldD j "hidden_act" d.hidden_act
    intermediate_size := getNatFieldD j "intermediate_size" d.intermediate_size
    num_heads := getNatFieldD j "num_heads" d.num_heads
    in_channels := getNatFieldD j "in_channels" d.in_channels
    patch_size := getNatFieldD j "patch_size" d.patch_size
    spatial_merge_size := getNatFieldD j "spatial_merge_size" d.spatial_merge_size
    temporal_patch_size := getNatFieldD j "temporal_patch_size" d.temporal_patch_size
    out_hidden_size := getNatFieldD j "out_hidden_size" d.out_hidden_size
    num_position_embeddings := getNatFieldD j "num_position_embeddings" d.num_position_embeddings
    initializer_range := getFloatFieldD j "initializer_range" d.initializer_range
  }

private def parseVLConfig
    (root : Json)
    (d : VLConfig := {})
    : VLConfig :=
  let textJson :=
    match getObjVal? root "text_config" with
    | some t => t
    | none => root
  let textCfg :=
    match getObjVal? root "text_config" with
    | some _ =>
      -- Parse true nested text config using Qwen35 text parser defaults.
      let defaults := d.text_config
      let hidden :=
        match getObjVal? textJson "hidden_size" >>= getNat? with
        | some n => n.toUInt64
        | none => defaults.hidden_size
      let heads :=
        match getObjVal? textJson "num_attention_heads" >>= getNat? with
        | some n => n.toUInt64
        | none => defaults.num_attention_heads
      let headDimDefault :=
        if heads == 0 then defaults.head_dim else hidden / heads
      let ropeTheta :=
        match getObjVal? textJson "rope_parameters" >>= (fun x => getObjVal? x "rope_theta") >>= getFloat? with
        | some x => x
        | none =>
          match getObjVal? textJson "rope_theta" >>= getFloat? with
          | some x => x
          | none => defaults.rope_theta
      let partialRotary :=
        match getObjVal? textJson "rope_parameters" >>= (fun x => getObjVal? x "partial_rotary_factor") >>= getFloat? with
        | some x => x
        | none =>
          match getObjVal? textJson "partial_rotary_factor" >>= getFloat? with
          | some x => x
          | none => defaults.partial_rotary_factor
      let layerTypes : Array LayerType :=
        match getObjVal? textJson "layer_types" with
        | some (.arr xs) =>
          let mapped := xs.map (fun x => (getString? x >>= LayerType.ofString?))
          if mapped.all Option.isSome then
            mapped.map (fun x => x.getD .linearAttention)
          else
            defaults.layer_types
        | _ => defaults.layer_types
      {
        vocab_size := getNatFieldD textJson "vocab_size" defaults.vocab_size
        hidden_size := hidden
        intermediate_size := getNatFieldD textJson "intermediate_size" defaults.intermediate_size
        num_hidden_layers := getNatFieldD textJson "num_hidden_layers" defaults.num_hidden_layers
        num_attention_heads := heads
        num_key_value_heads := getNatFieldD textJson "num_key_value_heads" defaults.num_key_value_heads
        head_dim := getNatFieldD textJson "head_dim" headDimDefault
        rope_theta := ropeTheta
        partial_rotary_factor := partialRotary
        rms_norm_eps := getFloatFieldD textJson "rms_norm_eps" defaults.rms_norm_eps
        max_position_embeddings := getNatFieldD textJson "max_position_embeddings" defaults.max_position_embeddings
        attention_bias := getBoolFieldD textJson "attention_bias" defaults.attention_bias
        attention_dropout := getFloatFieldD textJson "attention_dropout" defaults.attention_dropout
        hidden_act := getStringFieldD textJson "hidden_act" defaults.hidden_act
        linear_conv_kernel_dim := getNatFieldD textJson "linear_conv_kernel_dim" defaults.linear_conv_kernel_dim
        linear_key_head_dim := getNatFieldD textJson "linear_key_head_dim" defaults.linear_key_head_dim
        linear_value_head_dim := getNatFieldD textJson "linear_value_head_dim" defaults.linear_value_head_dim
        linear_num_key_heads := getNatFieldD textJson "linear_num_key_heads" defaults.linear_num_key_heads
        linear_num_value_heads := getNatFieldD textJson "linear_num_value_heads" defaults.linear_num_value_heads
        layer_types := layerTypes
        full_attention_interval := getNatFieldD textJson "full_attention_interval" defaults.full_attention_interval
        moe_intermediate_size := getNatFieldD textJson "moe_intermediate_size" defaults.moe_intermediate_size
        shared_expert_intermediate_size :=
          getNatFieldD textJson "shared_expert_intermediate_size" defaults.shared_expert_intermediate_size
        num_experts_per_tok := getNatFieldD textJson "num_experts_per_tok" defaults.num_experts_per_tok
        num_experts := getNatFieldD textJson "num_experts" defaults.num_experts
        use_cache := getBoolFieldD textJson "use_cache" defaults.use_cache
        tie_word_embeddings :=
          match getObjVal? root "tie_word_embeddings" >>= getBool? with
          | some x => x
          | none => getBoolFieldD textJson "tie_word_embeddings" defaults.tie_word_embeddings
        pad_token_id :=
          match getObjVal? textJson "pad_token_id" >>= getNat? with
          | some x => some x.toUInt64
          | none => defaults.pad_token_id
        bos_token_id :=
          match getObjVal? textJson "bos_token_id" >>= getNat? with
          | some x => some x.toUInt64
          | none => defaults.bos_token_id
        eos_token_id :=
          match getObjVal? textJson "eos_token_id" >>= getNat? with
          | some x => some x.toUInt64
          | none => defaults.eos_token_id
      }
    | none =>
      -- Backward compatible: plain text-only config at root.
      let cfg := Config.normalize <| {
        d.text_config with
        vocab_size := getNatFieldD root "vocab_size" d.text_config.vocab_size
      }
      cfg

  let visionCfg :=
    match getObjVal? root "vision_config" with
    | some v => parseVisionConfig v d.vision_config
    | none => d.vision_config

  let tieWordEmbeddings :=
    match getObjVal? root "tie_word_embeddings" >>= getBool? with
    | some x => x
    | none => textCfg.tie_word_embeddings

  VLConfig.normalize {
    text_config := textCfg
    vision_config := visionCfg
    image_token_id := getNatFieldD root "image_token_id" d.image_token_id
    video_token_id := getNatFieldD root "video_token_id" d.video_token_id
    vision_start_token_id := getNatFieldD root "vision_start_token_id" d.vision_start_token_id
    vision_end_token_id := getNatFieldD root "vision_end_token_id" d.vision_end_token_id
    tie_word_embeddings := tieWordEmbeddings
  }

namespace VLConfig

/-- Load Qwen3.5 multimodal config from HuggingFace-style `config.json`. -/
def loadFromFile (path : String) (defaults : VLConfig := {}) : IO VLConfig := do
  let root ← parseJsonFile path
  pure (parseVLConfig root defaults)

/-- Load Qwen3.5 multimodal config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : VLConfig := {}) : IO VLConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end VLConfig

end torch.qwen35
