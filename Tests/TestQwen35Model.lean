import Tyr
import Tyr.Model.Qwen35
import LeanTest

open torch
open torch.qwen35

private def tinyDenseCfg : Config := {
  vocab_size := 64
  hidden_size := 32
  intermediate_size := 64
  num_hidden_layers := 3
  num_attention_heads := 4
  num_key_value_heads := 2
  head_dim := 8
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 128
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 4
  linear_value_head_dim := 4
  linear_num_key_heads := 2
  linear_num_value_heads := 4
  layer_types := #[.linearAttention, .fullAttention, .linearAttention]
  full_attention_interval := 2
  moe_intermediate_size := 16
  shared_expert_intermediate_size := 16
  num_experts_per_tok := 2
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def tinyMoeCfg : Config := {
  tinyDenseCfg with
  intermediate_size := 0
  moe_intermediate_size := 8
  shared_expert_intermediate_size := 16
  num_experts_per_tok := 2
  num_experts := 4
}

private def tinyBatchedEosCfg : Config := {
  vocab_size := 4
  hidden_size := 4
  intermediate_size := 4
  num_hidden_layers := 0
  num_attention_heads := 1
  num_key_value_heads := 1
  head_dim := 4
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 16
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 1
  linear_key_head_dim := 1
  linear_value_head_dim := 1
  linear_num_key_heads := 1
  linear_num_value_heads := 1
  layer_types := #[]
  full_attention_interval := 1
  moe_intermediate_size := 4
  shared_expert_intermediate_size := 4
  num_experts_per_tok := 1
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def tinyAttentionCfg : Config := {
  vocab_size := 8
  hidden_size := 4
  intermediate_size := 8
  num_hidden_layers := 1
  num_attention_heads := 2
  num_key_value_heads := 2
  head_dim := 2
  rope_theta := 10000.0
  partial_rotary_factor := 1.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 16
  attention_bias := false
  attention_dropout := 0.0
  hidden_act := "silu"
  linear_conv_kernel_dim := 1
  linear_key_head_dim := 1
  linear_value_head_dim := 1
  linear_num_key_heads := 1
  linear_num_value_heads := 1
  layer_types := #[.fullAttention]
  full_attention_interval := 1
  moe_intermediate_size := 4
  shared_expert_intermediate_size := 4
  num_experts_per_tok := 1
  num_experts := 0
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := some 0
  bos_token_id := some 1
  eos_token_id := some 2
}

private def approxEq (a b : Float) (tol : Float := 1e-4) : Bool :=
  Float.abs (a - b) <= tol

private def batchedEosDeterministicModel : IO (Qwen35ForCausalLM tinyBatchedEosCfg) := do
  let base ← Qwen35ForCausalLM.init tinyBatchedEosCfg false
  let embedVals : Array Int64 := #[
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]
  let lmHeadVals : Array Int64 := #[
    0, 0, 0, 0,
    0, 1, 1, 1,
    1, 0, 0, 0,
    0, 0, 0, 0
  ]
  let embed : T #[4, 4] := toFloat' (reshape (data.fromInt64Array embedVals) #[4, 4])
  let lmHead : T #[4, 4] := toFloat' (reshape (data.fromInt64Array lmHeadVals) #[4, 4])
  pure {
    base with
    model := { base.model with embed_tokens := embed }
    lmHead := lmHead
  }

private def rowTokenIds {seq : UInt64}
    (tokens : T #[2, seq])
    (row : UInt64)
    : IO (Array UInt64) := do
  let row2d : T #[1, seq] := data.slice tokens 0 row 1
  let flat : T #[seq] := reshape (data.toLong row2d) #[seq]
  data.tensorToUInt64Array flat

@[test]
def testQwen35DenseInitAndForward : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyDenseCfg
  let ids : T #[1, 5] := reshape (data.fromInt64Array #[1, 2, 3, 4, 5]) #[1, 5]
  let logits := model.forward tinyDenseCfg ids
  LeanTest.assertEqual logits.runtimeShape #[1, 5, tinyDenseCfg.vocab_size]
    "dense forward should return [batch, seq, vocab]"

@[test]
def testQwen35DenseGreedyCachedUncachedParity : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyDenseCfg
  let ids : T #[1, 4] := reshape (data.fromInt64Array #[3, 1, 2, 4]) #[1, 4]

  let ⟨seqCached, outCached⟩ ← model.generateGreedy tinyDenseCfg ids 4 #[]
  let ⟨seqUncached, outUncached⟩ ←
    model.generateUncached tinyDenseCfg ids 4 .greedy #[]

  LeanTest.assertEqual seqCached (4 + 4)
    "cached greedy should append maxNewTokens"
  LeanTest.assertEqual seqUncached (4 + 4)
    "uncached greedy should append maxNewTokens"

  let cachedFlat : T #[seqCached] := reshape (data.toLong outCached) #[seqCached]
  let uncachedFlat : T #[seqUncached] := reshape (data.toLong outUncached) #[seqUncached]
  let cachedIds ← data.tensorToUInt64Array cachedFlat
  let uncachedIds ← data.tensorToUInt64Array uncachedFlat

  LeanTest.assertEqual cachedIds uncachedIds
    "cached and uncached greedy should produce the same tokens"

@[test]
def testQwen35AttentionUsesPerHeadQGateLayout : IO Unit := do
  let cfg := tinyAttentionCfg
  let qProj : T #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size] :=
    reshape (data.fromFloatArray #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      10.0, 0.0, 0.0, 0.0,
      0.0, 10.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, -10.0, 0.0,
      0.0, 0.0, 0.0, -10.0
    ]) #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size]
  let ident : T #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size] :=
    reshape (data.fromFloatArray #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]) #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size]
  let oProj : T #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] :=
    reshape (data.fromFloatArray #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]) #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim]
  let attn : Qwen35Attention cfg := {
    q_proj := qProj
    k_proj := ident
    v_proj := ident
    o_proj := oProj
    q_norm := Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
    k_norm := Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
  }
  let x : T #[1, 1, cfg.hidden_size] :=
    reshape (data.fromFloatArray #[1.0, 2.0, 3.0, 4.0]) #[1, 1, cfg.hidden_size]
  let (cos, sin) := rotary.computeFreqsOnDevicePure 1 (Config.rotaryDim cfg) cfg.rope_theta x.device
  let out : T #[1, 1, cfg.hidden_size] := attn.forward cfg x cos sin true
  let vals ← data.tensorToFloatArray' (nn.eraseShape out)
  LeanTest.assertTrue (approxEq vals[0]! (1.0 / (1.0 + Float.exp (-10.0))) 1e-4)
    "head 0 dim 0 should use the head-local gate rows"
  LeanTest.assertTrue (approxEq vals[1]! (2.0 / (1.0 + Float.exp (-20.0))) 1e-4)
    "head 0 dim 1 should use the head-local gate rows"
  LeanTest.assertTrue (Float.abs vals[2]! < 1e-4)
    "head 1 dim 0 should be suppressed by its negative gate rows"
  LeanTest.assertTrue (Float.abs vals[3]! < 1e-4)
    "head 1 dim 1 should be suppressed by its negative gate rows"

@[test]
def testQwen35RMSNormCheckpointWeightIsZeroCentered : IO Unit := do
  let raw : T #[2] := data.fromFloatArray #[0.25, -0.5]
  let norm := Qwen35RMSNorm.fromCheckpointWeight raw
  let x : T #[1, 2] := reshape (data.fromFloatArray #[1.0, 1.0]) #[1, 2]
  let out := norm.forward2d x
  let vals ← data.tensorToFloatArray' (nn.eraseShape out)
  LeanTest.assertTrue (approxEq vals[0]! 1.25 1e-5)
    "checkpoint RMSNorm weight should be preserved directly"
  LeanTest.assertTrue (approxEq vals[1]! 0.5 1e-5)
    "Qwen3.5 RMSNorm checkpoint weights are already zero-centered"

@[test]
def testQwen35DenseGreedyStreamParity : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyDenseCfg
  let ids : T #[1, 4] := reshape (data.fromInt64Array #[7, 2, 5, 1]) #[1, 4]

  let streamedRef ← IO.mkRef (#[] : Array UInt64)
  let onStep : Qwen35ForCausalLM.StreamCallback 1 := fun _ nextTok => do
    let flat : T #[1] := reshape (data.toLong nextTok) #[1]
    let vals ← data.tensorToUInt64Array flat
    streamedRef.modify (fun xs => xs ++ vals)

  let ⟨seqStream, outStream⟩ ←
    model.generateStream tinyDenseCfg ids onStep 4 .greedy #[]
  let ⟨seqPlain, outPlain⟩ ←
    model.generate tinyDenseCfg ids 4 .greedy #[]

  LeanTest.assertEqual seqStream seqPlain
    "streaming and non-streaming cached generation should have equal output length"

  let streamFlat : T #[seqStream] := reshape (data.toLong outStream) #[seqStream]
  let plainFlat : T #[seqPlain] := reshape (data.toLong outPlain) #[seqPlain]
  let streamIds ← data.tensorToUInt64Array streamFlat
  let plainIds ← data.tensorToUInt64Array plainFlat
  LeanTest.assertEqual streamIds plainIds
    "streaming and non-streaming cached generation should produce identical token ids"

  let streamed ← streamedRef.get
  LeanTest.assertEqual streamed.size 4
    "stream callback should be invoked once per generated token"

@[test]
def testQwen35BatchedEosStopsPerRow : IO Unit := do
  let model ← batchedEosDeterministicModel
  let ids : T #[2, 1] := reshape (data.fromInt64Array #[0, 3]) #[2, 1]
  let eos := #[2]

  let ⟨seqCached, outCached⟩ ← model.generate tinyBatchedEosCfg ids 2 .greedy eos
  let ⟨seqUncached, outUncached⟩ ← model.generateUncached tinyBatchedEosCfg ids 2 .greedy eos

  LeanTest.assertEqual seqCached 3
    "cached generation should append maxNewTokens when at least one row is unfinished"
  LeanTest.assertEqual seqUncached 3
    "uncached generation should append maxNewTokens when at least one row is unfinished"

  let row0Cached ← rowTokenIds outCached 0
  let row1Cached ← rowTokenIds outCached 1
  let row0Uncached ← rowTokenIds outUncached 0
  let row1Uncached ← rowTokenIds outUncached 1

  LeanTest.assertEqual row0Cached #[0, 2, 2]
    "cached path should keep finished rows at EOS while other rows continue"
  LeanTest.assertEqual row1Cached #[3, 1, 1]
    "cached path should continue decoding unfinished rows"
  LeanTest.assertEqual row0Uncached #[0, 2, 2]
    "uncached path should keep finished rows at EOS while other rows continue"
  LeanTest.assertEqual row1Uncached #[3, 1, 1]
    "uncached path should continue decoding unfinished rows"

  LeanTest.assertEqual row0Cached row0Uncached
    "cached and uncached row 0 outputs should match in batched EOS handling"
  LeanTest.assertEqual row1Cached row1Uncached
    "cached and uncached row 1 outputs should match in batched EOS handling"

@[test]
def testQwen35MoeInitAndForward : IO Unit := do
  let model ← Qwen35ForCausalLM.init tinyMoeCfg
  let ids : T #[1, 3] := reshape (data.fromInt64Array #[5, 6, 7]) #[1, 3]
  let logits := model.forward tinyMoeCfg ids
  LeanTest.assertEqual logits.runtimeShape #[1, 3, tinyMoeCfg.vocab_size]
    "moe forward should return [batch, seq, vocab]"

@[test]
def testQwen35ConfigLoad : IO Unit := do
  let path := "/tmp/qwen35_config_test.json"
  let json :=
    "{\"vocab_size\":128,\"hidden_size\":48,\"intermediate_size\":96," ++
    "\"num_hidden_layers\":4,\"num_attention_heads\":6,\"num_key_value_heads\":2," ++
    "\"head_dim\":8,\"linear_num_key_heads\":3,\"linear_num_value_heads\":6," ++
    "\"linear_key_head_dim\":4,\"linear_value_head_dim\":4," ++
    "\"attn_output_gate\":true,\"mamba_ssm_dtype\":\"float32\"," ++
    "\"mlp_only_layers\":[1,3]," ++
    "\"moe_intermediate_size\":12,\"shared_expert_intermediate_size\":24," ++
    "\"num_experts\":8,\"num_experts_per_tok\":2," ++
    "\"layer_types\":[\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\"]," ++
    "\"rope_parameters\":{\"rope_theta\":500000.0,\"partial_rotary_factor\":0.5," ++
    "\"mrope_interleaved\":true,\"mrope_section\":[2,2,2]}," ++
    "\"mtp_num_hidden_layers\":1,\"mtp_use_dedicated_embeddings\":false}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 128 "vocab_size should load"
  LeanTest.assertEqual cfg.hidden_size 48 "hidden_size should load"
  LeanTest.assertEqual cfg.num_experts 8 "num_experts should load"
  LeanTest.assertEqual cfg.layer_types.size 4 "layer_types should parse"
  LeanTest.assertEqual cfg.rope_theta 500000.0 "rope theta should load from rope_parameters"
  LeanTest.assertEqual cfg.attn_output_gate true "attention output gate flag should load"
  LeanTest.assertEqual cfg.mamba_ssm_dtype "float32" "mamba ssm dtype should load"
  LeanTest.assertEqual cfg.mlp_only_layers #[1, 3] "mlp-only layers should load"
  LeanTest.assertEqual cfg.mrope_interleaved true "mrope interleaving flag should load"
  LeanTest.assertEqual cfg.mrope_section #[2, 2, 2] "mrope section should load"
  LeanTest.assertEqual cfg.mtp_num_hidden_layers 1 "mtp hidden layer count should load"
  LeanTest.assertEqual cfg.mtp_use_dedicated_embeddings false "mtp embedding flag should load"

@[test]
def testQwen35ConfigLoadNestedTextConfig : IO Unit := do
  let path := "/tmp/qwen35_config_test_nested.json"
  let json :=
    "{\"tie_word_embeddings\":true,\"text_config\":{" ++
    "\"vocab_size\":32000,\"hidden_size\":64,\"intermediate_size\":128," ++
    "\"num_hidden_layers\":6,\"num_attention_heads\":8,\"num_key_value_heads\":2," ++
    "\"head_dim\":8,\"linear_num_key_heads\":4,\"linear_num_value_heads\":8," ++
    "\"linear_key_head_dim\":4,\"linear_value_head_dim\":4," ++
    "\"moe_intermediate_size\":16,\"shared_expert_intermediate_size\":32," ++
    "\"num_experts\":16,\"num_experts_per_tok\":4," ++
    "\"attn_output_gate\":true," ++
    "\"eos_token_id\":151645," ++
    "\"rope_parameters\":{\"rope_theta\":10000000.0,\"partial_rotary_factor\":0.5," ++
    "\"mrope_interleaved\":true,\"mrope_section\":[11,11,10]}," ++
    "\"mtp_num_hidden_layers\":1,\"mtp_use_dedicated_embeddings\":false}}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 32000 "nested text_config vocab_size should load"
  LeanTest.assertEqual cfg.hidden_size 64 "nested text_config hidden_size should load"
  LeanTest.assertEqual cfg.tie_word_embeddings true "tie_word_embeddings should load from top level"
  LeanTest.assertEqual cfg.num_experts 16 "nested text_config num_experts should load"
  LeanTest.assertEqual cfg.eos_token_id (some 151645) "nested text_config eos token should load"
  LeanTest.assertEqual cfg.attn_output_gate true "nested text_config attention gate should load"
  LeanTest.assertEqual cfg.mrope_interleaved true "nested mrope interleaving flag should load"
  LeanTest.assertEqual cfg.mrope_section #[11, 11, 10] "nested mrope section should load"
  LeanTest.assertEqual cfg.mtp_num_hidden_layers 1 "nested mtp hidden layer count should load"
  LeanTest.assertEqual cfg.mtp_use_dedicated_embeddings false "nested mtp embedding flag should load"
