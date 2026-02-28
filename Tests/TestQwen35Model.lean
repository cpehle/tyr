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
    "\"moe_intermediate_size\":12,\"shared_expert_intermediate_size\":24," ++
    "\"num_experts\":8,\"num_experts_per_tok\":2," ++
    "\"layer_types\":[\"linear_attention\",\"full_attention\",\"linear_attention\",\"full_attention\"]," ++
    "\"rope_parameters\":{\"rope_theta\":500000.0,\"partial_rotary_factor\":0.5}}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 128 "vocab_size should load"
  LeanTest.assertEqual cfg.hidden_size 48 "hidden_size should load"
  LeanTest.assertEqual cfg.num_experts 8 "num_experts should load"
  LeanTest.assertEqual cfg.layer_types.size 4 "layer_types should parse"
  LeanTest.assertEqual cfg.rope_theta 500000.0 "rope theta should load from rope_parameters"

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
    "\"eos_token_id\":151645," ++
    "\"rope_parameters\":{\"rope_theta\":10000000.0,\"partial_rotary_factor\":0.5}}}"
  IO.FS.writeFile path json

  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 32000 "nested text_config vocab_size should load"
  LeanTest.assertEqual cfg.hidden_size 64 "nested text_config hidden_size should load"
  LeanTest.assertEqual cfg.tie_word_embeddings true "tie_word_embeddings should load from top level"
  LeanTest.assertEqual cfg.num_experts 16 "nested text_config num_experts should load"
  LeanTest.assertEqual cfg.eos_token_id (some 151645) "nested text_config eos token should load"
