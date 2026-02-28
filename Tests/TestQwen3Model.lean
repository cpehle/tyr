import Tyr
import Tyr.Model.Qwen3
import LeanTest

open torch
open torch.qwen3

private def tinyCfg : Config := {
  vocab_size := 64
  hidden_size := 32
  intermediate_size := 64
  num_hidden_layers := 2
  num_attention_heads := 4
  num_key_value_heads := 2
  head_dim := 8
  rope_theta := 10000.0
  rms_norm_eps := 1e-6
  max_position_embeddings := 128
}

@[test]
def testQwen3InitAndForward : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : T #[1, 4] := reshape (data.fromInt64Array #[1, 2, 3, 4]) #[1, 4]
  let logits := model.forward ids
  LeanTest.assertEqual logits.runtimeShape #[1, 4, tinyCfg.vocab_size]
    "forward should return [batch, seq, vocab]"

@[test]
def testQwen3GreedyCachedUncachedParity : IO Unit := do
  let model ← Qwen3ForCausalLM.init tinyCfg
  let ids : T #[1, 3] := reshape (data.fromInt64Array #[3, 2, 1]) #[1, 3]

  let ⟨outSeqCached, outCached⟩ ← model.generateGreedy ids 4 #[]
  let ⟨outSeqUncached, outUncached⟩ ← model.generateGreedyUncached ids 4 #[]

  LeanTest.assertEqual outSeqCached (3 + 4)
    "cached generation should append maxNewTokens"
  LeanTest.assertEqual outSeqUncached (3 + 4)
    "uncached generation should append maxNewTokens"

  let cachedFlat : T #[outSeqCached] := reshape (data.toLong outCached) #[outSeqCached]
  let uncachedFlat : T #[outSeqUncached] := reshape (data.toLong outUncached) #[outSeqUncached]
  let cachedIds ← data.tensorToUInt64Array cachedFlat
  let uncachedIds ← data.tensorToUInt64Array uncachedFlat

  LeanTest.assertEqual cachedIds uncachedIds
    "cached and uncached greedy decoding should produce identical tokens"

@[test]
def testQwen3ConfigLoad : IO Unit := do
  let path := "/tmp/qwen3_config_test.json"
  let json :=
    "{\"vocab_size\":128,\"hidden_size\":48,\"intermediate_size\":96," ++
    "\"num_hidden_layers\":3,\"num_attention_heads\":6,\"num_key_value_heads\":2," ++
    "\"rope_theta\":500000.0,\"max_position_embeddings\":4096}"
  IO.FS.writeFile path json
  let cfg ← Config.loadFromFile path
  LeanTest.assertEqual cfg.vocab_size 128 "vocab_size should load from json"
  LeanTest.assertEqual cfg.hidden_size 48 "hidden_size should load from json"
  LeanTest.assertEqual cfg.num_attention_heads 6 "num_attention_heads should load from json"
  LeanTest.assertEqual cfg.head_dim 8 "head_dim should default to hidden_size / num_attention_heads"
