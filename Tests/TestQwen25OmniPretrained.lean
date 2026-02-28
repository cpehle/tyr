import Tyr.Model.Qwen25Omni
import LeanTest

open torch.qwen25omni

private def ensureDir (path : String) : IO Unit :=
  IO.FS.createDirAll ⟨path⟩

@[test]
def testQwen25OmniHubShardFilesFromIndex : IO Unit := do
  let dir := "/tmp/qwen25omni_hub_index_test"
  ensureDir dir
  let indexPath := s!"{dir}/model.safetensors.index.json"
  let json :=
    "{\"metadata\":{},\"weight_map\":{" ++
    "\"thinker.model.embed_tokens.weight\":\"model-00001-of-00002.safetensors\"," ++
    "\"thinker.model.layers.0.self_attn.q_proj.weight\":\"model-00002-of-00002.safetensors\"," ++
    "\"thinker.model.layers.0.self_attn.k_proj.weight\":\"model-00001-of-00002.safetensors\"}}"
  IO.FS.writeFile indexPath json

  let shards ← hub.shardFilesFromIndexFile indexPath
  LeanTest.assertEqual shards.size 2 "index parser should return unique shard names"
  LeanTest.assertTrue (shards.contains "model-00001-of-00002.safetensors")
    "first shard should be present"
  LeanTest.assertTrue (shards.contains "model-00002-of-00002.safetensors")
    "second shard should be present"

@[test]
def testQwen25OmniHubDetectWeightLayout : IO Unit := do
  let shardedDir := "/tmp/qwen25omni_hub_layout_sharded"
  ensureDir shardedDir
  IO.FS.writeFile s!"{shardedDir}/model.safetensors.index.json" "{\"metadata\":{},\"weight_map\":{}}"
  let isSharded ← hub.detectWeightLayout shardedDir
  LeanTest.assertEqual isSharded true "index file should detect sharded layout"

  let singleDir := "/tmp/qwen25omni_hub_layout_single"
  ensureDir singleDir
  IO.FS.writeFile s!"{singleDir}/model.safetensors" ""
  let isShardedSingle ← hub.detectWeightLayout singleDir
  LeanTest.assertEqual isShardedSingle false "single safetensors file should detect single layout"

@[test]
def testQwen25OmniHubResolveLocalPretrainedDir : IO Unit := do
  let localDir := "/tmp/qwen25omni_hub_local_dir"
  ensureDir localDir
  IO.FS.writeFile s!"{localDir}/config.json" "{}"
  IO.FS.writeFile s!"{localDir}/model.safetensors" ""

  let resolved ← hub.resolvePretrainedDir localDir
  LeanTest.assertEqual resolved localDir "existing local model directory should resolve without network"

@[test]
def testQwen25OmniCollectionListCoverage : IO Unit := do
  LeanTest.assertTrue (hub.isQwen25OmniCollectionRepoId "Qwen/Qwen2.5-Omni-3B")
    "collection list should include 3B"
  LeanTest.assertTrue (hub.isQwen25OmniCollectionRepoId "Qwen/Qwen2.5-Omni-7B")
    "collection list should include 7B"

