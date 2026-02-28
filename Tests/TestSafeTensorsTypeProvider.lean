/-
  Tests/TestSafeTensorsTypeProvider.lean

  Coverage for:
  - runtime safetensors schema introspection
  - elaboration-time `safetensors_type_provider`
-/
import Tyr.SafeTensors
import LeanTest

open torch

safetensors_type_provider "Tests/fixtures/safetensors/single.safetensors" as SingleSafe
safetensors_type_provider "Tests/fixtures/safetensors/sharded" as ShardedSafe
safetensors_type_provider "Tests/fixtures/safetensors/indexed.safetensors" as IndexedSafe
safetensors_type_provider "Tests/fixtures/safetensors/indexed_dir" as IndexedDirSafe

@[test]
def testSafeTensorsIntrospectionSingle : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/single.safetensors"
  LeanTest.assertEqual schema.sourceIsDirectory false "single source should not be a directory"
  LeanTest.assertEqual schema.tensors.size 1 "single fixture should have one tensor"

  let some tensor := schema.find? "linear.weight"
    | throw <| IO.userError "expected tensor 'linear.weight'"
  LeanTest.assertEqual tensor.dtype "F32" "dtype should be parsed from safetensors header"
  LeanTest.assertTrue (tensor.shape == #[2, 3]) "shape should match safetensors header"
  LeanTest.assertEqual tensor.sourceFile "" "single-file introspection should use empty sourceFile"

@[test]
def testSafeTensorsIntrospectionSharded : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/sharded"
  LeanTest.assertEqual schema.sourceIsDirectory true "sharded source should be a directory"
  LeanTest.assertEqual schema.tensors.size 2 "sharded fixture should have two tensors"

  let some embed := schema.find? "embed.weight"
    | throw <| IO.userError "expected tensor 'embed.weight'"
  LeanTest.assertEqual embed.sourceFile "part1.safetensors" "embed tensor should map to part1 shard"
  LeanTest.assertTrue (embed.shape == #[2, 2]) "embed tensor shape should match header"

  let some bias := schema.find? "proj.bias"
    | throw <| IO.userError "expected tensor 'proj.bias'"
  LeanTest.assertEqual bias.sourceFile "part2.safetensors" "bias tensor should map to part2 shard"
  LeanTest.assertTrue (bias.shape == #[3]) "bias tensor shape should match header"

@[test]
def testSafeTensorsIntrospectionShardedIndexJson : IO Unit := do
  let schema ← safetensors.introspect "Tests/fixtures/safetensors/indexed_dir"
  LeanTest.assertEqual schema.sourceIsDirectory true "indexed sharded source should be a directory"
  LeanTest.assertEqual schema.tensors.size 2 "weight_map should define exactly two tensors"

  let some embed := schema.find? "embed.weight"
    | throw <| IO.userError "expected tensor 'embed.weight'"
  LeanTest.assertEqual embed.sourceFile "part1.safetensors" "embed tensor should map through index json"

  let some bias := schema.find? "proj.bias"
    | throw <| IO.userError "expected tensor 'proj.bias'"
  LeanTest.assertEqual bias.sourceFile "part2.safetensors" "proj.bias tensor should map through index json"

  let unmapped := schema.find? "linear.weight"
  LeanTest.assertTrue unmapped.isNone "tensors from unmapped shard files should not be exposed"

@[test]
def testSafeTensorsTypeProviderSingle : IO Unit := do
  LeanTest.assertEqual SingleSafe.sourceIsDirectory false "provider should detect single-file source"
  LeanTest.assertEqual SingleSafe.tensorCount 1 "provider tensor count should match fixture"
  LeanTest.assertTrue (SingleSafe.hasTensor "linear.weight") "provider should include linear.weight"
  LeanTest.assertEqual SingleSafe.fieldToTensorName [("linear_weight", "linear.weight")]
    "field map should expose generated field names for aggregate record"

  let t ← SingleSafe.load_linear_weight
  LeanTest.assertTrue (t.runtimeShape == #[2, 3]) "typed loader should enforce generated shape"
  LeanTest.assertEqual SingleSafe.linear_weightSpec.dtype "F32" "generated spec should retain dtype"

  let weights ← SingleSafe.loadAll
  LeanTest.assertTrue (weights.linear.weight.runtimeShape == #[2, 3])
    "hierarchical aggregate typed record should expose nested typed tensor fields"

@[test]
def testSafeTensorsTypeProviderSharded : IO Unit := do
  LeanTest.assertEqual ShardedSafe.sourceIsDirectory true "provider should detect sharded directory source"
  LeanTest.assertEqual ShardedSafe.tensorCount 2 "provider tensor count should match sharded fixture"
  LeanTest.assertTrue (ShardedSafe.hasTensor "embed.weight") "provider should include embed.weight"
  LeanTest.assertTrue (ShardedSafe.hasTensor "proj.bias") "provider should include proj.bias"

  let e ← ShardedSafe.load_embed_weight
  LeanTest.assertTrue (e.runtimeShape == #[2, 2]) "embed loader should return generated typed shape"

  let b ← ShardedSafe.load_proj_bias
  LeanTest.assertTrue (b.runtimeShape == #[3]) "bias loader should return generated typed shape"
  LeanTest.assertEqual ShardedSafe.proj_biasSpec.sourceFile "part2.safetensors"
    "generated tensor spec should expose shard source file"

  let weights ← ShardedSafe.loadAll
  LeanTest.assertTrue (weights.embed.weight.runtimeShape == #[2, 2])
    "hierarchical aggregate typed record should expose nested embed tensor"
  LeanTest.assertTrue (weights.proj.bias.runtimeShape == #[3])
    "hierarchical aggregate typed record should expose nested bias tensor"

@[test]
def testSafeTensorsTypeProviderIndexedHierarchy : IO Unit := do
  LeanTest.assertEqual IndexedSafe.tensorCount 2 "indexed fixture should have two tensors"
  let weights ← IndexedSafe.loadAll
  LeanTest.assertEqual weights.layers.size 2 "numeric path segments should produce an indexed collection"
  LeanTest.assertTrue (weights.layers[0]!.weight.runtimeShape == #[2])
    "first indexed subtree should expose typed nested tensor"
  LeanTest.assertTrue (weights.layers[1]!.weight.runtimeShape == #[2])
    "second indexed subtree should expose typed nested tensor"

@[test]
def testSafeTensorsTypeProviderShardedIndexJson : IO Unit := do
  LeanTest.assertEqual IndexedDirSafe.sourceIsDirectory true "provider should detect sharded index directory"
  LeanTest.assertEqual IndexedDirSafe.tensorCount 2 "provider should expose only index-mapped tensors"
  LeanTest.assertTrue (IndexedDirSafe.hasTensor "embed.weight")
    "index-mapped tensor should exist"
  LeanTest.assertTrue (!(IndexedDirSafe.hasTensor "linear.weight"))
    "tensor from unmapped shard should not be generated"

  let weights ← IndexedDirSafe.loadAll
  LeanTest.assertTrue (weights.embed.weight.runtimeShape == #[2, 2])
    "index-backed loadAll should load embed tensor from mapped shard"
  LeanTest.assertTrue (weights.proj.bias.runtimeShape == #[3])
    "index-backed loadAll should load proj bias tensor from mapped shard"
