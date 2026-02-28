/-
  Tests/TestSafeTensorsTypeProviderErrors.lean

  Compile-time failure-path coverage for `safetensors_type_provider`.
-/
import Tyr.SafeTensors

/--
error: safetensors_type_provider: non-contiguous numeric path segment under 'layers'. expected index 1, found 2.
-/
#guard_msgs in
safetensors_type_provider "Tests/fixtures/safetensors/provider_errors/noncontiguous.safetensors" as BadNonContiguous

/--
error: safetensors_type_provider: non-uniform indexed subtree under 'layers'. All numeric siblings must share the same schema to form an indexed collection.
-/
#guard_msgs in
safetensors_type_provider "Tests/fixtures/safetensors/provider_errors/nonuniform.safetensors" as BadNonUniform
