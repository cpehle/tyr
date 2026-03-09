import Tyr.AD.Frontend.Signature

/-!
# Tyr.AD.Frontend.Companion

Small structured companion helpers over `FrontendADSignature`.

These helpers intentionally stay backend-agnostic: a frontend/backend pair can
compute flat leaf outputs/cotangents however it likes, and this layer rebuilds
ordinary `TensorStruct`-shaped Lean values at the boundary.
-/

namespace Tyr.AD.Frontend

open torch

/-- Structured result for scalar-loss gradient style APIs. -/
structure StructuredGradResult (Params : Type) where
  params : Params

/-- Structured result for pullback / VJP style APIs. -/
structure StructuredVJPResult (Params Inputs : Type) where
  params : Params
  inputs : Inputs

/-- Structured output together with a structured pullback. -/
structure StructuredPullback (Params Inputs Outputs : Type) where
  output : Outputs
  pullback : Outputs → Except String (StructuredVJPResult Params Inputs)

namespace FrontendADSignature

private def requireSingleBoundary (label : String) (count : Nat) : Except String Unit := do
  if count != 1 then
    throw <| s!"Structured frontend companion expects exactly one {label} boundary, " ++
      s!"signature has {count}."

private def flattenSingleOutputCotangent
    {Outputs : Type}
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (sig : FrontendADSignature)
    (cotangent : Outputs) :
    Except String (Array TensorLeafValue) := do
  requireSingleBoundary "output" sig.outputs.size
  let some boundary := sig.outputs[0]? | throw "Internal structured output boundary lookup failure."
  boundary.flattenValue cotangent

/--
Rebuild the single parameter boundary from a flat invar cotangent buffer.

This is the first structured `grad`-style helper on top of the flat frontend
leaf order.
-/
def rebuildSingleParamGrad
    {Params : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    (sig : FrontendADSignature)
    (paramsTemplate : Params)
    (invarCotangentLeaves : Array TensorLeafValue) :
    Except String (StructuredGradResult Params) := do
  requireSingleBoundary "param" sig.params.size
  pure { params := ← sig.rebuildParamValue 0 paramsTemplate invarCotangentLeaves }

/--
Rebuild single structured parameter/input cotangents from a flat invar
cotangent buffer.
-/
def rebuildSingleVJPResult
    {Params Inputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    (sig : FrontendADSignature)
    (paramsTemplate : Params)
    (inputsTemplate : Inputs)
    (invarCotangentLeaves : Array TensorLeafValue) :
    Except String (StructuredVJPResult Params Inputs) := do
  requireSingleBoundary "param" sig.params.size
  requireSingleBoundary "input" sig.inputs.size
  pure {
    params := ← sig.rebuildParamValue 0 paramsTemplate invarCotangentLeaves
    inputs := ← sig.rebuildInputValue 0 inputsTemplate invarCotangentLeaves
  }

/--
Wrap a flat single-output pullback into a structured Lean-facing pullback.

The backend callback consumes flattened output cotangent leaves and returns flat
invar cotangent leaves. This helper rebuilds both the primal output and the
pullback result using the frontend signature.
-/
def buildSingleStructuredPullback
    {Params Inputs Outputs : Type}
    [ToTensorStructSchema Params]
    [TensorStructFlatten Params]
    [ToTensorStructSchema Inputs]
    [TensorStructFlatten Inputs]
    [ToTensorStructSchema Outputs]
    [TensorStructFlatten Outputs]
    (sig : FrontendADSignature)
    (paramsTemplate : Params)
    (inputsTemplate : Inputs)
    (outputTemplate : Outputs)
    (flatOutputLeaves : Array TensorLeafValue)
    (flatPullback : Array TensorLeafValue → Except String (Array TensorLeafValue)) :
    Except String (StructuredPullback Params Inputs Outputs) := do
  requireSingleBoundary "param" sig.params.size
  requireSingleBoundary "input" sig.inputs.size
  requireSingleBoundary "output" sig.outputs.size
  let output ← sig.rebuildOutputValue 0 outputTemplate flatOutputLeaves
  pure {
    output := output
    pullback := fun outputCotangent => do
      let outputCotangentLeaves ← flattenSingleOutputCotangent sig outputCotangent
      let invarCotangentLeaves ← flatPullback outputCotangentLeaves
      sig.rebuildSingleVJPResult paramsTemplate inputsTemplate invarCotangentLeaves
  }

end FrontendADSignature

end Tyr.AD.Frontend
