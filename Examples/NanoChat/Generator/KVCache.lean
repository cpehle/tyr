/-
  Tyr/Generator/KVCache.lean

  Key-Value cache for efficient autoregressive generation.

  During generation, we cache the keys and values computed for each position
  to avoid recomputing them at every step. This provides 10-100x speedup
  for autoregressive generation.

  Based on nanochat's engine.py KV cache implementation.
-/
import Tyr.Torch
import Tyr.TensorStruct

namespace torch.Generator.KVCache

open torch

/-- KV cache for a single attention layer -/
structure LayerCache (batch maxSeqLen numKvHeads headDim : UInt64) where
  /-- Cached keys: [batch, num_kv_heads, max_seq_len, head_dim] -/
  keys : T #[batch, numKvHeads, maxSeqLen, headDim]
  /-- Cached values: [batch, num_kv_heads, max_seq_len, head_dim] -/
  values : T #[batch, numKvHeads, maxSeqLen, headDim]
  deriving Repr

/-- Full KV cache for all layers -/
structure Cache (numLayers batch maxSeqLen numKvHeads headDim : UInt64) where
  /-- Per-layer KV caches -/
  layers : Array (LayerCache batch maxSeqLen numKvHeads headDim)
  /-- Current sequence length for each batch element -/
  seqLens : Array UInt64
  /-- Maximum sequence length this cache can hold -/
  maxLen : UInt64
  deriving Repr

/-- Initialize empty KV cache for given dimensions -/
def Cache.init (numLayers batch maxSeqLen numKvHeads headDim : UInt64)
    : Cache numLayers batch maxSeqLen numKvHeads headDim :=
  let emptyLayer : LayerCache batch maxSeqLen numKvHeads headDim := {
    keys := zeros #[batch, numKvHeads, maxSeqLen, headDim]
    values := zeros #[batch, numKvHeads, maxSeqLen, headDim]
  }
  let layers := Array.mk (List.replicate numLayers.toNat emptyLayer)
  let seqLens := Array.mk (List.replicate batch.toNat (0 : UInt64))
  { layers, seqLens, maxLen := maxSeqLen }

/-- Increment sequence lengths for all batch elements -/
def Cache.incrementSeqLens (cache : Cache numLayers batch maxSeqLen numKvHeads headDim)
    : Cache numLayers batch maxSeqLen numKvHeads headDim :=
  { cache with seqLens := cache.seqLens.map (· + 1) }

/-- Get current sequence length (assumes all batch elements have same length) -/
def Cache.currentLen (cache : Cache numLayers batch maxSeqLen numKvHeads headDim) : UInt64 :=
  cache.seqLens[0]?.getD 0

/-- Check if cache has room for more tokens -/
def Cache.hasRoom (cache : Cache numLayers batch maxSeqLen numKvHeads headDim) : Bool :=
  cache.currentLen < cache.maxLen

/-- Get layer cache at index -/
def Cache.getLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim) (idx : Nat)
    : Option (LayerCache batch maxSeqLen numKvHeads headDim) :=
  cache.layers[idx]?

/-- Set layer cache at index -/
def Cache.setLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim)
    (idx : Nat) (layer : LayerCache batch maxSeqLen numKvHeads headDim)
    : Cache numLayers batch maxSeqLen numKvHeads headDim :=
  if idx < cache.layers.size then
    { cache with layers := cache.layers.set! idx layer }
  else
    cache

/-- TensorStruct instance for LayerCache -/
instance {batch maxSeqLen numKvHeads headDim : UInt64}
    : TensorStruct (LayerCache batch maxSeqLen numKvHeads headDim) where
  map f c := { keys := f c.keys, values := f c.values }
  mapM f c := do pure { keys := ← f c.keys, values := ← f c.values }
  zipWith f c1 c2 := { keys := f c1.keys c2.keys, values := f c1.values c2.values }
  fold f init c := f c.values (f c.keys init)

/-- TensorStruct instance for Cache -/
instance {numLayers batch maxSeqLen numKvHeads headDim : UInt64}
    : TensorStruct (Cache numLayers batch maxSeqLen numKvHeads headDim) where
  map f c := { c with layers := c.layers.map (TensorStruct.map f) }
  mapM f c := do
    let layers ← c.layers.mapM (TensorStruct.mapM f)
    pure { c with layers }
  zipWith f c1 c2 := { c1 with
    layers := Array.zipWith (TensorStruct.zipWith f) c1.layers c2.layers
  }
  fold f init c := c.layers.foldl (fun acc l => TensorStruct.fold f acc l) init

end torch.Generator.KVCache
