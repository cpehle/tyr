/- Tyr/Inference/KVCache.lean

  Key-Value cache for autoregressive decode generation.

  Layout: each layer holds preallocated K, V buffers of shape
  `[batch, numKvHeads, maxSeqLen, headDim]`. During generation we write the
  new token's K, V at position `cache.currentLen` via `sliceScatter` — note
  this is NOT in-place: the C++ op clones the buffer and overwrites the slice
  in the copy (`cc/src/tyr.cpp` `lean_torch_slice_scatter_along_dim`) — and
  slice the cache to `[..., currentLen+1, ...]` for the attention call.

  The attention call goes through `nn.tyrFlashAttn4d`, which the C++
  `tyr::flash_attn` operator dispatches to the native TK decode kernel
  (`tkMhaH100DecodeFwd`) when shape-eligible (BF16, qSeq=1,
  head_dim ∈ {64, 128, 256}, GQA-valid) and to PyTorch SDPA otherwise. The
  kernel handles non-multiple-of-64 `kvSeq` via a runtime tail mask, so the
  cache view never has to be padded.

  This module is the canonical home for the cache abstraction. NanoChat,
  Qwen, and any other generation-loop code should import from here. (The
  earlier copy at `Examples/NanoChat/Generator/KVCache.lean` is a thin
  re-export shim.) -/
import Tyr.Torch
import Tyr.TensorStruct

namespace torch.Generator.KVCache

open torch
open torch.data

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
  deriving Repr, Inhabited

/-- Initialize empty KV cache for given dimensions.
    `device` defaults to CPU; pass `Device.CUDA n` (or a non-default value) so
    the cache buffers live on the same device as the Q/K/V tensors that will be
    appended into them — `tyrFlashAttn4d` rejects cross-device inputs. -/
def Cache.init (numLayers batch maxSeqLen numKvHeads headDim : UInt64)
    (device : Device := Device.CPU)
    : Cache numLayers batch maxSeqLen numKvHeads headDim :=
  let emptyLayer : LayerCache batch maxSeqLen numKvHeads headDim := {
    keys := toBFloat16' (zeros #[batch, numKvHeads, maxSeqLen, headDim]
              (requires_grad := false) (device := device))
    values := toBFloat16' (zeros #[batch, numKvHeads, maxSeqLen, headDim]
              (requires_grad := false) (device := device))
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

/-- Append a single new K, V token at runtime position `pos`.

    The new K and V tensors have shape `[batch, numKvHeads, 1, headDim]` (one
    token per batch row) and get written into the layer's cache at the
    `pos`-th slot along the sequence dimension. The cache's `seqLens` is not
    touched here — call `Cache.incrementSeqLens` after appending to all layers
    of a step to advance position. -/
def LayerCache.append (lc : LayerCache batch maxSeqLen numKvHeads headDim)
    (newK newV : T #[batch, numKvHeads, 1, headDim]) (pos : UInt64)
    : LayerCache batch maxSeqLen numKvHeads headDim :=
  { keys := sliceScatter lc.keys 2 pos newK
    values := sliceScatter lc.values 2 pos newV }

/-- Slice the cache K and V tensors to cover positions `[0, validLen)`.

    Returns `(K_view, V_view)` of shape `[batch, numKvHeads, validLen, headDim]`,
    suitable to pass directly to `tkMhaH100DecodeFwd` (which expects contiguous
    `[batch, numKvHeads, kv_seq, headDim]` and handles the tail mask when
    `validLen` is not a multiple of 64). -/
def LayerCache.attentionView (lc : LayerCache batch maxSeqLen numKvHeads headDim)
    (validLen : UInt64)
    : T #[batch, numKvHeads, validLen, headDim]
      × T #[batch, numKvHeads, validLen, headDim] :=
  (slice lc.keys 2 0 validLen, slice lc.values 2 0 validLen)

/-- Append a single new K, V token to the layer cache at `cache.currentLen` and
    return the updated full cache (with that layer rewritten). Does NOT advance
    seqLens — call `incrementSeqLens` once per generated token after appending
    to every layer for the step. -/
def Cache.appendLayer (cache : Cache numLayers batch maxSeqLen numKvHeads headDim)
    (layerIdx : Nat) (newK newV : T #[batch, numKvHeads, 1, headDim])
    : Cache numLayers batch maxSeqLen numKvHeads headDim :=
  match cache.layers[layerIdx]? with
  | none => cache
  | some lc =>
    let pos := cache.currentLen
    cache.setLayer layerIdx (lc.append newK newV pos)

/-- One step of cached decode-attention for a single layer.

    1. Write `newK`, `newV` (one token per batch row) into layer `layerIdx`'s
       cache at position `cache.currentLen`.
    2. Slice the layer's cache to the resulting valid length (`currentLen + 1`).
    3. Call `tyrFlashAttn4d` on `(newQ, K_view, V_view)`.

    Returns the updated full cache and the attention output for the new query
    token. The C++ `tyr::flash_attn` dispatcher routes to `tkMhaH100DecodeFwd`
    when the shape is decode-eligible (head_dim ∈ {64, 128, 256}, qSeq==1,
    BF16, GQA-valid), and falls back to PyTorch SDPA otherwise.

    Panics if `layerIdx` is out of range — an invalid index is a programming
    error, not a condition to silently absorb.

    Does NOT advance `seqLens` — call `cache.incrementSeqLens` once per
    generated token after all layers have been processed for the step. -/
def Cache.attendLayer
    {numQHeads : UInt64}
    (cache : Cache numLayers batch maxSeqLen numKvHeads headDim)
    (layerIdx : Nat)
    (newQ : T #[batch, numQHeads, 1, headDim])
    (newK newV : T #[batch, numKvHeads, 1, headDim])
    (enableGqa : Bool := false)
    : Cache numLayers batch maxSeqLen numKvHeads headDim
      × T #[batch, numQHeads, 1, headDim] :=
  let cache' := cache.appendLayer layerIdx newK newV
  let validLen := cache.currentLen + 1
  match cache'.layers[layerIdx]? with
  | none =>
    panic! s!"Cache.attendLayer: layerIdx {layerIdx} out of range (cache has {cache'.layers.size} layers)"
  | some lc =>
    let (kView, vView) := lc.attentionView validLen
    let output := nn.tyrFlashAttn4d newQ kView vView none 0.0 false none enableGqa
    (cache', output)

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
