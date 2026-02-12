/- Minimal ThunderKittens-style FlashAttention forward kernel for 64x64 tiles. -/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- FlashAttention forward for a single 64x64 K/V tile.
    Shapes are expected to be [B, H, 64, 64], one query tile per block row. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 1
  let scale : Float := 0.125 -- 1 / sqrt(64)

  let coord ← blockCoord2D

  let q ← allocRT .BFloat16 tileSize tileSize
  let k ← allocRT .BFloat16 tileSize tileSize
  let v ← allocRT .BFloat16 tileSize tileSize .Col
  let o ← zeroRT .Float32 tileSize tileSize

  let softmaxState ← allocSoftmaxState .Float32 tileSize

  let qShared ← allocST .BFloat16 tileSize tileSize
  let kShared ← allocST .BFloat16 tileSize tileSize
  let vShared ← allocST .BFloat16 tileSize tileSize .Col
  let oShared ← allocST .BFloat16 tileSize tileSize

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT .Float32 tileSize tileSize
    let p ← allocRT .BFloat16 tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    makeCausal s s (some (-1e10))
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

end Tyr.GPU.Kernels
