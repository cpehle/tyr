/-
  Tyr/GPU/Kernels/Examples.lean

  Example GPU kernels demonstrating the native Lean4 DSL syntax:
  - @[gpu_kernel] attribute for kernel registration
  - for _ in [lo:hi] loop syntax
  - Type-safe tile operations
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Codegen.ArchConfig

namespace Tyr.GPU.Kernels.Examples

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Example 1: Simple GEMM with for loop syntax -/

/-- Simple GEMM using the new for loop syntax -/
@[gpu_kernel .SM90]
def simpleGemm : KernelM Unit := do
  comment "=== Simple GEMM ==="

  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  -- Using forLoop for iteration
  forLoop 0 8 do
    load a aShared
    load b bShared
    mma c a b c
    sync

/-- Build simple GEMM kernel -/
def simpleGemmKernel : Kernel :=
  buildKernelM "simple_gemm" .SM90 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .Float32, isPointer := true },
    { name := "M", dtype := .Float32, isPointer := false },
    { name := "N", dtype := .Float32, isPointer := false },
    { name := "K", dtype := .Float32, isPointer := false }
  ] simpleGemm

/-! ## Example 2: FlashAttention with attribute -/

/-- FlashAttention forward kernel -/
@[gpu_kernel .SM90]
def flashAttnFwd : KernelM Unit := do
  comment "=== FlashAttention Forward ==="

  -- Register tiles
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Row-wise softmax tracking
  let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  -- Load Q (long-resident)
  load q qShared

  -- Main loop over K, V blocks
  forLoop 0 4 do
    -- Load K, V
    load k kShared
    load v vShared

    -- Attention scores: S = Q @ K^T
    mmaT s q k s

    -- Causal mask
    makeCausal s s (some (-1e10))

    -- Online softmax
    rowMaxAccum rowMax s rowMax
    subCol s s rowMax
    exp s s
    rowSumAccum rowSum s rowSum

    -- Convert and accumulate output
    let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert p s
    mma o p v o

    sync

  -- Final normalization
  divCol o o rowSum

/-- Build FlashAttention kernel -/
def flashAttnFwdKernel : Kernel :=
  buildKernelM "flash_attn_fwd" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "seq_len", dtype := .Float32, isPointer := false }
  ] flashAttnFwd

/-! ## Example 3: Ampere (SM80) kernel -/

/-- Simple kernel targeting Ampere A100 -/
@[gpu_kernel .SM80]
def ampereGemm : KernelM Unit := do
  comment "=== Ampere GEMM (SM80) ==="

  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  -- SM80 uses smaller pipeline stages
  forLoop 0 4 do
    load a aShared
    load b bShared
    mma c a b c
    sync

def ampereGemmKernel : Kernel :=
  buildKernelM "ampere_gemm" .SM80 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .Float32, isPointer := true }
  ] ampereGemm

/-! ## Example 4: Blackwell (SM100) kernel -/

/-- Kernel targeting Blackwell B200 -/
@[gpu_kernel .SM100]
def blackwellGemm : KernelM Unit := do
  comment "=== Blackwell GEMM (SM100) ==="

  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  -- SM100 can handle deeper pipelines
  forLoop 0 8 do
    load a aShared
    load b bShared
    mma c a b c
    sync

def blackwellGemmKernel : Kernel :=
  buildKernelM "blackwell_gemm" .SM100 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .Float32, isPointer := true }
  ] blackwellGemm

/-! ## Example 5: LayerNorm with loop syntax -/

/-- LayerNorm kernel -/
@[gpu_kernel .SM90]
def layerNorm : KernelM Unit := do
  comment "=== LayerNorm ==="

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let xf : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let temp : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  let mean : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let var : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let weight : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let bias : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let weightShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let biasShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  -- Load weight and bias (long-resident)
  load weight weightShared
  load bias biasShared

  forLoop 0 16 do
    -- Load input
    load x xShared

    -- Convert to float32
    convert xf x

    -- Compute mean
    rowSum mean xf

    -- Subtract mean
    subCol temp xf mean

    -- Compute variance
    mul xf temp temp
    rowSum var xf

    -- Scale and shift
    let weightF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let biasF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    convert weightF weight
    convert biasF bias
    mul temp temp weightF
    add temp temp biasF

    -- Convert back and store
    convert x temp
    store xShared x

    sync

def layerNormKernel : Kernel :=
  buildKernelM "layer_norm" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "weight_ptr", dtype := .BFloat16, isPointer := true },
    { name := "bias_ptr", dtype := .BFloat16, isPointer := true },
    { name := "out_ptr", dtype := .BFloat16, isPointer := true },
    { name := "hidden_dim", dtype := .Float32, isPointer := false }
  ] layerNorm


/-! ## Generated Code Output -/

-- Print generated kernels
#eval IO.println "=== Simple GEMM ===" *> IO.println (generateKernel simpleGemmKernel)
#eval IO.println "\n=== FlashAttention ===" *> IO.println (generateKernel flashAttnFwdKernel)
#eval IO.println "\n=== Ampere GEMM (SM80) ===" *> IO.println (generateKernel ampereGemmKernel)
#eval IO.println "\n=== Blackwell GEMM (SM100) ===" *> IO.println (generateKernel blackwellGemmKernel)
#eval IO.println "\n=== LayerNorm ===" *> IO.println (generateKernel layerNormKernel)

end Tyr.GPU.Kernels.Examples
