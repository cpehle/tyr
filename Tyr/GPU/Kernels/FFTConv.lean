/-
  Tyr/GPU/Kernels/FFTConv.lean

  FFT-based convolution kernel implementation.
  Based on ThunderKittens fftconv_pc.cu patterns.

  Key features:
  - FFT via matrix multiplication (not traditional Cooley-Tukey)
  - Complex number handling via CRT (zero-cost abstraction over real/imag RTs)
  - All matrices in bf16, MMA accumulator in f32, copy back to bf16
  - Matches ThunderKittens semantics exactly

  Type conventions (matching ThunderKittens):
  - crt_fl<M, N> → CRT Float32 M N (accumulator)
  - crt_bf<M, N> → CRT BFloat16 M N (working tiles)
  - cst_bf<M, N> → CST BFloat16 M N (shared memory)
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

namespace Tyr.GPU.Kernels.FFTConv

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## FFT via Matrix Multiplication

ThunderKittens FFTConv algorithm (from fftconv_pc.cu):

1. F @ X        -- Left multiply by DFT matrix (X is real input)
2. *= tw        -- Apply twiddle factors (elementwise complex)
3. Y @ F        -- Right multiply by DFT matrix
4. *= kf        -- Apply filter in frequency domain (elementwise complex)
5. Y @ F_inv    -- Right multiply by inverse DFT
6. *= tw_inv    -- Apply inverse twiddle
7. F_inv @ Y    -- Left multiply by inverse DFT
8. Output real  -- Take real part of result

All matrices (F, F_inv, tw, tw_inv, kf) are complex bf16.
Input and output are real bf16.
-/

/-- FFT Convolution forward pass - matches ThunderKittens fftconv_pc.cu -/
@[gpu_kernel .SM90]
def fftConvFwd : KernelM Unit := do
  comment "=== FFT Convolution Forward (ThunderKittens semantics) ==="

  -- Input tile (real only, bf16) - corresponds to args.input.x
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- MMA accumulator (complex f32) - corresponds to crt_fl<16, 64> mma_reg
  let mmaReg : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64

  -- Working tiles (complex bf16) - corresponds to crt_bf<16, 64> accum, tmp
  let accum : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tmp : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  -- Output tile (real only, bf16)
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory (complex bf16) - corresponds to scratch_block
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let fShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64       -- DFT matrix
  let fInvShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64   -- Inverse DFT
  let twShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64     -- Twiddle factors
  let twInvShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64  -- Inverse twiddle (pre-transposed)
  let kfShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64     -- Filter in freq domain
  let tmpShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64    -- Scratch for AtB
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  -- Register tiles for DFT matrices
  let f : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let fInv : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tw : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let twInv : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let kf : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  comment "Load persistent DFT matrices and factors"
  loadComplex f fShared
  loadComplex fInv fInvShared
  loadComplex tw twShared
  loadComplex twInv twInvShared

  comment "Process batches"
  forLoop 0 16 do
    comment "Load input (real only)"
    load x xShared

    comment "Load filter for this head"
    loadComplex kf kfShared

    comment "Step 1: F @ X (left multiply)"
    comment "mma_reg.real = f.real @ x, mma_reg.imag = f.imag @ x"
    -- Since x is real, the complex multiply simplifies:
    -- (f_r + i*f_i) @ x_r = f_r @ x_r + i*(f_i @ x_r)
    mma mmaReg.real f.real x (← zeroRT .Float32 64 64)
    mma mmaReg.imag f.imag x (← zeroRT .Float32 64 64)

    comment "Copy f32 accumulator to bf16 working tile"
    convert accum.real mmaReg.real
    convert accum.imag mmaReg.imag

    comment "Step 2: Apply twiddle (elementwise complex multiply)"
    -- accum = accum * tw (matches warp::mul(accum, accum, tmp))
    complexMul accum accum tw

    sync

    comment "Step 3: Y @ F (right multiply)"
    -- ThunderKittens uses: mm_AB(mma_reg, accum, f) with complex semantics
    complexMma mmaReg accum f (← zeroCRT .Float32 64 64)
    convert accum.real mmaReg.real
    convert accum.imag mmaReg.imag

    comment "Step 4: Apply filter (elementwise complex multiply)"
    -- accum = accum * kf (matches warp::mul(accum, accum, tmp))
    complexMul accum accum kf

    comment "Step 5: Y @ F_inv (right multiply by inverse DFT)"
    complexMma mmaReg accum fInv (← zeroCRT .Float32 64 64)
    convert accum.real mmaReg.real
    convert accum.imag mmaReg.imag

    comment "Step 6: Apply inverse twiddle (elementwise complex multiply)"
    -- accum = accum * twInv (matches warp::mul(accum, accum, tmp))
    complexMul accum accum twInv

    comment "Step 7: Store to shared memory for F_inv @ Y (A^T @ B pattern)"
    storeComplex tmpShared accum
    sync

    comment "Step 8: F_inv @ Y (left multiply by inverse DFT)"
    loadComplex tmp tmpShared
    complexMma mmaReg fInv tmp (← zeroCRT .Float32 64 64)

    comment "Step 9: Output real part"
    convert out mmaReg.real
    store outShared out

    sync

/-- Build FFT Convolution forward kernel -/
def fftConvFwdKernel : Kernel :=
  buildKernelM "fftconv_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "kf_real", dtype := .BFloat16, isPointer := true },
    { name := "kf_imag", dtype := .BFloat16, isPointer := true },
    { name := "f_real", dtype := .BFloat16, isPointer := true },
    { name := "f_imag", dtype := .BFloat16, isPointer := true },
    { name := "finv_real", dtype := .BFloat16, isPointer := true },
    { name := "finv_imag", dtype := .BFloat16, isPointer := true },
    { name := "tw_real", dtype := .BFloat16, isPointer := true },
    { name := "tw_imag", dtype := .BFloat16, isPointer := true },
    { name := "twinv_real", dtype := .BFloat16, isPointer := true },
    { name := "twinv_imag", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "batch_size", dtype := .Float32, isPointer := false },
    { name := "num_heads", dtype := .Float32, isPointer := false }
  ] fftConvFwd

/-! ## Persistent Cache Variant

This variant keeps DFT matrices resident in shared memory
across multiple batches, only reloading the per-head filter.
-/

/-- FFT Convolution with persistent DFT matrices -/
@[gpu_kernel .SM90]
def fftConvPersistentFwd : KernelM Unit := do
  comment "=== FFT Convolution (Persistent Cache) ==="

  -- Working tiles
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let mmaReg : CRT GpuFloat.Float32 64 64 ← zeroCRT .Float32 64 64
  let accum : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tmp : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Persistent shared memory for DFT matrices
  let fShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let fInvShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let twShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let twInvShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64

  -- Per-head filter (reloaded)
  let kfShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64

  -- Input/output/scratch
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let tmpShared : CST GpuFloat.BFloat16 64 64 ← allocCST .BFloat16 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  -- Register copies
  let f : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let fInv : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let tw : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let twInv : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64
  let kf : CRT GpuFloat.BFloat16 64 64 ← allocCRT .BFloat16 64 64

  comment "Load DFT matrices (persistent across all iterations)"
  loadComplex f fShared
  loadComplex fInv fInvShared
  loadComplex tw twShared
  loadComplex twInv twInvShared

  comment "Process batches with persistent cache"
  forLoop 0 16 do
    comment "Load per-head filter"
    loadComplex kf kfShared

    comment "Load input"
    load x xShared

    comment "FFT convolution steps (same as non-persistent)"

    comment "1. F @ X"
    mma mmaReg.real f.real x (← zeroRT .Float32 64 64)
    mma mmaReg.imag f.imag x (← zeroRT .Float32 64 64)
    convert accum.real mmaReg.real
    convert accum.imag mmaReg.imag

    comment "2. *= tw"
    complexMul accum accum tw

    sync

    comment "3. Y @ F"
    complexMma mmaReg accum f (← zeroCRT .Float32 64 64)
    convert accum.real mmaReg.real
    convert accum.imag mmaReg.imag

    comment "4. *= kf"
    complexMul accum accum kf

    comment "5. Y @ F_inv"
    complexMma mmaReg accum fInv (← zeroCRT .Float32 64 64)
    convert accum.real mmaReg.real
    convert accum.imag mmaReg.imag

    comment "6. *= tw_inv"
    complexMul accum accum twInv

    comment "7. Store, then F_inv @ Y"
    storeComplex tmpShared accum
    sync
    loadComplex tmp tmpShared
    complexMma mmaReg fInv tmp (← zeroCRT .Float32 64 64)

    comment "8. Output real part"
    convert out mmaReg.real
    store outShared out

    sync

def fftConvPersistentFwdKernel : Kernel :=
  buildKernelM "fftconv_persistent_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "kf_real", dtype := .BFloat16, isPointer := true },
    { name := "kf_imag", dtype := .BFloat16, isPointer := true },
    { name := "f_real", dtype := .BFloat16, isPointer := true },
    { name := "f_imag", dtype := .BFloat16, isPointer := true },
    { name := "finv_real", dtype := .BFloat16, isPointer := true },
    { name := "finv_imag", dtype := .BFloat16, isPointer := true },
    { name := "tw_real", dtype := .BFloat16, isPointer := true },
    { name := "tw_imag", dtype := .BFloat16, isPointer := true },
    { name := "twinv_real", dtype := .BFloat16, isPointer := true },
    { name := "twinv_imag", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "batch_size", dtype := .Float32, isPointer := false },
    { name := "num_heads", dtype := .Float32, isPointer := false }
  ] fftConvPersistentFwd

-- Print generated kernels
#eval IO.println "=== FFT Conv ===" *> IO.println (generateKernel fftConvFwdKernel)
#eval IO.println "\n=== FFT Conv Persistent ===" *> IO.println (generateKernel fftConvPersistentFwdKernel)

end Tyr.GPU.Kernels.FFTConv
