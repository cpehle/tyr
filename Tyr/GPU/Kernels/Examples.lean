import Tyr.GPU.Codegen.Constraints
import Tyr.GPU.Codegen.Arch
import Tyr.GPU.Kernels.Prelude

/-!
# Tyr.GPU.Kernels.Examples

Canonical `@[gpu_kernel]` examples for the Tyr GPU DSL.

This module is the single source of truth for example kernel declarations used by
docs and IDE navigation. Kernels here follow the full integration pattern:

- explicit kernel inputs (`GPtr`, `KVal`),
- explicit global-memory I/O (`loadGlobal`, `storeGlobal`),
- explicit runtime coordinates (`blockCoord2D`, `KIdx` with `krange`).

If you are looking for minimal lowering-only snippets, use `Tyr.GPU.Codegen.Tests`.
-/

namespace Tyr.GPU.Kernels.Examples

open Tyr.GPU
open Tyr.GPU.Codegen

private def gemmMainLoop (numKBlocks : Nat)
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.Float32)
    (m : KVal UInt64) (n : KVal UInt64) (k : KVal UInt64)
    : KernelM Unit := do
  let _ := m
  let _ := n
  let _ := k
  let coord ← blockCoord2D

  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let aS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bS : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let cS : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64

  for kBlk in krange 0 numKBlocks do
    loadGlobal aS aPtr (coord.withCol kBlk.id)
    loadGlobal bS bPtr (coord.withRow kBlk.id)
    sync
    load a aS
    load b bS
    mma c a b c
    sync

  store cS c
  storeGlobal cPtr cS coord

/-! ## Canonical GEMM Examples -/

/-- Hopper (SM90) GEMM example with full kernel inputs and global-memory I/O. -/
@[gpu_kernel .SM90]
def simpleGemm
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.Float32)
    (m : KVal UInt64) (n : KVal UInt64) (k : KVal UInt64)
    : KernelM Unit := do
  comment "=== Example GEMM (SM90) ==="
  gemmMainLoop 8 aPtr bPtr cPtr m n k

/-- Ampere (SM80) variant of the canonical GEMM example. -/
@[gpu_kernel .SM80]
def ampereGemm
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.Float32)
    (m : KVal UInt64) (n : KVal UInt64) (k : KVal UInt64)
    : KernelM Unit := do
  comment "=== Example GEMM (SM80) ==="
  gemmMainLoop 4 aPtr bPtr cPtr m n k

/-- Blackwell (SM100) variant of the canonical GEMM example. -/
@[gpu_kernel .SM100]
def blackwellGemm
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.Float32)
    (m : KVal UInt64) (n : KVal UInt64) (k : KVal UInt64)
    : KernelM Unit := do
  comment "=== Example GEMM (SM100) ==="
  gemmMainLoop 8 aPtr bPtr cPtr m n k

/-! ## Canonical Attention Example -/

/-- FlashAttention-style forward example with explicit `GPtr`/`KVal` inputs. -/
@[gpu_kernel .SM90]
def flashAttnFwd
    (qPtr : GPtr GpuFloat.BFloat16)
    (kPtr : GPtr GpuFloat.BFloat16)
    (vPtr : GPtr GpuFloat.BFloat16)
    (oPtr : GPtr GpuFloat.BFloat16)
    (seqLen : KVal UInt64)
    (headDim : KVal UInt64)
    : KernelM Unit := do
  let _ := seqLen
  let _ := headDim
  let coord ← blockCoord2D
  let numKvBlocks : Nat := 4

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let qS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vS : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let oS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  loadGlobal qS qPtr coord
  sync
  load q qS

  for kvBlk in krange 0 numKvBlocks do
    loadGlobal kS kPtr (coord.withRow kvBlk.id)
    loadGlobal vS vPtr (coord.withRow kvBlk.id)
    sync
    load k kS
    load v vS
    mmaT s q k s
    makeCausal s s (some (-1e10))
    rowMaxAccum rowMax s rowMax
    subCol s s rowMax
    exp s s
    rowSumAccum rowSum s rowSum
    convert p s
    mma o p v o
    sync

  divCol o o rowSum
  let oBf16 : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  convert oBf16 o
  store oS oBf16
  storeGlobal oPtr oS coord

/-! ## Canonical Normalization Example -/

/-- LayerNorm-style example with explicit pointer/scalar inputs. -/
@[gpu_kernel .SM90]
def layerNorm
    (xPtr : GPtr GpuFloat.BFloat16)
    (weightPtr : GPtr GpuFloat.BFloat16)
    (biasPtr : GPtr GpuFloat.BFloat16)
    (outPtr : GPtr GpuFloat.BFloat16)
    (n : KVal UInt64)
    : KernelM Unit := do
  let _ := n
  let coord ← blockCoord2D

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let xF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let centered : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let varTmp : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let w : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let wF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let bF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let mean : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let var : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let xS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let outS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  loadGlobal xS xPtr coord
  loadGlobal wS weightPtr coord
  loadGlobal bS biasPtr coord
  sync

  load x xS
  load w wS
  load b bS
  convert xF x
  rowSum mean xF
  subCol centered xF mean
  mul varTmp centered centered
  rowSum var varTmp
  convert wF w
  convert bF b
  mul centered centered wF
  add centered centered bF

  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  convert out centered
  store outS out
  storeGlobal outPtr outS coord

/-! ## Compatibility Aliases -/

/-- Deprecated compatibility shim for older architecture-polymorphic examples. -/
@[deprecated simpleGemm (since := "2026-03-05")]
def simpleGemmPoly (arch : ArchLevel) [HasMMA arch] : ArchKernelM arch Unit := do
  archComment "Deprecated: use full-input example kernels in this module."
  let a ← Arch.ArchKernelM.liftPortable (allocRT .BFloat16 64 64)
  let b ← Arch.ArchKernelM.liftPortable (allocRT .BFloat16 64 64 .Col)
  let c ← Arch.ArchKernelM.liftPortable (zeroRT .Float32 64 64)
  smartMMA c a b c
  archSync

/-- Deprecated alias kept for older docs and external references. -/
@[deprecated simpleGemm (since := "2026-03-05")]
abbrev simpleGemmConstrained := simpleGemm

/-- Deprecated alias kept for older docs and external references. -/
@[deprecated simpleGemm (since := "2026-03-05")]
abbrev simpleGemmNew := simpleGemm

/-- Deprecated alias kept for older docs and external references. -/
@[deprecated flashAttnFwd (since := "2026-03-05")]
abbrev flashAttnFwdNew := flashAttnFwd

-- Verify generated companion declarations on canonical examples.

end Tyr.GPU.Kernels.Examples
