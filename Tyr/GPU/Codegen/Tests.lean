/-
  Tyr/GPU/Codegen/Tests.lean

  Expect-style tests using #guard_msgs for GPU kernel DSL.
  These tests verify exact code generation output.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.ArchConfig
import Tyr.GPU.Codegen.Arch
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Codegen.Tests

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Basic Declaration Tests -/

/-- Simple kernel with one tile -/
def simpleTileKernel : Kernel :=
  buildKernelM "simple_tile" .SM90 #[] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void simple_tile(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel simpleTileKernel)

/-- Kernel with multiple tile types -/
def multiTileKernel : Kernel :=
  buildKernelM "multi_tile" .SM90 #[] do
    let _rt : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _st : ST GpuFloat.Float32 32 64 .Col ← allocST .Float32 32 64 .Col
    let _rv : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let _sv : SV GpuFloat.BFloat16 128 ← allocSV .BFloat16 128

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void multi_tile(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
  st<float, 32, 64, col_l> v1;
  rv<float, 64> v2;
  sv<bf16, 128> v3;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel multiTileKernel)

/-! ## MMA Tests -/

/-- MMA with AB mode -/
def mmaABKernel : Kernel :=
  buildKernelM "mma_ab" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma c a b c

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mma_ab(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
  rt<bf16, 64, 64, col_l> v1;
  rt<float, 64, 64, row_l> v2;
  zero(v2, v2);
  mma_AB(v2, v0, v1, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel mmaABKernel)

/-- MMA with ABt mode (B transposed) -/
def mmaABtKernel : Kernel :=
  buildKernelM "mma_abt" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mmaT c a b c

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mma_abt(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
  rt<bf16, 64, 64, row_l> v1;
  rt<float, 64, 64, row_l> v2;
  zero(v2, v2);
  mma_ABt(v2, v0, v1, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel mmaABtKernel)

/-! ## Loop Tests -/

/-- Simple for loop -/
def simpleLoopKernel : Kernel :=
  buildKernelM "simple_loop" .SM90 #[] do
    forLoop 0 4 do
      sync

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void simple_loop(/* TODO: params */) {
  for (int v0 = 0; v0 < 4; v0++) {
    sync(0);
  }
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel simpleLoopKernel)

/-- Nested loops -/
def nestedLoopKernel : Kernel :=
  buildKernelM "nested_loop" .SM90 #[] do
    forLoop 0 2 do
      forLoop 0 3 do
        sync 1

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void nested_loop(/* TODO: params */) {
  for (int v0 = 0; v0 < 2; v0++) {
    for (int v1 = 0; v1 < 3; v1++) {
      sync(1);
    }
  }
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel nestedLoopKernel)

/-! ## Architecture Guard Tests -/

/-- SM80 (Ampere) kernel -/
def sm80Kernel : Kernel :=
  buildKernelM "sm80_kernel" .SM80 #[] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_SM80)
__global__ void sm80_kernel(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel sm80Kernel)

/-- SM100 (Blackwell) kernel -/
def sm100Kernel : Kernel :=
  buildKernelM "sm100_kernel" .SM100 #[] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void sm100_kernel(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel sm100Kernel)

/-! ## Reduction and Broadcast Tests -/

/-- Row reduction operations -/
def rowReductionKernel : Kernel :=
  buildKernelM "row_reduction" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    rowSum v t
    rowMax v t
    rowSumAccum v t v

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void row_reduction(/* TODO: params */) {
  rt<float, 64, 64, row_l> v0;
  rv<float, 64> v1;
  row_sum(v1, v0);
  row_max(v1, v0);
  row_sum(v1, v0, v1);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel rowReductionKernel)

/-- Column broadcast operations -/
def colBroadcastKernel : Kernel :=
  buildKernelM "col_broadcast" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    subCol t t v
    mulCol t t v
    divCol t t v

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void col_broadcast(/* TODO: params */) {
  rt<float, 64, 64, row_l> v0;
  rv<float, 64> v1;
  sub_col(v0, v0, v1);
  mul_col(v0, v0, v1);
  div_col(v0, v0, v1);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel colBroadcastKernel)

/-! ## Masking Tests -/

/-- Causal and triangular masks -/
def maskKernel : Kernel :=
  buildKernelM "mask_ops" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    makeCausal t t (some (-1e10))
    tril t t 0 none
    triu t t 1 (some 0.0)

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mask_ops(/* TODO: params */) {
  rt<float, 64, 64, row_l> v0;
  make_causal(v0, v0, -10000000000.000000);
  tril(v0, v0, 0);
  triu(v0, v0, 1, 0.000000);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel maskKernel)

/-! ## Memory Operations Tests -/

/-- Load, store, and atomic operations -/
def memoryKernel : Kernel :=
  buildKernelM "memory_ops" .SM90 #[] do
    let r : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let s : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    load r s
    store s r
    let rF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let sF : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
    storeAdd sF rF

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void memory_ops(/* TODO: params */) {
  rt<bf16, 64, 64, row_l> v0;
  st<bf16, 64, 64, row_l> v1;
  load(v0, v1);
  store(v1, v0);
  rt<float, 64, 64, row_l> v2;
  st<float, 64, 64, row_l> v3;
  store_add(v3, v2);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel memoryKernel)

/-! ## Parameter Tests -/

/-- Kernel with typed parameters -/
def paramKernel : Kernel :=
  buildKernelM "with_params" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "y_ptr", dtype := .Float32, isPointer := true },
    { name := "n", dtype := .Float32, isPointer := false }
  ] do
    let _t : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void with_params(bf16* x_ptr, float* y_ptr, float n) {
  rt<bf16, 64, 64, row_l> v3;
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel paramKernel)

/-! ## Complete Mini-FlashAttention Test -/

/-- Minimal FlashAttention structure -/
def miniFlashAttn : Kernel :=
  buildKernelM "mini_flash_attn" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true }
  ] do
    comment "Tiles"
    let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    comment "Vectors"
    let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
    let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    comment "Shared"
    let qS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let kS : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let vS : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

    load q qS
    forLoop 0 4 do
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

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void mini_flash_attn(bf16* Q, bf16* K, bf16* V, bf16* O) {
  // Tiles
  rt<bf16, 64, 64, row_l> v4;
  rt<bf16, 64, 64, row_l> v5;
  rt<bf16, 64, 64, col_l> v6;
  rt<float, 64, 64, row_l> v7;
  rt<float, 64, 64, row_l> v8;
  zero(v8, v8);
  rt<bf16, 64, 64, row_l> v9;
  // Vectors
  rv<float, 64> v10;
  neg_infty(v10, v10);
  rv<float, 64> v11;
  // Shared
  st<bf16, 64, 64, row_l> v12;
  st<bf16, 64, 64, row_l> v13;
  st<bf16, 64, 64, col_l> v14;
  load(v4, v12);
  for (int v15 = 0; v15 < 4; v15++) {
    load(v5, v13);
    load(v6, v14);
    mma_ABt(v7, v4, v5, v7);
    make_causal(v7, v7, -10000000000.000000);
    row_max(v10, v7, v10);
    sub_col(v7, v7, v10);
    exp(v7, v7);
    row_sum(v11, v7, v11);
    copy(v9, v7);
    mma_AB(v8, v9, v6, v8);
    sync(0);
  }
  div_col(v8, v8, v11);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel miniFlashAttn)

/-! ## Architecture Capability Tests -/

-- Test architecture capabilities
#guard GpuArch.SM80.capabilities.hasTMA = false
#guard GpuArch.SM90.capabilities.hasTMA = true
#guard GpuArch.SM100.capabilities.hasTMA = true

#guard GpuArch.SM80.capabilities.hasWGMMA = false
#guard GpuArch.SM90.capabilities.hasWGMMA = true
#guard GpuArch.SM100.capabilities.hasWGMMA = true

#guard GpuArch.SM80.capabilities.hasFP8 = false
#guard GpuArch.SM90.capabilities.hasFP8 = true
#guard GpuArch.SM100.capabilities.hasFP8 = true

-- Test dtype support
#guard GpuArch.SM80.supportsDtype .BFloat16 = true
#guard GpuArch.SM80.supportsDtype .FP8E4M3 = false
#guard GpuArch.SM90.supportsDtype .FP8E4M3 = true
#guard GpuArch.SM100.supportsDtype .FP8E5M2 = true

-- Test default config generation
#guard (ArchKernelConfig.default .SM80).pipelineStages = 2
#guard (ArchKernelConfig.default .SM90).pipelineStages = 4
#guard (ArchKernelConfig.default .SM100).pipelineStages = 4

#guard (ArchKernelConfig.default .SM80).useWGMMA = false
#guard (ArchKernelConfig.default .SM90).useWGMMA = true

/-! ## Polymorphic Kernel Tests -/

open Arch in
/-- Example polymorphic kernel that adapts to different GPU architectures.
    When marked with @[gpu_kernel] (no arch argument), it generates
    kernel variants for SM80, SM90, and SM100 automatically.

    The function takes an ArchLevel parameter and returns ArchKernelM arch Unit.
    Instance resolution happens at compile time when the kernel is instantiated
    for a specific architecture. -/
def examplePolyMatmul [HasMMA arch] [ArchConfig arch]
    (_A : GPtr .BFloat16) (_B : GPtr .BFloat16) (_C : GPtr .BFloat16)
    : ArchKernelM arch Unit := do
  -- Get architecture-specific configuration via typeclass
  let cfg := ArchConfig.toRecord (arch := arch)
  let (tileM, tileN, _) := cfg.mmaTileSize

  -- Emit architecture info as a comment
  archComment s!"Tile: {tileM}x{tileN}, TMA: {cfg.hasTMA}, WGMMA: {cfg.hasWGMMA}"

  -- Allocate tiles
  let a ← ArchKernelM.liftPortable (allocRT .BFloat16 64 64)
  let b ← ArchKernelM.liftPortable (allocRT .BFloat16 64 64 .Col)
  let c ← ArchKernelM.liftPortable (zeroRT .Float32 64 64)

  -- smartMMA dispatches at compile time via typeclass:
  -- - Ampere: plain mma_AB
  -- - Hopper/Blackwell: mma_fence + mma_AB + mma_commit_group
  smartMMA c a b c

  archSync

/-- Test that the polymorphic kernel generates code for Ampere (SM80) -/
def polyKernelSM80 : Kernel :=
  buildKernelM "examplePolyMatmul_SM80" .SM80 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .BFloat16, isPointer := true }
  ] (examplePolyMatmul (arch := .Ampere)
      (GPtr.mk ⟨0⟩ "A")
      (GPtr.mk ⟨1⟩ "B")
      (GPtr.mk ⟨2⟩ "C")).run

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_SM80)
__global__ void examplePolyMatmul_SM80(bf16* A, bf16* B, bf16* C) {
  // Tile: 16x16, TMA: false, WGMMA: false
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  rt<float, 64, 64, row_l> v5;
  zero(v5, v5);
  mma_AB(v5, v3, v4, v5);
  sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel polyKernelSM80)

/-- Test that the polymorphic kernel generates code for Hopper (SM90) with WGMMA -/
def polyKernelSM90 : Kernel :=
  buildKernelM "examplePolyMatmul_SM90" .SM90 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .BFloat16, isPointer := true }
  ] (examplePolyMatmul (arch := .Hopper)
      (GPtr.mk ⟨0⟩ "A")
      (GPtr.mk ⟨1⟩ "B")
      (GPtr.mk ⟨2⟩ "C")).run

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_HOPPER)
__global__ void examplePolyMatmul_SM90(bf16* A, bf16* B, bf16* C) {
  // Tile: 64x64, TMA: true, WGMMA: true
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  rt<float, 64, 64, row_l> v5;
  zero(v5, v5);
  mma_fence(v5);
  mma_AB(v5, v3, v4, v5);
  mma_commit_group();
  sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel polyKernelSM90)

/-- Test that the polymorphic kernel generates code for Blackwell (SM100) -/
def polyKernelSM100 : Kernel :=
  buildKernelM "examplePolyMatmul_SM100" .SM100 #[
    { name := "A", dtype := .BFloat16, isPointer := true },
    { name := "B", dtype := .BFloat16, isPointer := true },
    { name := "C", dtype := .BFloat16, isPointer := true }
  ] (examplePolyMatmul (arch := .Blackwell)
      (GPtr.mk ⟨0⟩ "A")
      (GPtr.mk ⟨1⟩ "B")
      (GPtr.mk ⟨2⟩ "C")).run

/--
info: #include <kittens.cuh>
using namespace kittens;

#if defined(KITTENS_BLACKWELL)
__global__ void examplePolyMatmul_SM100(bf16* A, bf16* B, bf16* C) {
  // Tile: 64x64, TMA: true, WGMMA: true
  rt<bf16, 64, 64, row_l> v3;
  rt<bf16, 64, 64, col_l> v4;
  rt<float, 64, 64, row_l> v5;
  zero(v5, v5);
  mma_fence(v5);
  mma_AB(v5, v3, v4, v5);
  mma_commit_group();
  sync(0);
}
#endif
-/
#guard_msgs in
#eval IO.println (generateKernel polyKernelSM100)

/-! ## Polymorphic Kernel Pattern

The recommended pattern for architecture-polymorphic kernels uses typeclass constraints
for compile-time dispatch. When instantiated with a concrete architecture, the correct
implementation is selected at compile time with zero runtime overhead.

```lean
-- Define polymorphic kernel with typeclass constraint
def myPolyKernel [HasMMA arch] (input : GPtr .BFloat16) (output : GPtr .BFloat16)
    : ArchKernelM arch Unit := do
  let cfg := ArchConfig.toRecord (arch := arch)
  -- Tile sizes, etc. are resolved at compile time
  let a ← ArchKernelM.liftPortable (allocRT .BFloat16 64 64)
  ...
  -- smartMMA dispatches via typeclass - compile-time selection
  smartMMA dst a b c

-- Instantiate for each architecture (generates different code)
def kernel_SM80 := buildKernelM "k" .SM80 #[...] (myPolyKernel (arch := .Ampere) ...).run
def kernel_SM90 := buildKernelM "k" .SM90 #[...] (myPolyKernel (arch := .Hopper) ...).run
```

The `@[gpu_kernel]` attribute can automate this instantiation when enhanced to detect
the typeclass-constrained pattern.
-/

end Tyr.GPU.Codegen.Tests
