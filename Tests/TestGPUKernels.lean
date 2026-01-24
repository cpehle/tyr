/-
  Tests/TestGPUKernels.lean

  Tests for the native Lean4 GPU kernel DSL.
  Tests type safety, code generation, and architecture specialization.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import LeanTest

namespace Tests.GPUKernels

open Tyr.GPU
open Tyr.GPU.Codegen
open LeanTest

/-! ## Basic Tile Allocation Tests -/

/-- Test basic register tile allocation -/
@[test]
def testAllocRT : IO Unit := do
  let kernel := buildKernelM "test_alloc_rt" .SM90 #[] do
    let _a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _b : RT GpuFloat.Float32 32 32 .Col ← allocRT .Float32 32 32 .Col
    pure ()

  assertEqual kernel.body.size 2 "Should have 2 declarations"
  match kernel.body[0]! with
  | .declRT _ dtype rows cols layout =>
    assertEqual dtype GpuFloat.BFloat16 "First tile should be BFloat16"
    assertEqual rows 64 "First tile should have 64 rows"
    assertEqual cols 64 "First tile should have 64 cols"
    assertEqual layout TileLayout.Row "First tile should be row-major"
  | _ => fail "First statement should be declRT"

/-- Test shared tile allocation tracks memory -/
@[test]
def testAllocSTTracksMemory : IO Unit := do
  let kernel := buildKernelM "test_shared_mem" .SM90 #[] do
    let _s1 : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64  -- 64*64*2 = 8192
    let _s2 : ST GpuFloat.Float32 32 32 ← allocST .Float32 32 32    -- 32*32*4 = 4096
    pure ()

  assertEqual kernel.sharedMemBytes (8192 + 4096) "Should track shared memory"

/-- Test vector allocation -/
@[test]
def testAllocRV : IO Unit := do
  let kernel := buildKernelM "test_alloc_rv" .SM90 #[] do
    let _v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    pure ()

  match kernel.body[0]! with
  | .declRV _ dtype len =>
    assertEqual dtype GpuFloat.Float32 "Vector should be Float32"
    assertEqual len 64 "Vector should have length 64"
  | _ => fail "Should be declRV"

/-- Test zero-initialized allocation -/
@[test]
def testZeroRT : IO Unit := do
  let kernel := buildKernelM "test_zero_rt" .SM90 #[] do
    let _z : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    pure ()

  assertEqual kernel.body.size 2 "Should have decl + zero"
  match kernel.body[1]! with
  | .unary op _ _ => assertEqual op UnaryOp.Zero "Should be zero operation"
  | _ => fail "Second statement should be unary zero"

/-! ## MMA Type Safety Tests -/

/-- Test MMA generates correct IR -/
@[test]
def testMMAGeneration : IO Unit := do
  let kernel := buildKernelM "test_mma" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma c a b c
    pure ()

  -- Find the MMA statement
  let mmaStmt := kernel.body.find? fun s =>
    match s with
    | .mma _ _ _ _ _ => true
    | _ => false

  assertTrue mmaStmt.isSome "Should contain MMA statement"
  match mmaStmt with
  | some (.mma trans _ _ _ _) =>
    assertEqual trans MMATranspose.AB "Should be AB transpose mode"
  | _ => fail "Should be MMA with AB mode"

/-- Test MMA with B transposed -/
@[test]
def testMMAT : IO Unit := do
  let kernel := buildKernelM "test_mma_t" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64  -- Row-major for transpose
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mmaT c a b c
    pure ()

  let mmaStmt := kernel.body.find? fun s =>
    match s with
    | .mma _ _ _ _ _ => true
    | _ => false

  match mmaStmt with
  | some (.mma trans _ _ _ _) =>
    assertEqual trans MMATranspose.ABt "Should be ABt transpose mode"
  | _ => fail "Should be MMA with ABt mode"

/-! ## Loop Generation Tests -/

/-- Test for loop generates correct structure -/
@[test]
def testForLoop : IO Unit := do
  let kernel := buildKernelM "test_loop" .SM90 #[] do
    forLoop 0 16 do
      sync
    pure ()

  assertEqual kernel.body.size 1 "Should have one loop"
  match kernel.body[0]! with
  | .forLoop _ lo hi body =>
    assertEqual lo 0 "Loop should start at 0"
    assertEqual hi 16 "Loop should end at 16"
    assertEqual body.size 1 "Loop body should have 1 statement"
  | _ => fail "Should be forLoop"

/-- Test nested loops -/
@[test]
def testNestedLoops : IO Unit := do
  let kernel := buildKernelM "test_nested" .SM90 #[] do
    forLoop 0 4 do
      forLoop 0 8 do
        sync
    pure ()

  match kernel.body[0]! with
  | .forLoop _ _ _ outerBody =>
    match outerBody[0]! with
    | .forLoop _ lo hi innerBody =>
      assertEqual lo 0 "Inner loop start"
      assertEqual hi 8 "Inner loop end"
      assertEqual innerBody.size 1 "Inner body size"
    | _ => fail "Should have inner forLoop"
  | _ => fail "Should have outer forLoop"

/-! ## Code Generation Tests -/

/-- Test C++ code generation for declarations -/
@[test]
def testCodeGenDeclarations : IO Unit := do
  let kernel := buildKernelM "test_codegen_decl" .SM90 #[] do
    let _rt : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _st : ST GpuFloat.Float32 32 64 .Col ← allocST .Float32 32 64 .Col
    let _rv : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "rt<bf16, 64, 64, row_l>") "Should have RT declaration"
  assertTrue (code.containsSubstr "st<float, 32, 64, col_l>") "Should have ST declaration"
  assertTrue (code.containsSubstr "rv<float, 64") "Should have RV declaration"

/-- Test C++ code generation for operations -/
@[test]
def testCodeGenOperations : IO Unit := do
  let kernel := buildKernelM "test_codegen_ops" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    add a a b
    exp a a
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "add(") "Should have add operation"
  assertTrue (code.containsSubstr "exp(") "Should have exp operation"

/-- Test C++ code generation for MMA -/
@[test]
def testCodeGenMMA : IO Unit := do
  let kernel := buildKernelM "test_codegen_mma" .SM90 #[] do
    let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma c a b c
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "mma_AB(") "Should have mma_AB call"

/-- Test architecture guard in generated code -/
@[test]
def testCodeGenArchGuard : IO Unit := do
  let kernelSM90 := buildKernelM "test_sm90" .SM90 #[] (pure ())
  let kernelSM80 := buildKernelM "test_sm80" .SM80 #[] (pure ())
  let kernelSM100 := buildKernelM "test_sm100" .SM100 #[] (pure ())

  let code90 := generateKernel kernelSM90
  let code80 := generateKernel kernelSM80
  let code100 := generateKernel kernelSM100

  assertTrue (code90.containsSubstr "KITTENS_HOPPER") "SM90 should use HOPPER guard"
  assertTrue (code80.containsSubstr "KITTENS_AMPERE") "SM80 should use AMPERE guard"
  assertTrue (code100.containsSubstr "KITTENS_BLACKWELL") "SM100 should use BLACKWELL guard"

/-! ## Reduction and Broadcast Tests -/

/-- Test row-wise reductions -/
@[test]
def testRowReduction : IO Unit := do
  let kernel := buildKernelM "test_row_reduce" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    rowSum v t
    rowMax v t
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "row_sum(") "Should have row_sum"
  assertTrue (code.containsSubstr "row_max(") "Should have row_max"

/-- Test column broadcast operations -/
@[test]
def testColBroadcast : IO Unit := do
  let kernel := buildKernelM "test_col_broadcast" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    subCol t t v
    divCol t t v
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sub_row(") "Should have sub_row"
  assertTrue (code.containsSubstr "div_row(") "Should have div_row"

/-! ## Masking Tests -/

/-- Test causal mask generation -/
@[test]
def testCausalMask : IO Unit := do
  let kernel := buildKernelM "test_causal" .SM90 #[] do
    let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    makeCausal s s (some (-1e10))
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "make_causal(") "Should have make_causal"
  assertTrue (code.containsSubstr "-10000000000") "Should have fill value"

/-- Test triangular masks -/
@[test]
def testTriangularMasks : IO Unit := do
  let kernel := buildKernelM "test_tril_triu" .SM90 #[] do
    let t : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    tril t t 0 (some 0.0)
    triu t t 1 none
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "tril(") "Should have tril"
  assertTrue (code.containsSubstr "triu(") "Should have triu"

/-! ## Synchronization Tests -/

/-- Test sync and arrive operations -/
@[test]
def testSynchronization : IO Unit := do
  let kernel := buildKernelM "test_sync" .SM90 #[] do
    sync 0
    arrive 1
    mmaCommitGroup
    mmaAsyncWait 2
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "sync(0)") "Should have sync"
  assertTrue (code.containsSubstr "arrive(1)") "Should have arrive"
  assertTrue (code.containsSubstr "mma_commit_group()") "Should have mma_commit_group"
  assertTrue (code.containsSubstr "mma_async_wait<2>()") "Should have mma_async_wait"

/-! ## Memory Operations Tests -/

/-- Test load/store operations -/
@[test]
def testLoadStore : IO Unit := do
  let kernel := buildKernelM "test_load_store" .SM90 #[] do
    let r : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let s : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    load r s
    store s r
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "load(") "Should have load"
  assertTrue (code.containsSubstr "store(") "Should have store"

/-- Test atomic store-add for gradients -/
@[test]
def testStoreAdd : IO Unit := do
  let kernel := buildKernelM "test_store_add" .SM90 #[] do
    let r : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let s : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
    storeAdd s r
    storeAddAsync s r
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "store_add(") "Should have store_add"
  assertTrue (code.containsSubstr "tma::store_add_async(") "Should have tma store_add_async"

/-! ## Type Conversion Tests -/

/-- Test type conversion -/
@[test]
def testConvert : IO Unit := do
  let kernel := buildKernelM "test_convert" .SM90 #[] do
    let bf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let fp : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    convert fp bf
    convert bf fp
    pure ()

  let code := generateKernel kernel
  -- convert uses copy in ThunderKittens
  assertTrue (code.containsSubstr "copy(") "Should have copy for conversion"

/-- Test layout swap -/
@[test]
def testSwapLayout : IO Unit := do
  let kernel := buildKernelM "test_swap_layout" .SM90 #[] do
    let row : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let col : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout row col
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "swap_layout(") "Should have swap_layout"

/-! ## Parameter Handling Tests -/

/-- Test kernel parameter generation -/
@[test]
def testKernelParams : IO Unit := do
  let kernel := buildKernelM "test_params" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "y_ptr", dtype := .Float32, isPointer := true },
    { name := "size", dtype := .Float32, isPointer := false }
  ] (pure ())

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "bf16* x_ptr") "Should have bf16 pointer param"
  assertTrue (code.containsSubstr "float* y_ptr") "Should have float pointer param"
  assertTrue (code.containsSubstr "float size") "Should have scalar param"

/-! ## Integration Tests -/

/-- Test complete FlashAttention-like kernel structure -/
@[test]
def testFlashAttnStructure : IO Unit := do
  let kernel := buildKernelM "flash_attn_test" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true }
  ] do
    let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

    let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
    let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

    let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

    load q qShared

    forLoop 0 4 do
      load k kShared
      mmaT s q k s
      makeCausal s s (some (-1e10))
      rowMaxAccum rowMax s rowMax
      subCol s s rowMax
      exp s s
      rowSumAccum rowSum s rowSum
      sync

    divCol o o rowSum

  let code := generateKernel kernel

  -- Check all expected components
  assertTrue (code.containsSubstr "flash_attn_test") "Should have kernel name"
  assertTrue (code.containsSubstr "bf16* Q") "Should have Q param"
  assertTrue (code.containsSubstr "mma_ABt(") "Should have mmaT"
  assertTrue (code.containsSubstr "make_causal(") "Should have causal mask"
  assertTrue (code.containsSubstr "row_max(") "Should have row_max"
  assertTrue (code.containsSubstr "for (int") "Should have for loop"
  assertTrue (code.containsSubstr "div_row(") "Should have final normalization"

/-- Test complete LayerNorm-like kernel structure -/
@[test]
def testLayerNormStructure : IO Unit := do
  let kernel := buildKernelM "layernorm_test" .SM90 #[] do
    let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let xf : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let mean : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let var : RV GpuFloat.Float32 64 ← allocRV .Float32 64

    convert xf x
    rowSum mean xf
    subCol xf xf mean
    mul xf xf xf
    rowSum var xf
    -- rsqrt would go here
    convert x xf

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "row_sum(") "Should have row_sum"
  assertTrue (code.containsSubstr "sub_row(") "Should have sub_row"
  assertTrue (code.containsSubstr "mul(") "Should have mul"

end Tests.GPUKernels
