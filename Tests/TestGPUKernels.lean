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
import Tyr.GPU.Codegen.Primitives
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute
import LeanTest

namespace Tests.GPUKernels

open Tyr.GPU
open Tyr.GPU.Codegen
open LeanTest

private def mkTestKernel
    (name : String)
    (arch : GpuArch)
    (body : Array KStmt)
    (params : Array KParam := #[])
    (sharedMemBytes : Nat := 0) : Kernel := {
  name := name
  arch := arch
  family := arch.toFamily
  params := params
  body := body
  sharedMemBytes := sharedMemBytes
}

private def assertContainsAll (code : String) (checks : Array (String × String)) : IO Unit := do
  for (needle, msg) in checks do
    assertTrue (code.containsSubstr needle) msg

private def assertNotContains (code : String) (needle msg : String) : IO Unit := do
  assertTrue (!(code.containsSubstr needle)) msg

/-! ## Basic Tile Allocation Tests -/

/-- Test basic register tile allocation -/
@[test]
def testAllocRT : IO Unit := do
  let kernel := buildKernelM "test_alloc_rt" .SM90 #[] do
    let _a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _b : Tyr.GPU.Codegen.RT GpuFloat.Float32 32 32 .Col ← allocRT .Float32 32 32 .Col
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
    let _s1 : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64  -- 64*64*2 = 8192
    let _s2 : Tyr.GPU.Codegen.ST GpuFloat.Float32 32 32 ← allocST .Float32 32 32    -- 32*32*4 = 4096
    pure ()

  assertEqual kernel.sharedMemBytes (8192 + 4096) "Should track shared memory"

/-- Test vector allocation -/
@[test]
def testAllocRV : IO Unit := do
  let kernel := buildKernelM "test_alloc_rv" .SM90 #[] do
    let _v : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
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
    let _z : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
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
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
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
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64  -- Row-major for transpose
    let c : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
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

@[test]
def testReverseLoopCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_reverse_loop"
    .SM90
    #[.forLoopRev { idx := 0 } 0 4 #[.comment "reverse body"]]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "for (int v0 = 4; v0-- > 0; )")
    "Reverse loops should lower to descending iteration order"

@[test]
def testReverseLoopValCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_reverse_loop_val"
    .SM90
    #[.forLoopValRev { idx := 0 } 2 { idx := 1 } #[.comment "reverse body"]]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "for (int v0 = v1; v0-- > 2; )")
    "Reverse value-bounded loops should lower to descending iteration order"

/-! ## Architecture Metadata Tests -/

@[test]
def testKernelDefaultsFamilyFromArchFloor : IO Unit := do
  let hopperKernel := buildKernelM "hopper_floor" .SM90 #[] do
    pure ()
  let blackwellKernel := buildKernelM "blackwell_floor" .SM100 #[] do
    pure ()

  assertEqual hopperKernel.arch .SM90 "Kernel floor should stay SM90"
  assertEqual hopperKernel.family .Hopper "SM90 kernels should default to Hopper family"
  assertEqual blackwellKernel.arch .SM100 "Kernel floor should stay SM100"
  assertEqual blackwellKernel.family .Blackwell "SM100 kernels should default to Blackwell family"

@[test]
def testFamilyOverrideChangesGuardWithoutChangingFloor : IO Unit := do
  let kernel := buildKernelM "gb10_guard" .SM90 #[] do
    setFamily .Blackwell
    let _tile : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    pure ()

  assertEqual kernel.arch .SM90 "GB10-compatible kernels should keep an SM90 capability floor"
  assertEqual kernel.family .Blackwell "GB10-compatible kernels should emit under the Blackwell family guard"

  let code := generateKernel kernel
  assertContainsAll code #[
    ("#if defined(KITTENS_BLACKWELL)",
      "Family override should drive the emitted availability guard")
  ]
  assertNotContains code "#if defined(KITTENS_HOPPER)"
    "Family override should replace the default Hopper guard"

@[test]
def testSetArchResetsFamilyToArchDefault : IO Unit := do
  let kernel := buildKernelM "family_reset" .SM90 #[] do
    setFamily .Blackwell
    setArch .SM80
    pure ()

  assertEqual kernel.arch .SM80 "setArch should update the capability floor"
  assertEqual kernel.family .Ampere "setArch should restore the default family for the new floor"

@[test]
def testCppLauncherUsesFamilyGuardAndFloorMessage : IO Unit := do
  let cpp := generateCppLauncherCode
    "gb10_floor_launcher"
    .SM90
    .Blackwell
    #[{ name := "x", dtype := .BFloat16, isPointer := true }]

  assertContainsAll cpp #[
    ("#if defined(KITTENS_BLACKWELL)",
      "Launchers should be guarded by the build family, not the capability floor"),
    ("requires Blackwell family, SM90 floor",
      "Unavailable-launcher diagnostics should report both family and floor")
  ]

/-! ## Code Generation Tests -/

/-- Test C++ code generation for declarations -/
@[test]
def testCodeGenDeclarations : IO Unit := do
  let kernel := buildKernelM "test_codegen_decl" .SM90 #[] do
    let _rt : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _st : Tyr.GPU.Codegen.ST GpuFloat.Float32 32 64 .Col ← allocST .Float32 32 64 .Col
    let _rv : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "rt<bf16, 64, 64, row_l>") "Should have RT declaration"
  assertTrue (code.containsSubstr "__shared__ st<float, 32, 64>") "Should have ST declaration"
  assertTrue (code.containsSubstr "rv<float, 64") "Should have RV declaration"

@[test]
def testUnsupportedSliceRowsCodegenFailsLoudly : IO Unit := do
  let kernel := mkTestKernel
    "test_bad_slice_rows"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declTT { idx := 1 } .Float32 64 64,
      .sliceRows { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "static_assert(false, \"unsupported sliceRows between non-matching tile kinds\")")
    "Unsupported sliceRows lowering should fail loudly instead of becoming a comment"

@[test]
def testMixedSliceRowsFromSharedToRegisterCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_rows_mixed"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declST { idx := 1 } .Float32 64 64 .Row,
      .sliceRows { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_src_sub = v1.template subtile<64, 64>(make_int2(0, 0));")
    "Shared-to-register sliceRows should create a shared subtile view"
  assertTrue (code.containsSubstr "kittens::warp::copy(v0, _tk_src_sub);")
    "Shared-to-register sliceRows should copy from the shared subtile into the register tile"

@[test]
def testMixedSliceRowsFromRegisterToSharedCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_rows_into_shared_supported"
    .SM90
    #[
      .declST { idx := 0 } .Float32 64 64 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .sliceRows { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_dst_sub = v0.template subtile<64, 64>(make_int2(0, 0));")
    "Register-to-shared sliceRows should create a shared destination subtile"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_sub, v1);")
    "Register-to-shared sliceRows should copy the register tile into the shared subtile"

@[test]
def testMixedSliceColsFromSharedToRegisterCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_cols_mixed"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declST { idx := 1 } .Float32 64 64 .Row,
      .sliceCols { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_src_sub = v1.template subtile<64, 64>(make_int2(0, 0));")
    "Shared-to-register sliceCols should create a shared subtile view"
  assertTrue (code.containsSubstr "kittens::warp::copy(v0, _tk_src_sub);")
    "Shared-to-register sliceCols should copy from the shared subtile into the register tile"

@[test]
def testMixedSliceColsFromRegisterToSharedCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_slice_cols_into_shared_supported"
    .SM90
    #[
      .declST { idx := 0 } .Float32 64 64 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .sliceCols { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_dst_sub = v0.template subtile<64, 64>(make_int2(0, 0));")
    "Register-to-shared sliceCols should create a shared destination subtile"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_sub, v1);")
    "Register-to-shared sliceCols should copy the register tile into the shared subtile"

@[test]
def testMixedConcatColsIntoSharedCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_concat_cols_into_shared"
    .SM90
    #[
      .declST { idx := 0 } .Float32 64 128 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declST { idx := 2 } .Float32 64 64 .Row,
      .concatCols { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "auto _tk_dst_left = v0.template subtile<64, 64>(make_int2(0, 0));")
    "Shared concat should create a left subtile"
  assertTrue (code.containsSubstr "auto _tk_dst_right = v0.template subtile<64, 64>(make_int2(0, 64));")
    "Shared concat should create a right subtile"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_left, v1);")
    "Shared concat should accept register-tile left inputs"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_dst_right, v2);")
    "Shared concat should accept shared-tile right inputs"

@[test]
def testMixedConcatColsIntoRegisterCodegen : IO Unit := do
  let kernel := mkTestKernel
    "test_concat_cols_into_register_supported"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 128 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declST { idx := 2 } .Float32 64 64 .Row,
      .concatCols { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "rt<float, 64, 64, row_l> _tk_right_rt;")
    "Mixed concat into a register tile should materialize a temporary register tile for the shared input"
  assertTrue (code.containsSubstr "auto _tk_right_sub = v2.template subtile<64, 64>(make_int2(0, 0));")
    "Mixed concat into a register tile should create a shared subtile view for the shared input"
  assertTrue (code.containsSubstr "kittens::warp::copy(_tk_right_rt, _tk_right_sub);")
    "Mixed concat into a register tile should copy the shared side into the temporary register tile"

@[test]
def testEqMaskCodegenEmitsHelpers : IO Unit := do
  let kernel := mkTestKernel
    "test_eq_mask"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declRT { idx := 2 } .Float32 64 64 .Row,
      .eqMask { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "tk_eq_mask(v0, v1, v2);")
    "eqMask should lower to the dedicated helper call"
  assertTrue (code.containsSubstr "template<ducks::rt::all T>")
    "eqMask helper templates should be emitted when the kernel uses eqMask"

@[test]
def testUnsupportedSliceColsCodegenFailsLoudly : IO Unit := do
  let kernel := mkTestKernel
    "test_bad_slice_cols"
    .SM90
    #[
      .declRT { idx := 0 } .Float32 64 64 .Row,
      .declTT { idx := 1 } .Float32 64 64,
      .sliceCols { idx := 0 } { idx := 1 } 0 64
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "static_assert(false, \"unsupported sliceCols between non-matching tile kinds\")")
    "Unsupported sliceCols lowering should fail loudly instead of becoming a comment"

@[test]
def testUnsupportedConcatColsCodegenFailsLoudly : IO Unit := do
  let kernel := mkTestKernel
    "test_bad_concat_cols"
    .SM90
    #[
      .declTT { idx := 0 } .Float32 64 128,
      .declRT { idx := 1 } .Float32 64 64 .Row,
      .declRT { idx := 2 } .Float32 64 64 .Row,
      .concatCols { idx := 0 } { idx := 1 } { idx := 2 }
    ]

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "static_assert(false, \"unsupported concatCols between non-matching tile kinds\")")
    "Unsupported concatCols lowering should fail loudly instead of becoming a comment"

/-- Test C++ code generation for operations -/
@[test]
def testCodeGenOperations : IO Unit := do
  let kernel := buildKernelM "test_codegen_ops" .SM90 #[] do
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
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
    let a : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let b : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let c : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
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
    let t : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
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
    let t : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let v : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
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
    let s : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    makeCausal s s (some (-1e10))
    pure ()

  let code := generateKernel kernel
  assertTrue (code.containsSubstr "make_causal(") "Should have make_causal"
  assertTrue (code.containsSubstr "-10000000000") "Should have fill value"

/-- Test triangular masks -/
@[test]
def testTriangularMasks : IO Unit := do
  let kernel := buildKernelM "test_tril_triu" .SM90 #[] do
    let t : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
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
    let r : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let s : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
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
    let r : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let s : Tyr.GPU.Codegen.ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
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
    let bf : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let fp : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
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
    let row : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let col : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
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
  assertTrue (code.containsSubstr "gl<bf16, 1, 1, -1, -1> v0") "Should have bf16 global descriptor param"
  assertTrue (code.containsSubstr "gl<float, 1, 1, -1, -1> v1") "Should have float global descriptor param"
  assertTrue (code.containsSubstr "uint64_t v2") "Should have scalar param"

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
    let q : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let k : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let _v : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    let s : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let o : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

    let rowMax : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
    let rowSum : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64

    let qShared : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
    let kShared : Tyr.GPU.Codegen.ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

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
  assertTrue (code.containsSubstr "gl<bf16, 1, 1, -1, -1> v0") "Should have first global descriptor param"
  assertTrue (code.containsSubstr "mma_ABt(") "Should have mmaT"
  assertTrue (code.containsSubstr "make_causal(") "Should have causal mask"
  assertTrue (code.containsSubstr "row_max(") "Should have row_max"
  assertTrue (code.containsSubstr "for (int") "Should have for loop"
  assertTrue (code.containsSubstr "div_row(") "Should have final normalization"

/-- Test complete LayerNorm-like kernel structure -/
@[test]
def testLayerNormStructure : IO Unit := do
  let kernel := buildKernelM "layernorm_test" .SM90 #[] do
    let x : Tyr.GPU.Codegen.RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    let xf : Tyr.GPU.Codegen.RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let mean : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64
    let var : Tyr.GPU.Codegen.RV GpuFloat.Float32 64 ← allocRV .Float32 64

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
