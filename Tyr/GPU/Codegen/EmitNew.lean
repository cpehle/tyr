/-
  Tyr/GPU/Codegen/EmitNew.lean

  C++ code generation from VarId-based kernel IR.
  Produces ThunderKittens-compatible CUDA code.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.AST

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Generate C++ for a single statement -/
partial def generateStmt (indent : String := "  ") : KStmt → String
  -- Declarations
  | .declRT v dtype rows cols layout =>
    s!"{indent}rt<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {v.toIdent};\n"
  | .declST v dtype rows cols layout =>
    s!"{indent}st<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {v.toIdent};\n"
  | .declRV v dtype len =>
    s!"{indent}rv<{dtype.toCpp}, {len}> {v.toIdent};\n"
  | .declSV v dtype len =>
    s!"{indent}sv<{dtype.toCpp}, {len}> {v.toIdent};\n"
  | .declSemaphore v =>
    s!"{indent}semaphore {v.toIdent};\n"

  -- Kernel parameter declarations (these are part of the signature, not body)
  -- When they appear in the body, just emit a comment for debugging
  | .declGPtr v dtype name =>
    s!"{indent}// param: {dtype.toCpp}* {name} (v{v.idx})\n"
  | .declKVal v dtype name =>
    s!"{indent}// param: {dtype.toCpp} {name} (v{v.idx})\n"

  -- Memory operations
  | .load dst src => s!"{indent}load({dst.toIdent}, {src.toIdent});\n"
  | .store dst src => s!"{indent}store({dst.toIdent}, {src.toIdent});\n"
  | .loadAsync dst src => s!"{indent}load_async({dst.toIdent}, {src.toIdent});\n"
  | .storeAsync dst src => s!"{indent}store_async({dst.toIdent}, {src.toIdent});\n"
  | .storeAdd dst src => s!"{indent}store_add({dst.toIdent}, {src.toIdent});\n"
  | .storeAddAsync dst src => s!"{indent}tma::store_add_async({dst.toIdent}, {src.toIdent});\n"
  | .storeMinAsync dst src => s!"{indent}tma::store_min_async({dst.toIdent}, {src.toIdent});\n"
  | .prefetch src => s!"{indent}tma::prefetch({src.toIdent});\n"
  | .tmaExpect barrier bytes => s!"{indent}tma::expect_bytes({barrier.toIdent}, {bytes});\n"

  -- TMA operations with global pointers
  | .tmaLoad dst src coord =>
    s!"{indent}tma::load({dst.toIdent}, {src.toIdent}, {coord.toIdent});\n"
  | .tmaStore dst src coord =>
    s!"{indent}tma::store({dst.toIdent}, {src.toIdent}, {coord.toIdent});\n"

  -- Distributed / Multimem operations
  | .multimemLoadReduce op dst src =>
    s!"{indent}multimem::load_reduce_{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
  | .multimemStore dst src =>
    s!"{indent}multimem::store({dst.toIdent}, {src.toIdent});\n"
  | .multimemRed op dst src =>
    s!"{indent}multimem::reduce_{op.toCpp}({dst.toIdent}, {src.toIdent});\n"

  -- MMA operations
  | .mma trans dst a b c =>
    s!"{indent}mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"
  | .mm trans dst a b =>
    s!"{indent}mm_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
  | .mmaFence dst => s!"{indent}mma_fence({dst.toIdent});\n"
  | .mmaCommitGroup => s!"{indent}mma_commit_group();\n"
  | .mmaAsyncWait n => s!"{indent}mma_async_wait<{n}>();\n"

  -- Element-wise unary
  | .unary op dst src => s!"{indent}{op.toCpp}({dst.toIdent}, {src.toIdent});\n"

  -- Element-wise binary
  | .binary op dst a b =>
    s!"{indent}{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"

  -- Element-wise ternary (FMA)
  | .ternary op dst a b c =>
    s!"{indent}{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"

  -- Scalar operations
  | .scalarMul dst src scalar =>
    s!"{indent}mul({dst.toIdent}, {src.toIdent}, {scalar}f);\n"
  | .scalarAdd dst src scalar =>
    s!"{indent}add({dst.toIdent}, {src.toIdent}, {scalar}f);\n"

  -- Broadcasting
  | .broadcast axis dst vec =>
    s!"{indent}broadcast{axis.toSuffix}({dst.toIdent}, {vec.toIdent});\n"
  | .binaryBroadcast op axis dst tile vec =>
    s!"{indent}{op.toCpp}{axis.toSuffix}({dst.toIdent}, {tile.toIdent}, {vec.toIdent});\n"

  -- Reductions
  | .reduce op axis dst src =>
    s!"{indent}{axis.toPrefix}{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
  | .reduceAccum op axis dst src accum =>
    s!"{indent}{axis.toPrefix}{op.toCpp}({dst.toIdent}, {src.toIdent}, {accum.toIdent});\n"

  -- Scan operations
  | .cumsum axis dst src =>
    s!"{indent}{axis.toPrefix}cumsum({dst.toIdent}, {src.toIdent});\n"
  | .cumprod axis dst src =>
    s!"{indent}{axis.toPrefix}cumprod({dst.toIdent}, {src.toIdent});\n"

  -- Outer product
  | .outer dst a b =>
    s!"{indent}outer({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"

  -- Layout/type conversions
  | .swapLayout dst src => s!"{indent}swap_layout({dst.toIdent}, {src.toIdent});\n"
  | .transpose dst src => s!"{indent}transpose({dst.toIdent}, {src.toIdent});\n"
  | .convert dst src => s!"{indent}copy({dst.toIdent}, {src.toIdent});\n"

  -- Masking
  | .mask op dst src fillVal =>
    let fillStr := fillVal.map (fun v => s!", {v}") |>.getD ""
    match op with
    | .Tril d => s!"{indent}tril({dst.toIdent}, {src.toIdent}, {d}{fillStr});\n"
    | .Triu d => s!"{indent}triu({dst.toIdent}, {src.toIdent}, {d}{fillStr});\n"
    | .MakeCausal => s!"{indent}make_causal({dst.toIdent}, {src.toIdent}{fillStr});\n"
    | .MakeCausalT => s!"{indent}make_causal_t({dst.toIdent}, {src.toIdent}{fillStr});\n"
    | .RightFill c => s!"{indent}right_fill({dst.toIdent}, {src.toIdent}, {c}{fillStr});\n"
    | .LeftFill c => s!"{indent}left_fill({dst.toIdent}, {src.toIdent}, {c}{fillStr});\n"
    | .UpperFill r => s!"{indent}upper_fill({dst.toIdent}, {src.toIdent}, {r}{fillStr});\n"
    | .LowerFill r => s!"{indent}lower_fill({dst.toIdent}, {src.toIdent}, {r}{fillStr});\n"
    | .UpperRightFill r c => s!"{indent}upper_right_fill({dst.toIdent}, {src.toIdent}, {r}, {c}{fillStr});\n"

  -- Tile slicing
  | .sliceRows dst src startRow numRows =>
    s!"{indent}subtile<{startRow}, 0, {numRows}>({dst.toIdent}, {src.toIdent});\n"
  | .sliceCols dst src startCol numCols =>
    s!"{indent}subtile<0, {startCol}, -1, {numCols}>({dst.toIdent}, {src.toIdent});\n"

  -- Synchronization
  | .sync barrierId => s!"{indent}sync({barrierId});\n"
  | .arrive barrierId => s!"{indent}arrive({barrierId});\n"
  | .arriveAndWait barrierId => s!"{indent}arrive_and_wait({barrierId});\n"

  -- Named barriers (for FA3 warp specialization)
  | .namedBarrierSync id numThreads =>
    s!"{indent}kittens::named_barrier_sync<{id}, {numThreads}>();\n"
  | .namedBarrierArrive id numThreads =>
    s!"{indent}kittens::named_barrier_arrive<{id}, {numThreads}>();\n"

  -- Warp group operations (for FA3 warp specialization)
  | .warpGroupIdx dst =>
    s!"{indent}int {dst.toIdent} = kittens::warpgroup::warpgroup_idx();\n"
  | .electOneSync dst =>
    s!"{indent}bool {dst.toIdent} = kittens::elect_one_sync();\n"

  -- Fence operations (for WGMMA pipelining)
  | .fenceViewAsyncShared =>
    s!"{indent}__syncwarp();\n{indent}__fence_view_async_shared();\n"
  | .fenceProxyAsync =>
    s!"{indent}__fence_proxy_async();\n"

  -- Semaphore operations
  | .semaphore op sem =>
    match op with
    | .Init count => s!"{indent}init_semaphore({sem.toIdent}, {count});\n"
    | .Invalidate => s!"{indent}invalidate_semaphore({sem.toIdent});\n"
    | .Expect bytes => s!"{indent}expect({sem.toIdent}, {bytes});\n"
    | .Wait => s!"{indent}wait({sem.toIdent});\n"
    | .Arrive count => s!"{indent}arrive({sem.toIdent}, {count});\n"
    | .ArriveAndWait => s!"{indent}arrive_and_wait({sem.toIdent});\n"

  -- Control flow
  | .forLoop v lo hi body =>
    let bodyStr := body.toList.map (generateStmt (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {lo}; {v.toIdent} < {hi}; {v.toIdent}++) \{\n{bodyStr}{indent}}\n"
  | .ifStmt cond thenBody elseBody =>
    let thenStr := thenBody.toList.map (generateStmt (indent ++ "  ")) |>.foldl (· ++ ·) ""
    let elseStr := elseBody.toList.map (generateStmt (indent ++ "  ")) |>.foldl (· ++ ·) ""
    if elseBody.isEmpty then
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}}\n"
    else
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}} else \{\n{elseStr}{indent}}\n"
  | .ifWarpGroup wgIdx body =>
    let bodyStr := body.toList.map (generateStmt (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}if (kittens::warpgroup::warpgroup_idx() == {wgIdx}) \{\n{bodyStr}{indent}}\n"
  | .comment text => s!"{indent}// {text}\n"

/-- Generate kernel parameter list -/
def generateParams (params : Array KParam) : String :=
  let paramStrs := params.toList.map fun p =>
    if p.isPointer then s!"{p.dtype.toCpp}* {p.name}" else s!"{p.dtype.toCpp} {p.name}"
  String.intercalate ", " paramStrs

/-- Generate full kernel C++ code -/
def generateKernel (k : Kernel) : String :=
  let header := "#include <kittens.cuh>\nusing namespace kittens;\n\n"
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let body := k.body.toList.map (generateStmt "  ") |>.foldl (· ++ ·) ""
  let footer := "}\n#endif\n"
  header ++ archGuard ++ signature ++ body ++ footer

/-- Generate kernel with extern shared memory declaration -/
def generateKernelWithShared (k : Kernel) : String :=
  let header := "#include <kittens.cuh>\nusing namespace kittens;\n\n"
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let sharedDecl := if k.sharedMemBytes > 0
    then s!"  extern __shared__ char smem[{k.sharedMemBytes}];\n"
    else ""
  let body := k.body.toList.map (generateStmt "  ") |>.foldl (· ++ ·) ""
  let footer := "}\n#endif\n"
  header ++ archGuard ++ signature ++ sharedDecl ++ body ++ footer

/-- Generate CUDA launch configuration -/
structure LaunchCfg where
  gridDim : Nat × Nat × Nat := (1, 1, 1)
  blockDim : Nat × Nat × Nat := (128, 1, 1)
  sharedMem : Nat := 0
  deriving Repr, Inhabited

/-- Generate CUDA launch code -/
def generateLaunch (k : Kernel) (cfg : LaunchCfg) (args : List String) : String :=
  let (gx, gy, gz) := cfg.gridDim
  let (bx, byy, bz) := cfg.blockDim
  let argStr := String.intercalate ", " args
  s!"{k.name}<<<dim3({gx}, {gy}, {gz}), dim3({bx}, {byy}, {bz}), {cfg.sharedMem}>>>({argStr});\n"

/-- Write kernel to file -/
def writeKernelFile (k : Kernel) (path : String) : IO Unit := do
  let code := generateKernel k
  IO.FS.writeFile path code

end Tyr.GPU.Codegen
