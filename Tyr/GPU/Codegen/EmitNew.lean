/-
  Tyr/GPU/Codegen/EmitNew.lean

  C++ code generation from VarId-based kernel IR.
  Produces ThunderKittens-compatible CUDA code.
-/
import Std.Data.HashMap
import Std.Data.HashSet
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.AST

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Register vector layout for ThunderKittens row/col operations. -/
inductive RVLayout where
  | Naive
  | Align
  | Ortho
  deriving Repr, BEq, Hashable, Inhabited

private def RVLayout.toCpp : RVLayout → String
  | .Naive => "ducks::rv_layout::naive"
  | .Align => "ducks::rv_layout::align"
  | .Ortho => "ducks::rv_layout::ortho"

private def rowVecLayout : TileLayout → RVLayout
  | .Row => .Align
  | .Col => .Ortho

private def colVecLayout : TileLayout → RVLayout
  | .Row => .Ortho
  | .Col => .Align

structure RVLayoutState where
  layouts : Std.HashMap VarId RVLayout := {}
  conflicts : Std.HashSet VarId := {}

private def addRvLayout (st : RVLayoutState) (v : VarId) (layout : RVLayout) : RVLayoutState :=
  match st.layouts[v]? with
  | none => { st with layouts := st.layouts.insert v layout }
  | some existing =>
      if existing == layout then st
      else
        { layouts := st.layouts, conflicts := st.conflicts.insert v }

private def collectRtLayoutsStmt (acc : Std.HashMap VarId TileLayout) : KStmt → Std.HashMap VarId TileLayout
  | .declRT v _ _ _ layout => acc.insert v layout
  | .forLoop _ _ _ body => body.foldl collectRtLayoutsStmt acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl collectRtLayoutsStmt acc
      elseBody.foldl collectRtLayoutsStmt acc'
  | .ifWarpGroup _ body => body.foldl collectRtLayoutsStmt acc
  | _ => acc

private def rtLayoutOf (rtLayouts : Std.HashMap VarId TileLayout) (v : VarId) : TileLayout :=
  match rtLayouts[v]? with
  | some layout => layout
  | none => .Row

private def collectRvLayoutsStmt (rtLayouts : Std.HashMap VarId TileLayout) (st : RVLayoutState) :
    KStmt → RVLayoutState
  | .reduce _ axis dst src =>
      let tileLayout := rtLayoutOf rtLayouts src
      let layout := match axis with
        | .Row => colVecLayout tileLayout
        | .Col => rowVecLayout tileLayout
        | .Full => .Naive  -- Full reduction doesn't need specific layout
      addRvLayout st dst layout
  | .reduceAccum _ axis dst src accum =>
      let tileLayout := rtLayoutOf rtLayouts src
      let layout := match axis with
        | .Row => colVecLayout tileLayout
        | .Col => rowVecLayout tileLayout
        | .Full => .Naive  -- Full reduction doesn't need specific layout
      let st' := addRvLayout st dst layout
      addRvLayout st' accum layout
  | .broadcast axis dst vec =>
      let tileLayout := rtLayoutOf rtLayouts dst
      let layout := match axis with
        | .Row => rowVecLayout tileLayout
        | .Col => colVecLayout tileLayout
      addRvLayout st vec layout
  | .binaryBroadcast _ axis _ tile vec =>
      let tileLayout := rtLayoutOf rtLayouts tile
      let layout := match axis with
        | .Row => rowVecLayout tileLayout
        | .Col => colVecLayout tileLayout
      addRvLayout st vec layout
  | .outer dst a b =>
      let tileLayout := rtLayoutOf rtLayouts dst
      let st' := addRvLayout st a (colVecLayout tileLayout)
      addRvLayout st' b (rowVecLayout tileLayout)
  | .forLoop _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts) st
  | .ifStmt _ thenBody elseBody =>
      let st' := thenBody.foldl (collectRvLayoutsStmt rtLayouts) st
      elseBody.foldl (collectRvLayoutsStmt rtLayouts) st'
  | .ifWarpGroup _ body => body.foldl (collectRvLayoutsStmt rtLayouts) st
  | _ => st

private def inferRvLayouts (k : Kernel) : RVLayoutState :=
  let rtLayouts := k.body.foldl collectRtLayoutsStmt {}
  k.body.foldl (collectRvLayoutsStmt rtLayouts) {}

private def rvLayoutSuffix (rvLayouts : Std.HashMap VarId RVLayout) (v : VarId) : String :=
  match rvLayouts[v]? with
  | none => ""
  | some layout => s!", {layout.toCpp}"

private def layoutDiagnostics (conflicts : Std.HashSet VarId) : String :=
  conflicts.toList.foldl (fun acc v =>
    acc ++ s!"static_assert(false, \"RV layout conflict for {v.toIdent}\");\n") ""

/-- Generate C++ for a single statement -/
partial def generateStmt (rvLayouts : Std.HashMap VarId RVLayout) (indent : String := "  ") : KStmt → String
  -- Declarations
  | .declRT v dtype rows cols layout =>
    s!"{indent}rt<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {v.toIdent};\n"
  | .declST v dtype rows cols layout =>
    s!"{indent}__shared__ st<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {v.toIdent};\n"
  | .declRV v dtype len =>
    s!"{indent}rv<{dtype.toCpp}, {len}{rvLayoutSuffix rvLayouts v}> {v.toIdent};\n"
  | .declSV v dtype len =>
    s!"{indent}__shared__ sv<{dtype.toCpp}, {len}> {v.toIdent};\n"
  | .declSemaphore v =>
    s!"{indent}__shared__ semaphore {v.toIdent};\n"

  -- Kernel parameter declarations (these are part of the signature, not body)
  -- When they appear in the body, just emit a comment for debugging
  | .declGPtr v dtype name =>
    s!"{indent}// param: {dtype.toCpp}* {name} (v{v.idx})\n"
  | .declKVal v dtype name =>
    s!"{indent}// param: {dtype.toCpp} {name} (v{v.idx})\n"

  -- Memory operations
  | .load dst src => s!"{indent}warp::load({dst.toIdent}, {src.toIdent});\n"
  | .store dst src => s!"{indent}warp::store({dst.toIdent}, {src.toIdent});\n"
  | .loadAsync dst src => s!"{indent}warp::load_async({dst.toIdent}, {src.toIdent});\n"
  | .storeAsync dst src => s!"{indent}warp::store_async({dst.toIdent}, {src.toIdent});\n"
  | .storeAdd dst src => s!"{indent}store_add({dst.toIdent}, {src.toIdent});\n"
  | .storeAddAsync dst src => s!"{indent}warp::tma::store_add_async({dst.toIdent}, {src.toIdent});\n"
  | .storeMinAsync dst src => s!"{indent}warp::tma::store_min_async({dst.toIdent}, {src.toIdent});\n"
  | .prefetch src => s!"{indent}warp::tma::prefetch({src.toIdent});\n"
  | .tmaExpect barrier bytes => s!"{indent}warp::tma::expect_bytes({barrier.toIdent}, {bytes});\n"

  -- TMA operations with global pointers
  | .tmaLoad dst src coord =>
    s!"{indent}tma_load({dst.toIdent}, {src.toIdent}, {coord.toIdent});\n"
  | .tmaStore dst src coord =>
    s!"{indent}tma_store({dst.toIdent}, {src.toIdent}, {coord.toIdent});\n"

  -- Global memory operations with 4D coordinates (ThunderKittens style)
  | .loadGlobal dst src coordB coordD coordR coordC =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}});\n"
  | .storeGlobal dst src coordB coordD coordR coordC =>
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}});\n"
  | .loadGlobalAsync dst src coordB coordD coordR coordC sem =>
    s!"{indent}warp::tma::load_async({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}}, {sem.toIdent});\n"
  | .storeGlobalAsync dst src coordB coordD coordR coordC =>
    s!"{indent}warp::tma::store_async({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}});\n"
  | .storeGlobalAdd dst src coordB coordD coordR coordC =>
    s!"{indent}warp::tma::store_add_async({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}});\n"

  -- Vector global memory operations
  | .loadVecGlobal dst src offset =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, {offset.toIdent});\n"
  | .storeVecGlobal dst src offset =>
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, {offset.toIdent});\n"
  | .storeVecGlobalAdd dst src offset =>
    s!"{indent}store_add({dst.toIdent}, {src.toIdent}, {offset.toIdent});\n"

  -- Distributed / Multimem operations
  | .multimemLoadReduce op dst src =>
    s!"{indent}multimem::load_reduce_{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
  | .multimemStore dst src =>
    s!"{indent}multimem::store({dst.toIdent}, {src.toIdent});\n"
  | .multimemRed op dst src =>
    s!"{indent}multimem::reduce_{op.toCpp}({dst.toIdent}, {src.toIdent});\n"

  -- MMA operations
  | .mma trans dst a b c =>
    s!"{indent}warp::mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"
  | .mm trans dst a b =>
    s!"{indent}warp::mm_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
  | .mmaFence dst => s!"{indent}warpgroup::mma_fence({dst.toIdent});\n"
  | .mmaCommitGroup => s!"{indent}warpgroup::mma_commit_group();\n"
  | .mmaAsyncWait n => s!"{indent}warpgroup::mma_async_wait<{n}>();\n"

  -- Blackwell-specific MMA (tcgen05 / 2-CTA MMA)
  | .tcgen05Mma trans dst a b c =>
    s!"{indent}warpgroup::mma_{trans.toSuffix}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"

  -- Architecture-specific load variants
  | .cpAsyncLoad dst src coordB coordD coordR coordC _sem =>
    s!"{indent}warp::load_async({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}});\n"
  | .tmaLoadAsync dst src coordB coordD coordR coordC sem =>
    s!"{indent}warp::tma::load_async({dst.toIdent}, {src.toIdent}, \{.b={coordB.toIdent}, .d={coordD.toIdent}, .r={coordR.toIdent}, .c={coordC.toIdent}}, {sem.toIdent});\n"

  -- Element-wise unary
  | .unary op dst src => s!"{indent}warp::{op.toCpp}({dst.toIdent}, {src.toIdent});\n"

  -- Element-wise binary
  | .binary op dst a b =>
    s!"{indent}warp::{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"

  -- Element-wise ternary (FMA)
  | .ternary op dst a b c =>
    s!"{indent}warp::{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"

  -- Scalar operations
  | .scalarMul dst src scalar =>
    s!"{indent}warp::mul({dst.toIdent}, {src.toIdent}, {scalar}f);\n"
  | .scalarAdd dst src scalar =>
    s!"{indent}warp::add({dst.toIdent}, {src.toIdent}, {scalar}f);\n"

  -- Broadcasting
  | .broadcast axis dst vec =>
    let suffix := match axis with
      | .Row => "_col"
      | .Col => "_row"
    s!"{indent}warp::broadcast{suffix}({dst.toIdent}, {vec.toIdent});\n"
  | .binaryBroadcast op axis dst tile vec =>
    let suffix := match axis with
      | .Row => "_col"
      | .Col => "_row"
    s!"{indent}warp::{op.toCpp}{suffix}({dst.toIdent}, {tile.toIdent}, {vec.toIdent});\n"

  -- Reductions
  | .reduce op axis dst src =>
    s!"{indent}warp::{axis.toPrefix}{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
  | .reduceAccum op axis dst src accum =>
    s!"{indent}warp::{axis.toPrefix}{op.toCpp}({dst.toIdent}, {src.toIdent}, {accum.toIdent});\n"

  -- Scan operations
  | .cumsum axis dst src =>
    s!"{indent}warp::{axis.toPrefix}cumsum({dst.toIdent}, {src.toIdent});\n"
  | .cumprod axis dst src =>
    s!"{indent}warp::{axis.toPrefix}cumprod({dst.toIdent}, {src.toIdent});\n"

  -- Outer product
  | .outer dst a b =>
    s!"{indent}tk_outer({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"

  -- Layout/type conversions
  | .swapLayout dst src => s!"{indent}warp::swap_layout({dst.toIdent}, {src.toIdent});\n"
  | .transpose dst src => s!"{indent}warp::transpose({dst.toIdent}, {src.toIdent});\n"
  | .convert dst src => s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"

  -- Masking
  | .mask op dst src fillVal =>
    let fillStr := fillVal.map (fun v => s!", {v}") |>.getD ""
    match op with
    | .Tril d => s!"{indent}warp::tril({dst.toIdent}, {src.toIdent}, {d}{fillStr});\n"
    | .Triu d => s!"{indent}warp::triu({dst.toIdent}, {src.toIdent}, {d}{fillStr});\n"
    | .MakeCausal => s!"{indent}warp::make_causal({dst.toIdent}, {src.toIdent}{fillStr});\n"
    | .MakeCausalT => s!"{indent}warp::make_causal_t({dst.toIdent}, {src.toIdent}{fillStr});\n"
    | .RightFill c => s!"{indent}warp::right_fill({dst.toIdent}, {src.toIdent}, {c}{fillStr});\n"
    | .LeftFill c => s!"{indent}warp::left_fill({dst.toIdent}, {src.toIdent}, {c}{fillStr});\n"
    | .UpperFill r => s!"{indent}warp::upper_fill({dst.toIdent}, {src.toIdent}, {r}{fillStr});\n"
    | .LowerFill r => s!"{indent}warp::lower_fill({dst.toIdent}, {src.toIdent}, {r}{fillStr});\n"
    | .UpperRightFill r c => s!"{indent}warp::upper_right_fill({dst.toIdent}, {src.toIdent}, {r}, {c}{fillStr});\n"

  -- Tile slicing
  | .sliceRows dst src startRow numRows =>
    s!"{indent}tk_slice_rows<{startRow}, {numRows}>({dst.toIdent}, {src.toIdent});\n"
  | .sliceCols dst src startCol numCols =>
    s!"{indent}tk_slice_cols<{startCol}, {numCols}>({dst.toIdent}, {src.toIdent});\n"

  -- Synchronization
  | .sync barrierId => s!"{indent}warp::sync({barrierId});\n"
  | .arrive barrierId => s!"{indent}warp::arrive({barrierId});\n"
  | .arriveAndWait barrierId => s!"{indent}warp::arrive_and_wait(kittens::barrier<1>({barrierId}));\n"

  -- Named barriers (for FA3 warp specialization)
  | .namedBarrierSync id numThreads =>
    let numWarps := numThreads / 32
    s!"{indent}kittens::arrive_and_wait(kittens::barrier<{numWarps}>({id}));\n"
  | .namedBarrierArrive id numThreads =>
    let numWarps := numThreads / 32
    s!"{indent}kittens::arrive(kittens::barrier<{numWarps}>({id}));\n"

  -- Warp group operations (for FA3 warp specialization)
  | .warpGroupIdx dst =>
    s!"{indent}int {dst.toIdent} = kittens::warpgroup::groupid();\n"
  | .electOneSync dst =>
    s!"{indent}bool {dst.toIdent} = (kittens::laneid() == (__ffs(__activemask()) - 1));\n"

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
    let bodyStr := body.toList.map (generateStmt rvLayouts (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {lo}; {v.toIdent} < {hi}; {v.toIdent}++) \{\n{bodyStr}{indent}}\n"
  | .ifStmt cond thenBody elseBody =>
    let thenStr := thenBody.toList.map (generateStmt rvLayouts (indent ++ "  ")) |>.foldl (· ++ ·) ""
    let elseStr := elseBody.toList.map (generateStmt rvLayouts (indent ++ "  ")) |>.foldl (· ++ ·) ""
    if elseBody.isEmpty then
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}}\n"
    else
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}} else \{\n{elseStr}{indent}}\n"
  | .ifWarpGroup wgIdx body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}if (kittens::warpgroup::groupid() == {wgIdx}) \{\n{bodyStr}{indent}}\n"
  | .comment text => s!"{indent}// {text}\n"

  -- Block/thread index accessors
  | .getBlockIdx dst axis =>
    let axisName := match axis with | 0 => "x" | 1 => "y" | _ => "z"
    s!"{indent}int {dst.toIdent} = blockIdx.{axisName};\n"
  | .getThreadIdx dst axis =>
    let axisName := match axis with | 0 => "x" | 1 => "y" | _ => "z"
    s!"{indent}int {dst.toIdent} = threadIdx.{axisName};\n"

  -- Constants
  | .constInt dst value =>
    s!"{indent}int {dst.toIdent} = {value};\n"

/-- Generate kernel parameter list -/
def generateParams (params : Array KParam) : String :=
  let paramStrs := params.toList.map fun p =>
    if p.isPointer then s!"{p.dtype.toCpp}* {p.name}" else s!"{p.dtype.toCpp} {p.name}"
  String.intercalate ", " paramStrs

private partial def stmtUses (p : KStmt → Bool) : KStmt → Bool
  | stmt =>
    if p stmt then true else
      match stmt with
      | .forLoop _ _ _ body => body.any (stmtUses p)
      | .ifStmt _ thenBody elseBody =>
        thenBody.any (stmtUses p) || elseBody.any (stmtUses p)
      | .ifWarpGroup _ body => body.any (stmtUses p)
      | _ => false

private def bodyUses (p : KStmt → Bool) (body : Array KStmt) : Bool :=
  body.any (stmtUses p)

private def usesStoreAdd (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .storeAdd .. | .storeVecGlobalAdd .. => true
    | _ => false) k.body

private def usesLegacyTma (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .tmaLoad .. | .tmaStore .. => true
    | _ => false) k.body

private def usesSlice (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .sliceRows .. | .sliceCols .. => true
    | _ => false) k.body

private def usesOuter (k : Kernel) : Bool :=
  bodyUses (fun s => match s with
    | .outer .. => true
    | _ => false) k.body

private def storeAddHelpers : String :=
  "template<typename Dst, typename Src>\n" ++
  "__device__ inline void store_add(Dst &dst, const Src &src) {\n" ++
  "  kittens::warp::store(dst, src);\n" ++
  "}\n" ++
  "template<typename Dst, typename Src, typename Offset>\n" ++
  "__device__ inline void store_add(Dst &dst, const Src &src, const Offset &offset) {\n" ++
  "  kittens::warp::store(dst, src, offset);\n" ++
  "}\n\n"

private def legacyTmaHelpers : String :=
  "template<typename ST, typename GL, typename CoordT>\n" ++
  "__device__ inline void tma_load(ST &dst, const GL &src, CoordT coord) {\n" ++
  "  kittens::coord<> idx(static_cast<int>(coord));\n" ++
  "  kittens::warp::load(dst, src, idx);\n" ++
  "}\n" ++
  "template<typename GL, typename ST, typename CoordT>\n" ++
  "__device__ inline void tma_store(GL &dst, const ST &src, CoordT coord) {\n" ++
  "  kittens::coord<> idx(static_cast<int>(coord));\n" ++
  "  kittens::warp::store(dst, src, idx);\n" ++
  "}\n\n"

private def sliceHelpers : String :=
  "template<int START_ROW, int NUM_ROWS, typename DST, typename SRC>\n" ++
  "__device__ inline void tk_slice_rows(DST &dst, const SRC &src) {\n" ++
  "  if constexpr (kittens::ducks::rt::all<DST> && kittens::ducks::rt::all<SRC>) {\n" ++
  "    using T = typename DST::T;\n" ++
  "    constexpr int tile_rows = kittens::TILE_ROW_DIM<T>;\n" ++
  "    static_assert(DST::rows == NUM_ROWS, \"slice rows: dst rows mismatch\");\n" ++
  "    static_assert(START_ROW % tile_rows == 0, \"slice rows: unaligned start\");\n" ++
  "    static_assert(NUM_ROWS % tile_rows == 0, \"slice rows: unaligned size\");\n" ++
  "    constexpr int start_tile = START_ROW / tile_rows;\n" ++
  "    constexpr int row_tiles = NUM_ROWS / tile_rows;\n" ++
  "    #pragma unroll\n" ++
  "    for (int i = 0; i < row_tiles; i++) {\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < DST::width; j++) {\n" ++
  "        dst.tiles[i][j] = src.tiles[start_tile + i][j];\n" ++
  "      }\n" ++
  "    }\n" ++
  "  } else if constexpr (kittens::ducks::st::all<DST> && kittens::ducks::st::all<SRC>) {\n" ++
  "    static_assert(DST::rows == NUM_ROWS, \"slice rows: dst rows mismatch\");\n" ++
  "    auto src_sub = src.template subtile<NUM_ROWS, SRC::cols>(int2{START_ROW, 0});\n" ++
  "    kittens::warp::copy(dst, src_sub);\n" ++
  "  }\n" ++
  "}\n\n" ++
  "template<int START_COL, int NUM_COLS, typename DST, typename SRC>\n" ++
  "__device__ inline void tk_slice_cols(DST &dst, const SRC &src) {\n" ++
  "  if constexpr (kittens::ducks::rt::all<DST> && kittens::ducks::rt::all<SRC>) {\n" ++
  "    using T = typename DST::T;\n" ++
  "    constexpr int tile_cols = kittens::TILE_COL_DIM<T>;\n" ++
  "    static_assert(DST::cols == NUM_COLS, \"slice cols: dst cols mismatch\");\n" ++
  "    static_assert(START_COL % tile_cols == 0, \"slice cols: unaligned start\");\n" ++
  "    static_assert(NUM_COLS % tile_cols == 0, \"slice cols: unaligned size\");\n" ++
  "    constexpr int start_tile = START_COL / tile_cols;\n" ++
  "    constexpr int col_tiles = NUM_COLS / tile_cols;\n" ++
  "    #pragma unroll\n" ++
  "    for (int i = 0; i < DST::height; i++) {\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < col_tiles; j++) {\n" ++
  "        dst.tiles[i][j] = src.tiles[i][start_tile + j];\n" ++
  "      }\n" ++
  "    }\n" ++
  "  } else if constexpr (kittens::ducks::st::all<DST> && kittens::ducks::st::all<SRC>) {\n" ++
  "    static_assert(DST::cols == NUM_COLS, \"slice cols: dst cols mismatch\");\n" ++
  "    auto src_sub = src.template subtile<SRC::rows, NUM_COLS>(int2{0, START_COL});\n" ++
  "    kittens::warp::copy(dst, src_sub);\n" ++
  "  }\n" ++
  "}\n\n"

private def outerHelpers : String :=
  "template<typename RT, typename RVRow, typename RVCol>\n" ++
  "__device__ inline void tk_outer(RT &dst, const RVRow &row_vals, const RVCol &col_vals) {\n" ++
  "  RT row_tile;\n" ++
  "  RT col_tile;\n" ++
  "  kittens::warp::broadcast_row(row_tile, row_vals);\n" ++
  "  kittens::warp::broadcast_col(col_tile, col_vals);\n" ++
  "  kittens::warp::mul(dst, row_tile, col_tile);\n" ++
  "}\n\n"

private def generateHelpers (k : Kernel) : String := Id.run do
  let mut helpers := ""
  if usesStoreAdd k then
    helpers := helpers ++ storeAddHelpers
  if usesLegacyTma k then
    helpers := helpers ++ legacyTmaHelpers
  if usesSlice k then
    helpers := helpers ++ sliceHelpers
  if usesOuter k then
    helpers := helpers ++ outerHelpers
  return helpers

/-- Generate full kernel C++ code -/
def generateKernel (k : Kernel) : String :=
  let rvState := inferRvLayouts k
  let header :=
    "#include <kittens.cuh>\nusing namespace kittens;\n\n" ++
    generateHelpers k ++ layoutDiagnostics rvState.conflicts
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let body := k.body.toList.map (generateStmt rvState.layouts "  ") |>.foldl (· ++ ·) ""
  let footer := "}\n#endif\n"
  header ++ archGuard ++ signature ++ body ++ footer

/-- Generate kernel with extern shared memory declaration -/
def generateKernelWithShared (k : Kernel) : String :=
  let rvState := inferRvLayouts k
  let header :=
    "#include <kittens.cuh>\nusing namespace kittens;\n\n" ++
    generateHelpers k ++ layoutDiagnostics rvState.conflicts
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let sharedDecl := if k.sharedMemBytes > 0
    then s!"  extern __shared__ char smem[{k.sharedMemBytes}];\n"
    else ""
  let body := k.body.toList.map (generateStmt rvState.layouts "  ") |>.foldl (· ++ ·) ""
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
