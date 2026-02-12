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

private inductive TileKind where
  | RT
  | ST
  deriving Repr, BEq, Hashable, Inhabited

private structure TileInfo where
  kind : TileKind
  rows : Nat
  cols : Nat
  deriving Repr, BEq, Inhabited

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

private def collectTileInfoStmt (acc : Std.HashMap VarId TileInfo) : KStmt → Std.HashMap VarId TileInfo
  | .declRT v _ rows cols _ => acc.insert v { kind := .RT, rows := rows, cols := cols }
  | .declST v _ rows cols _ => acc.insert v { kind := .ST, rows := rows, cols := cols }
  | .forLoop _ _ _ body => body.foldl collectTileInfoStmt acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl collectTileInfoStmt acc
      elseBody.foldl collectTileInfoStmt acc'
  | .ifWarpGroup _ body => body.foldl collectTileInfoStmt acc
  | _ => acc

private def collectRvDeclsStmt (acc : Std.HashSet VarId) : KStmt → Std.HashSet VarId
  | .declRV v _ _ => acc.insert v
  | .forLoop _ _ _ body => body.foldl collectRvDeclsStmt acc
  | .ifStmt _ thenBody elseBody =>
      let acc' := thenBody.foldl collectRvDeclsStmt acc
      elseBody.foldl collectRvDeclsStmt acc'
  | .ifWarpGroup _ body => body.foldl collectRvDeclsStmt acc
  | _ => acc

private def rtLayoutOf (rtLayouts : Std.HashMap VarId TileLayout) (v : VarId) : TileLayout :=
  match rtLayouts[v]? with
  | some layout => layout
  | none => .Row

private def unifyRvVars (rvVars : Std.HashSet VarId) (st : RVLayoutState)
    (a b : VarId) : RVLayoutState :=
  if !(rvVars.contains a && rvVars.contains b) then
    st
  else
    match st.layouts[a]?, st.layouts[b]? with
    | none, none => st
    | some la, none => addRvLayout st b la
    | none, some lb => addRvLayout st a lb
    | some la, some lb =>
      if la == lb then st
      else
        { st with conflicts := (st.conflicts.insert a).insert b }

private def collectRvLayoutsStmt
    (rtLayouts : Std.HashMap VarId TileLayout)
    (rvVars : Std.HashSet VarId)
    (st : RVLayoutState) :
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
  | .unary _ dst src =>
      unifyRvVars rvVars st dst src
  | .binary _ dst a b =>
      let st' := unifyRvVars rvVars st dst a
      unifyRvVars rvVars st' dst b
  | .ternary _ dst a b c =>
      let st1 := unifyRvVars rvVars st dst a
      let st2 := unifyRvVars rvVars st1 dst b
      unifyRvVars rvVars st2 dst c
  | .scalarMul dst src _ =>
      unifyRvVars rvVars st dst src
  | .scalarAdd dst src _ =>
      unifyRvVars rvVars st dst src
  | .forLoop _ _ _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | .ifStmt _ thenBody elseBody =>
      let st' := thenBody.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
      elseBody.foldl (collectRvLayoutsStmt rtLayouts rvVars) st'
  | .ifWarpGroup _ body => body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
  | _ => st

private def inferRvDecls (k : Kernel) : Std.HashSet VarId :=
  k.body.foldl collectRvDeclsStmt {}

private def inferRvLayouts (k : Kernel) : RVLayoutState :=
  let rtLayouts := k.body.foldl collectRtLayoutsStmt {}
  let rvVars := inferRvDecls k
  Id.run do
    let mut st : RVLayoutState := {}
    -- Fixed-point approximation for layout propagation along op chains.
    for _ in [:16] do
      st := k.body.foldl (collectRvLayoutsStmt rtLayouts rvVars) st
    return st

private def rvLayoutSuffix (rvLayouts : Std.HashMap VarId RVLayout) (v : VarId) : String :=
  match rvLayouts[v]? with
  | none => ""
  | some layout => s!", {layout.toCpp}"

private def layoutDiagnostics (conflicts : Std.HashSet VarId) : String :=
  conflicts.toList.foldl (fun acc v =>
    acc ++ s!"static_assert(false, \"RV layout conflict for {v.toIdent}\");\n") ""

private def inferTileInfo (k : Kernel) : Std.HashMap VarId TileInfo :=
  k.body.foldl collectTileInfoStmt {}

/-- Generate C++ for a single statement -/
partial def generateStmt (rvLayouts : Std.HashMap VarId RVLayout)
    (rvVars : Std.HashSet VarId)
    (tileInfo : Std.HashMap VarId TileInfo) (indent : String := "  ") : KStmt → String
  -- Declarations
  | .declRT v dtype rows cols layout =>
    s!"{indent}rt<{dtype.toCpp}, {rows}, {cols}, {layout.toCpp}> {v.toIdent};\n"
  | .declST v dtype rows cols layout =>
    -- ThunderKittens shared tiles do not carry a row/col layout parameter.
    let _ := layout
    s!"{indent}__shared__ st<{dtype.toCpp}, {rows}, {cols}> {v.toIdent};\n"
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
    let (rowScale, colScale) := match tileInfo[dst]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})));\n"
  | .storeGlobal dst src coordB coordD coordR coordC =>
    let (rowScale, colScale) := match tileInfo[src]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})));\n"
  | .loadGlobalAsync dst src coordB coordD coordR coordC sem =>
    let (rowScale, colScale) := match tileInfo[dst]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}warp::tma::load_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})), {sem.toIdent});\n"
  | .storeGlobalAsync dst src coordB coordD coordR coordC =>
    let (rowScale, colScale) := match tileInfo[src]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}warp::tma::store_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})));\n"
  | .storeGlobalAdd dst src coordB coordD coordR coordC =>
    let (rowScale, colScale) := match tileInfo[src]? with
      | some info => (info.rows, info.cols)
      | none => (1, 1)
    s!"{indent}warp::tma::store_add_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, ({coordR.toIdent} * {rowScale}), ({coordC.toIdent} * {colScale})));\n"

  -- Vector global memory operations
  | .loadVecGlobal dst src offset =>
    s!"{indent}warp::load({dst.toIdent}, {src.toIdent}, " ++
    "{" ++ s!"{offset.toIdent}" ++ "});\n"
  | .storeVecGlobal dst src offset =>
    s!"{indent}warp::store({dst.toIdent}, {src.toIdent}, " ++
    "{" ++ s!"{offset.toIdent}" ++ "});\n"
  | .storeVecGlobalAdd dst src offset =>
    s!"{indent}store_add({dst.toIdent}, {src.toIdent}, " ++
    "{" ++ s!"{offset.toIdent}" ++ "});\n"

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
    s!"{indent}warp::load_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}));\n"
  | .tmaLoadAsync dst src coordB coordD coordR coordC sem =>
    s!"{indent}warp::tma::load_async({dst.toIdent}, {src.toIdent}, kittens::coord<>({coordB.toIdent}, {coordD.toIdent}, {coordR.toIdent}, {coordC.toIdent}), {sem.toIdent});\n"

  -- Element-wise unary
  | .unary .Neg dst src =>
    if dst == src then
      s!"{indent}warp::mul({dst.toIdent}, {src.toIdent}, -1.0f);\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::mul({dst.toIdent}, {dst.toIdent}, -1.0f);\n"
  | .unary .Zero dst _ =>
    s!"{indent}warp::zero({dst.toIdent});\n"
  | .unary .One dst _ =>
    s!"{indent}warp::one({dst.toIdent});\n"
  | .unary .PosInfty dst _ =>
    s!"{indent}warp::pos_infty({dst.toIdent});\n"
  | .unary .NegInfty dst _ =>
    s!"{indent}warp::neg_infty({dst.toIdent});\n"
  | .unary .Sqrt dst src =>
    if rvVars.contains dst && rvVars.contains src then
      let pre :=
        if dst == src then
          ""
        else
          s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
      pre ++
      s!"{indent}warp::apply({dst.toIdent}, {dst.toIdent}, [] __device__ (int _i, auto _x) \{\n" ++
      s!"{indent}  return static_cast<decltype(_x)>(::sqrtf(static_cast<float>(_x)));\n" ++
      s!"{indent}" ++ "});\n"
    else if dst == src then
      s!"{indent}warp::sqrt({dst.toIdent}, {src.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::sqrt({dst.toIdent}, {dst.toIdent});\n"
  | .unary .Rsqrt dst src =>
    if rvVars.contains dst && rvVars.contains src then
      let pre :=
        if dst == src then
          ""
        else
          s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
      pre ++
      s!"{indent}warp::apply({dst.toIdent}, {dst.toIdent}, [] __device__ (int _i, auto _x) \{\n" ++
      s!"{indent}  return static_cast<decltype(_x)>(::rsqrtf(static_cast<float>(_x)));\n" ++
      s!"{indent}" ++ "});\n"
    else if dst == src then
      s!"{indent}warp::rsqrt({dst.toIdent}, {src.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::rsqrt({dst.toIdent}, {dst.toIdent});\n"
  | .unary .Copy dst src =>
    s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n"
  | .unary op dst src =>
    if dst == src then
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {src.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {dst.toIdent});\n"

  -- Element-wise binary
  | .binary op dst a b =>
    if dst == a then
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent});\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {a.toIdent});\n" ++
      s!"{indent}warp::{op.toCpp}({dst.toIdent}, {dst.toIdent}, {b.toIdent});\n"

  -- Element-wise ternary (FMA)
  | .ternary op dst a b c =>
    s!"{indent}warp::{op.toCpp}({dst.toIdent}, {a.toIdent}, {b.toIdent}, {c.toIdent});\n"

  -- Scalar operations
  | .scalarMul dst src scalar =>
    if dst == src then
      s!"{indent}warp::mul({dst.toIdent}, {src.toIdent}, {scalar}f);\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::mul({dst.toIdent}, {dst.toIdent}, {scalar}f);\n"
  | .scalarAdd dst src scalar =>
    if dst == src then
      s!"{indent}warp::add({dst.toIdent}, {src.toIdent}, {scalar}f);\n"
    else
      s!"{indent}warp::copy({dst.toIdent}, {src.toIdent});\n" ++
      s!"{indent}warp::add({dst.toIdent}, {dst.toIdent}, {scalar}f);\n"

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
  | .transpose dst src =>
    if dst == src then
      s!"{indent}warp::transpose_inplace({dst.toIdent});\n"
    else
      s!"{indent}warp::transpose_sep({dst.toIdent}, {src.toIdent});\n"
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
    match tileInfo[dst]?, tileInfo[src]? with
    | some dstInfo, some srcInfo =>
      match dstInfo.kind, srcInfo.kind with
      | .RT, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_row_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_row_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_row_t>;\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {numRows}, \"slice rows: dst rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {srcInfo.cols}, \"slice rows: src/dst cols mismatch\");\n" ++
        s!"{indent}static_assert({startRow} + {numRows} <= {srcInfo.rows}, \"slice rows: out of bounds\");\n" ++
        s!"{indent}static_assert({startRow} % _tk_tile_rows == 0, \"slice rows: unaligned start\");\n" ++
        s!"{indent}static_assert({numRows} % _tk_tile_rows == 0, \"slice rows: unaligned size\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} % _tk_tile_cols == 0, \"slice rows: unaligned cols\");\n" ++
        s!"{indent}constexpr int _tk_start_tile = {startRow} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {numRows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_col_tiles = {dstInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_col_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = {src.toIdent}.tiles[_tk_start_tile + _tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | .ST, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {numRows}, \"slice rows: dst rows mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.cols} == {dstInfo.cols}, \"slice rows: src/dst cols mismatch\");\n" ++
        s!"{indent}auto _tk_src_sub = {src.toIdent}.template subtile<{numRows}, {srcInfo.cols}>(make_int2({startRow}, 0));\n" ++
        s!"{indent}kittens::warp::copy({dst.toIdent}, _tk_src_sub);\n" ++
        s!"{indent}}\n"
      | _, _ =>
        s!"{indent}// unsupported sliceRows between non-matching tile kinds\n"
    | _, _ =>
      s!"{indent}// unresolved tile info for sliceRows\n"
  | .sliceCols dst src startCol numCols =>
    match tileInfo[dst]?, tileInfo[src]? with
    | some dstInfo, some srcInfo =>
      match dstInfo.kind, srcInfo.kind with
      | .RT, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_col_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_col_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_col_t>;\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {numCols}, \"slice cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {srcInfo.rows}, \"slice cols: src/dst rows mismatch\");\n" ++
        s!"{indent}static_assert({startCol} + {numCols} <= {srcInfo.cols}, \"slice cols: out of bounds\");\n" ++
        s!"{indent}static_assert({startCol} % _tk_tile_cols == 0, \"slice cols: unaligned start\");\n" ++
        s!"{indent}static_assert({numCols} % _tk_tile_cols == 0, \"slice cols: unaligned size\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} % _tk_tile_rows == 0, \"slice cols: unaligned rows\");\n" ++
        s!"{indent}constexpr int _tk_start_tile = {startCol} / _tk_tile_cols;\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {dstInfo.rows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_col_tiles = {numCols} / _tk_tile_cols;\n" ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_col_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = {src.toIdent}.tiles[_tk_i][_tk_start_tile + _tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | .ST, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {numCols}, \"slice cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({srcInfo.rows} == {dstInfo.rows}, \"slice cols: src/dst rows mismatch\");\n" ++
        s!"{indent}auto _tk_src_sub = {src.toIdent}.template subtile<{srcInfo.rows}, {numCols}>(make_int2(0, {startCol}));\n" ++
        s!"{indent}kittens::warp::copy({dst.toIdent}, _tk_src_sub);\n" ++
        s!"{indent}}\n"
      | _, _ =>
        s!"{indent}// unsupported sliceCols between non-matching tile kinds\n"
    | _, _ =>
      s!"{indent}// unresolved tile info for sliceCols\n"
  | .concatCols dst left right =>
    match tileInfo[dst]?, tileInfo[left]?, tileInfo[right]? with
    | some dstInfo, some leftInfo, some rightInfo =>
      match dstInfo.kind, leftInfo.kind, rightInfo.kind with
      | .RT, .RT, .RT =>
        s!"{indent}\{\n" ++
        s!"{indent}using _tk_concat_t = typename decltype({dst.toIdent})::T;\n" ++
        s!"{indent}constexpr int _tk_tile_rows = kittens::TILE_ROW_DIM<_tk_concat_t>;\n" ++
        s!"{indent}constexpr int _tk_tile_cols = kittens::TILE_COL_DIM<_tk_concat_t>;\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {leftInfo.rows}, \"concat cols: left rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {rightInfo.rows}, \"concat cols: right rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {leftInfo.cols} + {rightInfo.cols}, \"concat cols: dst cols mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} % _tk_tile_rows == 0, \"concat cols: unaligned rows\");\n" ++
        s!"{indent}static_assert({leftInfo.cols} % _tk_tile_cols == 0, \"concat cols: unaligned left cols\");\n" ++
        s!"{indent}static_assert({rightInfo.cols} % _tk_tile_cols == 0, \"concat cols: unaligned right cols\");\n" ++
        s!"{indent}constexpr int _tk_row_tiles = {dstInfo.rows} / _tk_tile_rows;\n" ++
        s!"{indent}constexpr int _tk_left_tiles = {leftInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}constexpr int _tk_right_tiles = {rightInfo.cols} / _tk_tile_cols;\n" ++
        s!"{indent}#pragma unroll\n" ++
        s!"{indent}for (int _tk_i = 0; _tk_i < _tk_row_tiles; _tk_i++) \{\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_left_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_j] = {left.toIdent}.tiles[_tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}  #pragma unroll\n" ++
        s!"{indent}  for (int _tk_j = 0; _tk_j < _tk_right_tiles; _tk_j++) \{\n" ++
        s!"{indent}    {dst.toIdent}.tiles[_tk_i][_tk_left_tiles + _tk_j] = {right.toIdent}.tiles[_tk_i][_tk_j];\n" ++
        s!"{indent}  }\n" ++
        s!"{indent}}\n" ++
        s!"{indent}}\n"
      | .ST, .ST, .ST =>
        s!"{indent}\{\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {leftInfo.rows}, \"concat cols: left rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.rows} == {rightInfo.rows}, \"concat cols: right rows mismatch\");\n" ++
        s!"{indent}static_assert({dstInfo.cols} == {leftInfo.cols} + {rightInfo.cols}, \"concat cols: dst cols mismatch\");\n" ++
        s!"{indent}auto _tk_dst_left = {dst.toIdent}.template subtile<{dstInfo.rows}, {leftInfo.cols}>(make_int2(0, 0));\n" ++
        s!"{indent}auto _tk_dst_right = {dst.toIdent}.template subtile<{dstInfo.rows}, {rightInfo.cols}>(make_int2(0, {leftInfo.cols}));\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_left, {left.toIdent});\n" ++
        s!"{indent}kittens::warp::copy(_tk_dst_right, {right.toIdent});\n" ++
        s!"{indent}}\n"
      | _, _, _ =>
        s!"{indent}// unsupported concatCols between non-matching tile kinds\n"
    | _, _, _ =>
      s!"{indent}// unresolved tile info for concatCols\n"

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
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo (indent ++ "  ")) |>.foldl (· ++ ·) ""
    s!"{indent}for (int {v.toIdent} = {lo}; {v.toIdent} < {hi}; {v.toIdent}++) \{\n{bodyStr}{indent}}\n"
  | .ifStmt cond thenBody elseBody =>
    let thenStr := thenBody.toList.map (generateStmt rvLayouts rvVars tileInfo (indent ++ "  ")) |>.foldl (· ++ ·) ""
    let elseStr := elseBody.toList.map (generateStmt rvLayouts rvVars tileInfo (indent ++ "  ")) |>.foldl (· ++ ·) ""
    if elseBody.isEmpty then
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}}\n"
    else
      s!"{indent}if ({cond.toIdent}) \{\n{thenStr}{indent}} else \{\n{elseStr}{indent}}\n"
  | .ifWarpGroup wgIdx body =>
    let bodyStr := body.toList.map (generateStmt rvLayouts rvVars tileInfo (indent ++ "  ")) |>.foldl (· ++ ·) ""
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
  let paramStrs := Id.run do
    let mut out : List String := []
    for h : idx in [:params.size] do
      let p := params[idx]
      if p.isPointer then
        -- Kernel-side global pointers are ThunderKittens gl descriptors.
        -- Start with a minimal 2D dynamic shape (b=1, d=1, rows/cols runtime).
        out := out.concat s!"gl<{p.dtype.toCpp}, 1, 1, -1, -1> v{idx}"
      else
        out := out.concat s!"{p.scalarTy.toCpp} v{idx}"
    return out
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
    | .sliceRows .. | .sliceCols .. | .concatCols .. => true
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
  "}\n\n" ++
  "template<typename DST, typename LEFT, typename RIGHT>\n" ++
  "__device__ inline void tk_concat_cols(DST &dst, const LEFT &left, const RIGHT &right) {\n" ++
  "  static_assert(DST::rows == LEFT::rows, \"concat cols: left rows mismatch\");\n" ++
  "  static_assert(DST::rows == RIGHT::rows, \"concat cols: right rows mismatch\");\n" ++
  "  static_assert(DST::cols == LEFT::cols + RIGHT::cols, \"concat cols: dst cols mismatch\");\n" ++
  "  if constexpr (kittens::ducks::rt::all<DST> && kittens::ducks::rt::all<LEFT> && kittens::ducks::rt::all<RIGHT>) {\n" ++
  "    static_assert(DST::height == LEFT::height, \"concat cols: left height mismatch\");\n" ++
  "    static_assert(DST::height == RIGHT::height, \"concat cols: right height mismatch\");\n" ++
  "    #pragma unroll\n" ++
  "    for (int i = 0; i < DST::height; i++) {\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < LEFT::width; j++) {\n" ++
  "        dst.tiles[i][j] = left.tiles[i][j];\n" ++
  "      }\n" ++
  "      #pragma unroll\n" ++
  "      for (int j = 0; j < RIGHT::width; j++) {\n" ++
  "        dst.tiles[i][LEFT::width + j] = right.tiles[i][j];\n" ++
  "      }\n" ++
  "    }\n" ++
  "  } else if constexpr (kittens::ducks::st::all<DST> && kittens::ducks::st::all<LEFT> && kittens::ducks::st::all<RIGHT>) {\n" ++
  "    auto dst_left = dst.template subtile<DST::rows, LEFT::cols>(int2{0, 0});\n" ++
  "    auto dst_right = dst.template subtile<DST::rows, RIGHT::cols>(int2{0, LEFT::cols});\n" ++
  "    kittens::warp::copy(dst_left, left);\n" ++
  "    kittens::warp::copy(dst_right, right);\n" ++
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
  if usesOuter k then
    helpers := helpers ++ outerHelpers
  return helpers

private def generateHelpersForKernels (kernels : Array Kernel) : String := Id.run do
  let mut needStoreAdd := false
  let mut needLegacyTma := false
  let mut needOuter := false
  for k in kernels do
    if usesStoreAdd k then
      needStoreAdd := true
    if usesLegacyTma k then
      needLegacyTma := true
    if usesOuter k then
      needOuter := true
  let mut helpers := ""
  if needStoreAdd then
    helpers := helpers ++ storeAddHelpers
  if needLegacyTma then
    helpers := helpers ++ legacyTmaHelpers
  if needOuter then
    helpers := helpers ++ outerHelpers
  return helpers

/-- Per-kernel emission metadata that can be persisted safely. -/
structure KernelEmitInfo where
  /-- Kernel definition without helper templates. -/
  definition : String
  /-- Whether this kernel needs `store_add` helper templates. -/
  needsStoreAdd : Bool
  /-- Whether this kernel needs legacy TMA helper templates. -/
  needsLegacyTma : Bool
  /-- Whether this kernel needs slice helper templates. -/
  needsSlice : Bool
  /-- Whether this kernel needs outer-product helper templates. -/
  needsOuter : Bool
  deriving Repr, Inhabited

/-- Generate helper template block from precomputed helper flags. -/
def generateHelpersFromFlags (needStoreAdd needLegacyTma needSlice needOuter : Bool) : String := Id.run do
  let _ := needSlice
  let mut helpers := ""
  if needStoreAdd then
    helpers := helpers ++ storeAddHelpers
  if needLegacyTma then
    helpers := helpers ++ legacyTmaHelpers
  if needOuter then
    helpers := helpers ++ outerHelpers
  return helpers

private def generateKernelDefinition (k : Kernel) (emitSharedDecl : Bool := false) : String :=
  let rvState := inferRvLayouts k
  let rvVars := inferRvDecls k
  let tileInfo := inferTileInfo k
  let archGuard := s!"#if defined({k.arch.toGuard})\n"
  let paramStr := if k.params.isEmpty then "/* TODO: params */" else generateParams k.params
  let signature := s!"__global__ void {k.name}({paramStr}) \{\n"
  let sharedDecl := if emitSharedDecl && k.sharedMemBytes > 0
    then s!"  extern __shared__ char smem[{k.sharedMemBytes}];\n"
    else ""
  let body := k.body.toList.map (generateStmt rvState.layouts rvVars tileInfo "  ") |>.foldl (· ++ ·) ""
  let footer := "}\n#endif\n"
  layoutDiagnostics rvState.conflicts ++ archGuard ++ signature ++ sharedDecl ++ body ++ footer

/-- Generate emission metadata for a single kernel definition. -/
def generateKernelEmitInfo (k : Kernel) : KernelEmitInfo :=
  {
    definition := generateKernelDefinition k false
    needsStoreAdd := usesStoreAdd k
    needsLegacyTma := usesLegacyTma k
    needsSlice := usesSlice k
    needsOuter := usesOuter k
  }

/-- Generate one or more kernel definitions for inclusion in a single `.cu` translation unit.
    Assumes CUDA/ThunderKittens headers are already included by the caller. -/
def generateKernelDefinitions (kernels : Array Kernel) : String :=
  generateHelpersForKernels kernels ++
  (kernels.toList.map (fun k => generateKernelDefinition k false) |> String.intercalate "\n")

/-- Generate full kernel C++ code -/
def generateKernel (k : Kernel) : String :=
  let header := "#include <kittens.cuh>\nusing namespace kittens;\n\n"
  header ++ generateKernelDefinitions #[k]

/-- Generate kernel with extern shared memory declaration -/
def generateKernelWithShared (k : Kernel) : String :=
  let header :=
    "#include <kittens.cuh>\nusing namespace kittens;\n\n" ++
    generateHelpers k
  header ++ generateKernelDefinition k true

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
