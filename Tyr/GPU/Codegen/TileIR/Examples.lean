import Tyr.GPU.Codegen.TileIR.Frontend
import Tyr.GPU.Codegen.TileIR.Attribute
import Tyr.GPU.Codegen.TileIR.Toolchain

/-!
# Tyr.GPU.Codegen.TileIR.Examples

Backend-first examples written with the custom Lean TileIR frontend.

These are intentionally close to the public `cuda_tile.*` syntax so they can be
rendered and compiled through NVIDIA's actual TileIR tools once installed.
-/

namespace Tyr.GPU.Codegen.TileIR

private def f32Tile64x64 : TileType := tileTy .f32 #[64, 64]
private def f16Tile16x16 : TileType := tileTy .f16 #[16, 16]
private def f32Tile16x16 : TileType := tileTy .f32 #[16, 16]
private def f32Tile8 : TileType := tileTy .f32 #[8]
private def f32Scalar : TileType := scalarTileTy .f32
private def i32Scalar : TileType := scalarTileTy .i32
private def i1Scalar : TileType := scalarTileTy .i1
private def f32PtrTile : TileType := ptrTileTy .f32
private def i32PtrTile : TileType := ptrTileTy .i32
private def f32PtrVector8 : TileType := .tile (staticShape #[8]) (.ptr .f32)
private def i32Vector8 : TileType := tileTy .i32 #[8]
private def globalTensor : TileType := tileTy .f32 #[1]

private def tensorView32 : TensorViewType := {
  elem := .f32
  shape := #[.dynamic, .static 32]
  strides := #[.dynamic, .static 1]
}

private def tensorView1D : TensorViewType := {
  elem := .f32
  shape := #[.dynamic]
  strides := #[.static 1]
}

@[tileir_kernel]
def tileAlgebraDemo (lhsPtr : f32PtrTile) (rhsPtr : f32PtrTile) (outPtr : f32PtrTile) :=
  do
    ct.comment "Load two tiles, apply tile algebra, and store the result."
    let lhs ← ct.load lhsPtr : ct.Tile ct.f32 #[64, 64]
    let rhs ← ct.load rhsPtr : ct.Tile ct.f32 #[64, 64]
    let scale ← const_float 1.25 : ct.Scalar ct.f32
    let sum ← lhs + rhs
    let scaleTile ← ct.broadcast scale : ct.Tile ct.f32 #[64, 64]
    let weighted ← sum * scaleTile
    let expTile ← ct.exp2 weighted
    let out ← ct.sqrt expTile
    ct.store outPtr, out

@[tileir_kernel]
def controlFlowDemo (base : f32PtrTile) (pred : i1Scalar) (ptrs : f32PtrVector8) (idxs : i32Vector8) :=
  do
    ct.comment "Construct tensor and partition views, then exercise structured control flow."
    let view ← ct.tensor_view base : tensorView32
    let _partition ← ct.partition_view view : {
      tileShape := #[2, 2]
      tensor := tensorView32
      dimMap := #[0, 1]
    }
    let _shifted ← ct.offset ptrs, idxs : f32PtrVector8
    let zero ← const_int 0 : i32Scalar
    let upper ← const_int 4 : i32Scalar
    let stepVal ← const_int 1 : i32Scalar
    let acc0 ← const_float 0.0 : ct.Scalar ct.f32
    ct.print "partition view prepared"
    let selected ← if pred then do
      pure acc0
    else do
      let one ← const_float 1.0 : ct.Scalar ct.f32
      pure one
    let _acc ← for_tile _iv in (zero to upper, step stepVal) carrying (running := selected) do
      let next := ct.exp2 running
      continue_tile next
    pure ()

@[tileir_kernel]
def mmaLoopDemo (a : f16Tile16x16) (b : f16Tile16x16) (seed : f32Tile16x16) (outPtr : f32PtrTile)
    (zero : i32Scalar) (upper : i32Scalar) (stepVal : i32Scalar) :=
  do
    ct.comment "Iteratively accumulate an MMA tile and clamp it with named tile algebra forms."
    let acc ← for_tile _k in (zero to upper, step stepVal) carrying (running := seed) do
      let product ← ct.mma a, b, running
      let clampedHigh ← ct.max product, running
      let clampedLow ← ct.min clampedHigh, product
      continue_tile clampedLow
    ct.store outPtr, acc

@[tileir_kernel]
def surfaceTileAlgebraDemo
    (lhsPtr : ct.PtrTile ct.f32)
    (rhsPtr : ct.PtrTile ct.f32)
    (scalePtr : ct.PtrTile ct.f32)
    (outPtr : ct.PtrTile ct.f32) := do
  ct.comment "Surface syntax demo with implicit token threading."
  let lhs ← ct.load lhsPtr : ct.Tile ct.f32 #[64, 64]
  let rhs ← ct.load rhsPtr : ct.Tile ct.f32 #[64, 64]
  let scale ← ct.load scalePtr : ct.Scalar ct.f32
  let sum ← lhs + rhs
  let weighted ← sum * scale
  let expTile ← ct.exp2 weighted
  let out ← ct.sqrt expTile
  ct.store outPtr, out

@[tileir_kernel]
def surfaceControlFlowDemo
    (base : ct.PtrTile ct.f32)
    (pred : ct.Scalar ct.i1)
    (ptrs : f32PtrVector8)
    (idxs : i32Vector8) := do
  ct.comment "Surface syntax demo for structural control flow and view ops."
  let view ← ct.tensor_view base : tensorView32
  let _partition ← ct.partition_view view : {
    tileShape := #[2, 2]
    tensor := tensorView32
    dimMap := #[0, 1]
  }
  let _shifted ← ct.offset ptrs, idxs : f32PtrVector8
  let zero := const_int 0 : i32Scalar
  let upper := const_int 4 : i32Scalar
  let stepVal := const_int 1 : i32Scalar
  let acc0 := const_float 0.0 : ct.Scalar ct.f32
  ct.print "surface partition view prepared"
  let selected ← if pred then do
    pure acc0
  else do
    let one := const_float 1.0 : ct.Scalar ct.f32
    pure one
  let _acc ← for_tile _iv in (zero to upper, step stepVal) carrying (running := selected) do
    let next := ct.exp2 running
    continue_tile next
  pure ()

@[tileir_kernel]
def surfaceMmaDemo
    (a : ct.Tile ct.f16 #[16, 16])
    (b : ct.Tile ct.f16 #[16, 16])
    (seed : ct.Tile ct.f32 #[16, 16])
    (outPtr : ct.PtrTile ct.f32) := do
  ct.comment "Surface syntax demo for cutile-style MMA and tile algebra."
  let product ← ct.mma a, b, seed
  let clampedHigh ← ct.max product, seed
  let clampedLow ← ct.min clampedHigh, product
  ct.store outPtr, clampedLow

@[tileir_kernel]
def surfaceBlockIdDemo
    (pred : ct.Scalar ct.i1)
    (bidOutPtr : ct.PtrTile ct.i32)
    (gridOutPtr : ct.PtrTile ct.i32) := do
  ct.comment "Surface syntax demo for TileIR block ids, grid dims, and asserts."
  let bid ← ct.bid 0
  let blocks ← ct.num_blocks 0
  ct.assert pred, "surfaceBlockIdDemo requires pred = true"
  let bidTile ← ct.broadcast bid : ct.Tile ct.i32 #[8]
  let blocksTile ← ct.broadcast blocks : ct.Tile ct.i32 #[8]
  ct.store bidOutPtr, bidTile
  ct.store gridOutPtr, blocksTile

@[tileir_kernel]
def surfaceIotaDemo
    (pred : ct.Scalar ct.i1)
    (outPtr : ct.PtrTile ct.i32) := do
  ct.comment "Surface syntax demo for TileIR iota."
  ct.assert pred, "surfaceIotaDemo requires pred = true"
  let lanes ← ct.iota : ct.Tile ct.i32 #[8]
  ct.store outPtr, lanes

@[tileir_kernel]
def surfaceRangeDemo
    (outPtr : ct.PtrTile ct.f32) := do
  ct.comment "Surface syntax demo for ordinary `for ... in ct.range(...) do` loops."
  let zero := const_int 0 : i32Scalar
  let upper := const_int 4 : i32Scalar
  let stepVal := const_int 1 : i32Scalar
  for _iv in ct.range(zero, upper, stepVal) do
    ct.print "surface range iteration"
  let one := const_float 1.0 : f32Scalar
  let tile ← ct.broadcast one : ct.Tile ct.f32 #[8]
  ct.store outPtr, tile

@[tileir_kernel]
def surfaceStaticRangeDemo
    (outPtr : ct.PtrTile ct.f32)
    (LANES : ct.Const Nat) := do
  ct.comment "Lean-native compile-time unrolling via ct.static_range(ct.cdiv(...))."
  ct.static_assert (LANES == 8), s!"surfaceStaticRangeDemo expects LANES = 8, got {LANES}"
  for _iter in ct.static_range(ct.cdiv(LANES, 2)) do
    ct.print "surface static range iteration"
  let one := const_float 1.0 : ct.Scalar ct.f32
  let tile ← ct.broadcast one : ct.Tile ct.f32 #[LANES]
  ct.store outPtr, tile

@[tileir_kernel]
def surfaceStaticIterListDemo
    (outPtr : ct.PtrTile ct.f32) := do
  ct.comment "ct.static_iter should reuse ordinary Lean ForIn collections inside TileIR entries."
  for _axis in ct.static_iter(([0, 1, 2] : List Nat)) do
    ct.print "surface static list iteration"
  let one := const_float 1.0 : ct.Scalar ct.f32
  let tile ← ct.broadcast one : ct.Tile ct.f32 #[8]
  ct.store outPtr, tile

@[tileir_kernel]
def surfaceViewDemo
    (base : ct.PtrTile ct.f32)
    (outBase : ct.PtrTile ct.f32)
    (pred : ct.Scalar ct.i1) := do
  ct.comment "Surface syntax demo for view-based load/store."
  let bid ← ct.bid 0
  let inView ← ct.partition_view (ct.tensor_view base : tensorView1D) : {
    tileShape := #[8]
    tensor := tensorView1D
    dimMap := #[0]
  }
  let outView ← ct.partition_view (ct.tensor_view outBase : tensorView1D) : {
    tileShape := #[8]
    tensor := tensorView1D
    dimMap := #[0]
  }
  let tile ← ct.load_view inView, [bid] : ct.Tile ct.f32 #[8]
  ct.assert pred, "surfaceViewDemo requires pred = true"
  let one := const_float 1.0 : ct.Scalar ct.f32
  let bumped ← tile + one
  ct.store_view outView, [bid], bumped

@[tileir_kernel]
def surfaceConstShapeDemo
    (inPtr : ct.PtrTile ct.f32)
    (LANES : ct.Const Nat)
    (outPtr : ct.PtrTile ct.f32) := do
  ct.comment "Surface syntax demo for ct.Const-dependent tile shapes."
  ct.static_assert (LANES == 8), s!"surfaceConstShapeDemo expects LANES = 8, got {LANES}"
  let tile ← ct.load inPtr : ct.Tile ct.f32 #[LANES]
  ct.store outPtr, tile

@[tileir_kernel]
def surfaceStaticEvalDemo
    (lhsPtr : ct.PtrTile ct.f32)
    (useSqrt : ct.Const Bool)
    (rhsPtr : ct.PtrTile ct.f32)
    (outPtr : ct.PtrTile ct.f32) := do
  ct.comment "Surface syntax demo for ct.static_eval-driven specialization."
  let lhs ← ct.load lhsPtr : ct.Tile ct.f32 #[8]
  let rhs ← ct.load rhsPtr : ct.Tile ct.f32 #[8]
  let sum ← lhs + rhs
  let out ← ct.static_eval (if useSqrt then ct.sqrt sum else ct.exp2 sum)
  ct.store outPtr, out

@[tileir_kernel]
def surfaceIndexedVecAdd1D
    (a : ct.Array ct.f32)
    (b : ct.Array ct.f32)
    (c : ct.Array ct.f32)
    (TILE : ct.Const Nat) := do
  ct.comment "cutile-style direct tiled load/store for 1D vector addition."
  let bid := ct.bid 0
  let aTile ← ct.load a, index := (bid,), shape := (TILE,)
  let bTile ← ct.load b, index := (bid,), shape := (TILE,)
  ct.store c, index := (bid,), tile := aTile + bTile

@[tileir_kernel]
def surfaceIndexedVecAdd2D
    (a : ct.Array ct.f32)
    (b : ct.Array ct.f32)
    (c : ct.Array ct.f32)
    (TILE_X : ct.Const Nat)
    (TILE_Y : ct.Const Nat) := do
  ct.comment "cutile-style direct tiled load/store for 2D matrix addition."
  let bidX := ct.bid 0
  let bidY := ct.bid 1
  let aTile ← ct.load a, index := (bidX, bidY), shape := (TILE_X, TILE_Y)
  let bTile ← ct.load b, index := (bidX, bidY), shape := (TILE_X, TILE_Y)
  ct.store c, index := (bidX, bidY), tile := aTile + bTile

@[tileir_kernel]
def surfaceArangeDemo
    (outPtr : ct.PtrTile ct.i32)
    (LANES : ct.Const Nat) := do
  ct.comment "cutile-style ct.arange lowering through TileIR iota with an explicit dtype."
  let lanes ← ct.arange(LANES, dtype := ct.int32)
  ct.store outPtr, lanes

@[tileir_kernel]
def surfaceGatherScatterDemo
    (input : ct.Array ct.f32)
    (output : ct.Array ct.f32)
    (LANES : ct.Const Nat) := do
  ct.comment "cutile-style ct.gather/ct.scatter lowering through pointer offsets."
  let lanes := ct.arange(LANES, dtype := ct.int32)
  let tile ← ct.gather(input, lanes)
  let one := const_float 1.0 : ct.Scalar ct.float32
  let bumped := tile + one
  ct.scatter(output, lanes, bumped)

@[tileir_kernel]
def surfaceFillCastDemo
    (input : ct.Array ct.i32)
    (output : ct.Array ct.f32)
    (LANES : ct.Const Nat) := do
  ct.comment "cutile-style ct.full/ct.zeros lowering with scalar indexed loads and dtype metadata."
  let bid ← ct.bid 0
  let scalar ← ct.load input, index := bid, shape := ()
  let seed ← ct.full((LANES,), scalar, dtype := ct.f32)
  let zeros ← ct.zeros((LANES,), ct.f32)
  let ones ← ct.ones((LANES,), ct.f32)
  let withZeros ← seed + zeros
  let total ← withZeros + ones
  ct.store output, index := (bid,), tile := total

@[tileir_kernel]
def surfaceReshapeDemo
    (output : ct.Array ct.f32)
    (ROWS : ct.Const Nat)
    (COLS : ct.Const Nat) := do
  ct.comment "cutile-style ct.reshape lowering derived from the source tile type."
  let bid ← ct.bid 0
  let tile ← ct.ones((ROWS, COLS), output.dtype)
  let flat ← tile.reshape #[ROWS * COLS]
  ct.store output, index := (bid,), tile := flat

@[tileir_kernel]
def surfaceShapeMetadataDemo
    (input : ct.Array ct.i32)
    (output : ct.Array ct.f32)
    (LANES : ct.Const Nat) := do
  ct.comment "Lean-native metadata and positional surface demo with static shape checks."
  let bid ← ct.bid 0
  let ints ← ct.load(input, (bid,), (LANES,))
  ct.static_assert (ints.shape == #[LANES]), "loaded tile shape should track ct.Const parameters"
  ct.static_assert (ints.rank == 1), "loaded tile rank should be available as Lean metadata"
  ct.static_assert (ints.dtype == ct.i32), "loaded tile dtype should be available as Lean metadata"
  let floats ← ints.astype output.dtype
  let matrix ← ct.reshape(floats, (1, LANES))
  ct.static_assert (matrix.shape == #[1, LANES]), "reshaped tile should expose its new static shape"
  let flat ← matrix.reshape ints.shape
  ct.store(output, (bid,), flat)

@[tileir_kernel]
def surfaceMethodSyntaxDemo
    (input : ct.Array ct.i32)
    (output : ct.Array ct.f32)
    (LANES : ct.Const Nat) := do
  ct.comment "cutile-style method syntax demo for shape metadata, astype, and reshape."
  let bid ← ct.bid 0
  let ints ← ct.load(input, (bid,), (LANES,))
  let dims := ct.static_eval ints.shape
  ct.static_assert (dims == #[LANES]), "surfaceMethodSyntaxDemo expects a 1D tile"
  ct.static_assert (ints.ndim == 1), "surfaceMethodSyntaxDemo expects `ndim` metadata to match rank"
  let floats ← ints.astype output.dtype
  let matrix ← floats.reshape #[1, LANES]
  ct.static_assert (matrix.shape == #[1, LANES]), "reshaped tiles should expose their new metadata"
  let flat ← matrix.reshape #[LANES]
  ct.store(output, (bid,), flat)

@[tileir_kernel]
def surfaceTransformDemo
    (input : ct.Array ct.f32)
    (output : ct.Array ct.f32)
    (ROWS : ct.Const Nat)
    (COLS : ct.Const Nat) := do
  ct.comment "cutile-style ct.extract/ct.cat/ct.permute/ct.where lowering with Lean-native metadata."
  ct.static_assert (ROWS == 4), s!"surfaceTransformDemo expects ROWS = 4, got {ROWS}"
  ct.static_assert (COLS == 4), s!"surfaceTransformDemo expects COLS = 4, got {COLS}"
  let bid ← ct.bid 0
  let zero := const_int 0 : ct.Scalar ct.i32
  let one := const_int 1 : ct.Scalar ct.i32
  let tile ← ct.load input, index := (bid, zero), shape := (ROWS, COLS)
  ct.static_assert (tile.shape == #[ROWS, COLS]), "loaded transform tile should expose its static shape"
  let top ← ct.extract tile, index := (zero, zero), shape := (ROWS / 2, COLS)
  let bottom ← ct.extract tile, index := (one, zero), shape := (ROWS / 2, COLS)
  let swapped ← ct.cat((bottom, top), 0)
  let transposed ← swapped.permute #[1, 0]
  ct.static_assert (transposed.shape == #[COLS, ROWS]), "permuted tile should expose its transformed shape"
  let mask ← ct.full_like(transposed, true, dtype := ct.bool_)
  let zeros ← ct.zeros_like(transposed)
  let selected ← ct.where(mask, transposed, zeros)
  ct.store output, index := (bid, zero), tile := selected

@[tileir_kernel]
def surfaceTransposeBroadcastDemo
    (input : ct.Array ct.f32)
    (output : ct.Array ct.f32)
    (ROWS : ct.Const Nat)
    (COLS : ct.Const Nat) := do
  ct.comment "Lean-typed transpose and broadcast_to demo over the cutile-style surface."
  ct.static_assert (ROWS == 2), s!"surfaceTransposeBroadcastDemo expects ROWS = 2, got {ROWS}"
  ct.static_assert (COLS == 4), s!"surfaceTransposeBroadcastDemo expects COLS = 4, got {COLS}"
  let bid ← ct.bid 0
  let zero := const_int 0 : ct.Scalar ct.i32
  let tile ← ct.load input, index := (bid, zero), shape := (ROWS, COLS)
  let transposed ← tile.transpose
  ct.static_assert (transposed.shape == #[COLS, ROWS]), "transpose should swap the static shape metadata"
  let one := const_float 1.0 : ct.Scalar ct.f32
  let bias : ct.Val (ct.Tile ct.f32 #[COLS, ROWS]) ← ct.broadcast_to(one, (COLS, ROWS))
  let shifted ← transposed + bias
  ct.store output, index := (bid, zero), tile := shifted

@[tileir_kernel]
def surfaceCompareSelectDemo
    (input : ct.Array ct.f32)
    (output : ct.Array ct.f32)
    (ROWS : ct.Const Nat)
    (COLS : ct.Const Nat) := do
  ct.comment "Lean-typed compare/select demo with scalar broadcasting on both compare and where."
  ct.static_assert (ROWS == 2), s!"surfaceCompareSelectDemo expects ROWS = 2, got {ROWS}"
  ct.static_assert (COLS == 4), s!"surfaceCompareSelectDemo expects COLS = 4, got {COLS}"
  let bid ← ct.bid 0
  let zeroIdx := const_int 0 : ct.Scalar ct.i32
  let tile ← ct.load input, index := (bid, zeroIdx), shape := (ROWS, COLS)
  let zero := const_float 0.0 : ct.Scalar ct.f32
  let mask : ct.Val (ct.Tile ct.i1 #[ROWS, COLS]) ← ct.greater tile, zero
  let clamped : ct.Val (ct.Tile ct.f32 #[ROWS, COLS]) ← ct.where(mask, tile, zero)
  ct.store output, index := (bid, zeroIdx), tile := clamped

@[tileir_kernel]
def surfaceMaximumMinimumDemo
    (input : ct.Array ct.f32)
    (output : ct.Array ct.f32)
    (ROWS : ct.Const Nat)
    (COLS : ct.Const Nat) := do
  ct.comment "cutile-style ct.maximum/ct.minimum surface aliases over typed TileIR values."
  ct.static_assert (ROWS == 2), s!"surfaceMaximumMinimumDemo expects ROWS = 2, got {ROWS}"
  ct.static_assert (COLS == 4), s!"surfaceMaximumMinimumDemo expects COLS = 4, got {COLS}"
  let bid ← ct.bid 0
  let zeroIdx := const_int 0 : ct.Scalar ct.i32
  let tile ← ct.load input, index := (bid, zeroIdx), shape := (ROWS, COLS)
  let ones ← ct.ones_like(tile)
  let hi ← ct.maximum tile, ones
  let lo ← ct.minimum hi, tile
  ct.store output, index := (bid, zeroIdx), tile := lo

@[tileir_kernel]
def surfaceMatmulDemo
    (a : ct.Tile ct.f16 #[16, 16])
    (b : ct.Tile ct.f16 #[16, 16])
    (acc : ct.Tile ct.f32 #[16, 16])
    (out : ct.PtrTile ct.f32) := do
  ct.comment "Lean-typed ct.matmul surface alias with static shape indexing over TileIR tiles."
  ct.static_assert (a.shape == #[16, 16]), "surfaceMatmulDemo expects 16x16 lhs tiles"
  ct.static_assert (a.shape[0]! == 16), "surfaceMatmulDemo expects lhs row metadata to be indexable"
  ct.static_assert (b.shape[1]! == 16), "surfaceMatmulDemo expects rhs column metadata to be indexable"
  let prod ← a.matmul b acc
  ct.store out, prod

@[tileir_kernel]
def surfaceConstShapeDemo8 : Module :=
  surfaceConstShapeDemo 8

@[tileir_kernel]
def surfaceStaticEvalSqrtDemo : Module :=
  surfaceStaticEvalDemo true

@[tileir_kernel]
def surfaceStaticEvalExp2Demo : Module :=
  surfaceStaticEvalDemo false

@[tileir_kernel]
def surfaceIndexedVecAdd1D8 : Module :=
  surfaceIndexedVecAdd1D 8

@[tileir_kernel]
def surfaceIndexedVecAdd2D4x8 : Module :=
  surfaceIndexedVecAdd2D 4 8

@[tileir_kernel]
def surfaceArangeDemo8 : Module :=
  surfaceArangeDemo 8

@[tileir_kernel]
def surfaceStaticRangeDemo8 : Module :=
  surfaceStaticRangeDemo 8

@[tileir_kernel]
def surfaceGatherScatterDemo8 : Module :=
  surfaceGatherScatterDemo 8

@[tileir_kernel]
def surfaceFillCastDemo8 : Module :=
  surfaceFillCastDemo 8

@[tileir_kernel]
def surfaceReshapeDemo2x4 : Module :=
  surfaceReshapeDemo 2 4

@[tileir_kernel]
def surfaceShapeMetadataDemo8 : Module :=
  surfaceShapeMetadataDemo 8

@[tileir_kernel]
def surfaceMethodSyntaxDemo8 : Module :=
  surfaceMethodSyntaxDemo 8

@[tileir_kernel]
def surfaceTransformDemo4x4 : Module :=
  surfaceTransformDemo 4 4

@[tileir_kernel]
def surfaceTransposeBroadcastDemo2x4 : Module :=
  surfaceTransposeBroadcastDemo 2 4

@[tileir_kernel]
def surfaceCompareSelectDemo2x4 : Module :=
  surfaceCompareSelectDemo 2 4

@[tileir_kernel]
def surfaceMaximumMinimumDemo2x4 : Module :=
  surfaceMaximumMinimumDemo 2 4

@[tileir_kernel]
def surfaceMatmulDemo16 : Module :=
  surfaceMatmulDemo

def tileAlgebraArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt tileAlgebraDemo outDir opts

def controlFlowArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt controlFlowDemo outDir opts

def mmaLoopArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt mmaLoopDemo outDir opts

def surfaceTileAlgebraArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceTileAlgebraDemo outDir opts

def surfaceControlFlowArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceControlFlowDemo outDir opts

def surfaceMmaArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceMmaDemo outDir opts

def surfaceBlockIdArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceBlockIdDemo outDir opts

def surfaceIotaArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceIotaDemo outDir opts

def surfaceConstShapeArtifacts (lanes : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceConstShapeDemo lanes) outDir opts

def surfaceStaticEvalArtifacts (useSqrt : Bool) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceStaticEvalDemo useSqrt) outDir opts

def surfaceIndexedVecAdd1DArtifacts (tile : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceIndexedVecAdd1D tile) outDir opts

def surfaceIndexedVecAdd2DArtifacts
    (tileX tileY : Nat)
    (outDir : System.FilePath)
    (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceIndexedVecAdd2D tileX tileY) outDir opts

def surfaceArangeArtifacts (lanes : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceArangeDemo lanes) outDir opts

def surfaceGatherScatterArtifacts (lanes : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceGatherScatterDemo lanes) outDir opts

def surfaceFillCastArtifacts (lanes : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceFillCastDemo lanes) outDir opts

def surfaceReshapeArtifacts
    (rows cols : Nat)
    (outDir : System.FilePath)
    (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceReshapeDemo rows cols) outDir opts

def surfaceShapeMetadataArtifacts (lanes : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceShapeMetadataDemo lanes) outDir opts

def surfaceMethodSyntaxArtifacts (lanes : Nat) (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceMethodSyntaxDemo lanes) outDir opts

def surfaceTransformArtifacts
    (rows cols : Nat)
    (outDir : System.FilePath)
    (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceTransformDemo rows cols) outDir opts

def surfaceCompareSelectArtifacts
    (rows cols : Nat)
    (outDir : System.FilePath)
    (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt (surfaceCompareSelectDemo rows cols) outDir opts

def surfaceConstShapeDemo8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceConstShapeDemo8 outDir opts

def surfaceStaticEvalSqrtArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceStaticEvalSqrtDemo outDir opts

def surfaceStaticEvalExp2Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceStaticEvalExp2Demo outDir opts

def surfaceIndexedVecAdd1D8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceIndexedVecAdd1D8 outDir opts

def surfaceIndexedVecAdd2D4x8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceIndexedVecAdd2D4x8 outDir opts

def surfaceArangeDemo8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceArangeDemo8 outDir opts

def surfaceGatherScatterDemo8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceGatherScatterDemo8 outDir opts

def surfaceFillCastDemo8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceFillCastDemo8 outDir opts

def surfaceReshapeDemo2x4Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceReshapeDemo2x4 outDir opts

def surfaceShapeMetadataDemo8Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceShapeMetadataDemo8 outDir opts

def surfaceTransformDemo4x4Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceTransformDemo4x4 outDir opts

def surfaceCompareSelectDemo2x4Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceCompareSelectDemo2x4 outDir opts

def surfaceMatmulDemo16Artifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceMatmulDemo16 outDir opts

def surfaceRangeArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceRangeDemo outDir opts

def surfaceViewArtifacts (outDir : System.FilePath) (opts : CompileOptions := {})
    : IO (Except ToolError ArtifactPaths) :=
  compileModuleAt surfaceViewDemo outDir opts

end Tyr.GPU.Codegen.TileIR
