import Tyr.GPU.Codegen.TileIR.Expr

/-!
# Tyr.GPU.Codegen.TileIR.Builder

Backend-first builder DSL for constructing TileIR modules in Lean.

This is intentionally a lightweight replacement for the Python-side kernel DSL:

- `@[tileir_kernel]` still marks the entrypoint declaration,
- authors build TileIR with Lean `do` notation instead of raw AST arrays,
- and the result stays a plain `Tyr.GPU.Codegen.TileIR.Module`.
-/

namespace Tyr.GPU.Codegen.TileIR

structure Value where
  name : String
  ty : TileType
  deriving Repr, Inhabited, BEq, DecidableEq

structure ValueAndToken where
  value : Value
  token : Value
  deriving Repr, Inhabited, BEq, DecidableEq

inductive LoopResult where
  | continue_ (values : Array Value)
  | break_ (values : Array Value)
  deriving Repr, Inhabited, BEq, DecidableEq

namespace Value

def binding (value : Value) : Binding :=
  { name := value.name, ty := value.ty }

def param (value : Value) : Param :=
  { name := value.name, ty := value.ty }

def dtype (value : Value) : ScalarType :=
  match value.ty with
  | .ptr elem => elem
  | .tile _ (.scalar elem) => elem
  | .tile _ (.ptr elem) => elem
  | .tensorView desc => desc.elem
  | .partitionView desc => desc.tensor.elem
  | .token => panic! "TileIR tokens do not have a dtype"

def shape? (value : Value) : Option (Array Nat) :=
  value.ty.staticShape?

def shape (value : Value) : Array Nat :=
  match value.shape? with
  | some shape => shape
  | none => panic! s!"TileIR value {value.name} does not have a statically known shape"

def rank (value : Value) : Nat :=
  value.shape.size

end Value

instance : Coe Value Binding where
  coe := Value.binding

instance : Coe Value Param where
  coe := Value.param

instance : Coe Value String where
  coe := Value.name

structure EntryState where
  stmts : Array Stmt := #[]
  nextId : Nat := 0
  currentToken? : Option Value := none
  tileBlockIds? : Option (Value × Value × Value) := none
  numTileBlocks? : Option (Value × Value × Value) := none
  deriving Inhabited

abbrev EntryM := ExceptT String (StateM EntryState)

structure ModuleState where
  globals : Array Global := #[]
  entries : Array Entry := #[]
  deriving Inhabited

abbrev ModuleM := ExceptT String (StateM ModuleState)

def arg (name : String) (ty : TileType) : Value :=
  { name, ty }

class ToTileLiteral (α : Type) where
  toLiteral : α → Literal

class FillValue (α : Type) where
  build : String → Array Nat → α → ScalarType → EntryM Value

instance : ToTileLiteral Int where
  toLiteral := .int

instance : ToTileLiteral Nat where
  toLiteral := fun value => .int (Int.ofNat value)

instance : ToTileLiteral Float where
  toLiteral := .float

instance : ToTileLiteral Bool where
  toLiteral := .bool

private def sanitizeHint (hint : String) : String :=
  let filtered := hint.toList.filter fun c => c.isAlphanum || c == '_'
  let base :=
    match filtered with
    | [] => "v"
    | cs => String.ofList cs
  match base.toList with
  | [] => "v"
  | c :: _ =>
      if c.isAlpha || c == '_' then
        base
      else
        "v" ++ base

private def emit (stmt : Stmt) : EntryM Unit :=
  modify fun st => { st with stmts := st.stmts.push stmt }

private def freshName (hint : String) : EntryM String := do
  let st ← get
  let base := sanitizeHint hint
  let name := s!"{base}{st.nextId}"
  set { st with nextId := st.nextId + 1 }
  pure name

private def freshValue (hint : String) (ty : TileType) : EntryM Value := do
  pure { name := ← freshName hint, ty }

private def captureBlockWithToken?
    (inputToken? : Option Value)
    (body : EntryM α)
    : EntryM (α × Array Stmt × Option Value) := do
  let outer ← get
  let (result, inner) := body.run {
    nextId := outer.nextId
    currentToken? := inputToken?
    tileBlockIds? := outer.tileBlockIds?
    numTileBlocks? := outer.numTileBlocks?
  }
  set { outer with nextId := inner.nextId }
  match result with
  | .ok value => pure (value, inner.stmts, inner.currentToken?)
  | .error err => throw err

private def captureBlock (body : EntryM α) : EntryM (α × Array Stmt × Option Value) := do
  captureBlockWithToken? (← get).currentToken? body

private def currentToken : EntryM Value := do
  let some token := (← get).currentToken?
    | throw "TileIR surface effect requires an implicit token, but none is available in this entry"
  pure token

private def setCurrentToken (token : Value) : EntryM Unit :=
  modify fun st => { st with currentToken? := some token }

private def initialToken? (params : Array Value) : Option Value :=
  params.foldl (fun acc param => if param.ty == .token then some param else acc) none

private def axisIndex (axis : Nat) : EntryM Nat := do
  if axis < 3 then
    pure axis
  else
    throw s!"TileIR axis must be one of 0, 1, or 2, but got {axis}"

private def ensureTileBlockIds : EntryM (Value × Value × Value) := do
  if let some cached := (← get).tileBlockIds? then
    pure cached
  else
    let x ← freshValue "bid_x" (scalarTileTy .i32)
    let y ← freshValue "bid_y" (scalarTileTy .i32)
    let z ← freshValue "bid_z" (scalarTileTy .i32)
    emit (.getTileBlockId x y z)
    let cached := (x, y, z)
    modify fun st => { st with tileBlockIds? := some cached }
    pure cached

private def ensureNumTileBlocks : EntryM (Value × Value × Value) := do
  if let some cached := (← get).numTileBlocks? then
    pure cached
  else
    let x ← freshValue "grid_x" (scalarTileTy .i32)
    let y ← freshValue "grid_y" (scalarTileTy .i32)
    let z ← freshValue "grid_z" (scalarTileTy .i32)
    emit (.getNumTileBlocks x y z)
    let cached := (x, y, z)
    modify fun st => { st with numTileBlocks? := some cached }
    pure cached

private def valueTileScalar? (value : Value) : Option ScalarType :=
  value.ty.literalScalar?

private def valueElemTypeShape? (value : Value) : Option (ElemType × Array Nat) :=
  match value.ty with
  | .tile shape elem =>
      let dims := shape.map fun
        | .static n => n
        | .dynamic => 0
      if shape.all (· != .dynamic) then
        some (elem, dims)
      else
        none
  | _ =>
      none

private def tileElemTypeShape (context : String) (value : Value) : EntryM (ElemType × Array Nat) := do
  match value.ty with
  | .tile shape elem =>
      let mut dims : Array Nat := #[]
      for dim in shape do
        match dim with
        | .static n =>
            dims := dims.push n
        | .dynamic =>
            throw s!"{context} requires statically known tile shapes"
      pure (elem, dims)
  | _ =>
      throw s!"{context} expected a tile value, but got {value.ty.render}"

private def ptrElementScalar? (value : Value) : Option ScalarType :=
  match value.ty with
  | .ptr elem => some elem
  | .tile _ (.ptr elem) => some elem
  | _ => none

private def tileTypeFromShape (elem : ElemType) (shape : Array Nat) : TileType :=
  .tile (staticShape shape) elem

private def scalarLikeTileTy (dtype : ScalarType) (shape : Array Nat) : TileType :=
  tileTypeFromShape (.scalar dtype) shape

private def defaultZeroLiteral (dtype : ScalarType) : Literal :=
  if dtype.isFloat then
    .float 0.0
  else if dtype == .i1 then
    .bool false
  else
    .int 0

private def defaultOneLiteral (dtype : ScalarType) : Literal :=
  if dtype.isFloat then
    .float 1.0
  else if dtype == .i1 then
    .bool true
  else
    .int 1

private def scalarSignedness? : ScalarType → Option Signedness
  | .i1 | .i8 | .i16 | .i32 | .i64 => some .signed
  | .u8 | .u16 | .u32 | .u64 | .index => some .unsigned
  | _ => none

private def castOpFor (src dst : ScalarType) : EntryM CastOp := do
  if src == dst then
    pure .bitcast
  else if src.isFloat && dst.isFloat then
    pure <| .ftof .nearestEven
  else if src.isFloat then
    let some signedness := scalarSignedness? dst
      | throw s!"TileIR surface `ct.astype` cannot cast floating-point tiles to {dst.render}"
    pure <| .ftoi signedness .nearestIntToZero
  else if dst.isFloat then
    let some signedness := scalarSignedness? src
      | throw s!"TileIR surface `ct.astype` cannot cast {src.render} tiles to floating-point"
    pure <| .itof signedness .nearestEven
  else
    let some srcWidth := src.bitWidth?
      | throw s!"TileIR surface `ct.astype` does not support source element type {src.render}"
    let some dstWidth := dst.bitWidth?
      | throw s!"TileIR surface `ct.astype` does not support destination element type {dst.render}"
    if srcWidth < dstWidth then
      let some signedness := scalarSignedness? src
        | throw s!"TileIR surface `ct.astype` cannot extend integers of type {src.render}"
      pure <| .exti signedness
    else if srcWidth > dstWidth then
      pure .trunci
    else
      throw s!"TileIR surface `ct.astype` does not yet support same-width integer casts from {src.render} to {dst.render}"

private def isIndexScalar : ScalarType → Bool
  | .i8 | .i16 | .i32 | .i64
  | .u8 | .u16 | .u32 | .u64
  | .index => true
  | _ => false

private def staticNatShape (context : String) (shape : Array ShapeDim) : EntryM (Array Nat) := do
  let mut dims : Array Nat := #[]
  for dim in shape do
    match dim with
    | .static n =>
        dims := dims.push n
    | .dynamic =>
        throw s!"{context} requires statically known tile shapes"
  pure dims

private def tileScalarShape (context : String) (value : Value) : EntryM (ScalarType × Array Nat) := do
  match value.ty with
  | .tile shape (.scalar elem) =>
      let dims ← staticNatShape context shape
      pure (elem, dims)
  | _ =>
      throw s!"{context} expected a scalar tile value, but got {value.ty.render}"

private def ensureNumericScalarTile (context : String) (value : Value) : EntryM (ScalarType × Array Nat) := do
  let (elem, shape) ← tileScalarShape context value
  unless elem.isIntegral || elem.isFloat do
    throw s!"{context} expected a numeric scalar tile, but got {value.ty.render}"
  pure (elem, shape)

private def ensureFloatScalarTile (context : String) (value : Value) : EntryM (ScalarType × Array Nat) := do
  let (elem, shape) ← tileScalarShape context value
  unless elem.isFloat do
    throw s!"{context} expected a floating-point scalar tile, but got {value.ty.render}"
  pure (elem, shape)

private def comparisonResultTy
    (context : String)
    (shape : Array Nat)
    (dstTy? : Option TileType)
    : EntryM TileType := do
  let inferredTy := scalarLikeTileTy .i1 shape
  match dstTy? with
  | none =>
      pure inferredTy
  | some dstTy =>
      unless dstTy == inferredTy do
        throw s!"{context} expected result type {inferredTy.render}, but got {dstTy.render}"
      pure dstTy

private def ensureBoolScalarTile (context : String) (value : Value) : EntryM Unit := do
  let (elem, shape) ← tileScalarShape context value
  unless elem == .i1 && shape.isEmpty do
    throw s!"{context} expected an i1 scalar tile, but got {value.ty.render}"

private def ensureIndexScalarTile (context : String) (value : Value) : EntryM ScalarType := do
  let (elem, shape) ← tileScalarShape context value
  unless shape.isEmpty && isIndexScalar elem do
    throw s!"{context} expected an integral scalar tile, but got {value.ty.render}"
  pure elem

private def promoteFloatDtype (context : String) (lhs rhs : ScalarType) : EntryM ScalarType := do
  unless lhs.isFloat && rhs.isFloat do
    throw s!"{context} expects floating-point tile operands, but got {lhs.render} and {rhs.render}"
  pure <|
    if lhs == rhs then
      lhs
    else if lhs == .f64 || rhs == .f64 then
      .f64
    else
      .f32

private def ensureStaticElemCountEq
    (context : String)
    (lhs rhs : Array Nat) : EntryM Unit := do
  let lhsCount := lhs.foldl (init := 1) fun acc dim => acc * dim
  let rhsCount := rhs.foldl (init := 1) fun acc dim => acc * dim
  unless lhsCount == rhsCount do
    throw s!"{context} requires the same element count, but got {lhsCount} and {rhsCount}"

private def renderNatShape (shape : Array Nat) : String :=
  s!"[{String.intercalate ", " (shape.toList.map toString)}]"

private partial def isPowerOfTwo : Nat → Bool
  | 0 => false
  | 1 => true
  | n =>
      if n % 2 == 0 then
        isPowerOfTwo (n / 2)
      else
        false

private def ensurePowerOfTwoShape
    (context : String)
    (shape : Array Nat) : EntryM Unit := do
  for dim in shape do
    unless isPowerOfTwo dim do
      throw s!"{context} requires power-of-two tile dimensions, but got {renderNatShape shape}"

private def ensurePermutation
    (context : String)
    (perm : Array Nat)
    (rank : Nat) : EntryM Unit := do
  unless perm.size == rank do
    throw s!"{context} expected a permutation of rank {rank}, but got {renderNatShape perm}"
  let mut seen := Array.replicate rank false
  for axis in perm do
    unless axis < rank do
      throw s!"{context} expected axes in [0, {rank}), but got {renderNatShape perm}"
    if seen[axis]! then
      throw s!"{context} expected a permutation without duplicates, but got {renderNatShape perm}"
    seen := seen.set! axis true

private def permuteShape (shape perm : Array Nat) : Array Nat :=
  perm.map fun axis => shape[axis]!

private def offsetResultPtrTy (context : String) (base idx : Value) : EntryM TileType := do
  let some elem := ptrElementScalar? base
    | throw s!"{context} expected a pointer-typed source, but got {base.ty.render}"
  let (_, idxShape) ← tileScalarShape context idx
  pure <| .tile (staticShape idxShape) (.ptr elem)

private def gatherValueTy (context : String) (base idx : Value) : EntryM TileType := do
  let some elem := ptrElementScalar? base
    | throw s!"{context} expected a pointer-typed source, but got {base.ty.render}"
  let (idxElem, idxShape) ← tileScalarShape context idx
  unless isIndexScalar idxElem do
    throw s!"{context} expected integral gather/scatter indices, but got {idx.ty.render}"
  pure <| .tile (staticShape idxShape) (.scalar elem)

private def rowMajorViewStrides (rank : Nat) : Array ShapeDim :=
  Array.ofFn (n := rank) fun i : Fin rank =>
    if i.1 + 1 == rank then
      .static 1
    else
      .dynamic

private def directTensorViewType (base : Value) (rank : Nat) : EntryM TensorViewType := do
  let some elem := ptrElementScalar? base
    | throw s!"TileIR surface `ct.load`/`ct.store` with `index := ...` expected a pointer-typed source, but got {base.ty.render}"
  if rank == 0 then
    throw "TileIR surface indexed `ct.load`/`ct.store` requires at least one tile index"
  pure {
    elem := elem
    shape := Array.replicate rank .dynamic
    strides := rowMajorViewStrides rank
  }

private def ensureMatchingTypes
    (context : String)
    (expected actual : Array Value)
    : EntryM (Array TileType) := do
  if expected.size != actual.size then
    throw s!"{context} expected {expected.size} result(s), but got {actual.size}"
  let mut tys : Array TileType := #[]
  for i in [0:expected.size] do
    let lhs := expected[i]!
    let rhs := actual[i]!
    if lhs.ty != rhs.ty then
      throw s!"{context} result {i} has type mismatch: expected {lhs.ty.render}, got {rhs.ty.render}"
    tys := tys.push lhs.ty
  pure tys

private def freshResults (hint : String) (tys : Array TileType) : EntryM (Array Value) := do
  let mut results : Array Value := #[]
  for i in [0:tys.size] do
    let suffix := if tys.size == 1 then hint else s!"{hint}{i}"
    results := results.push (← freshValue suffix tys[i]!)
  pure results

def comment (text : String) : EntryM Unit :=
  emit (.comment text)

def const (hint : String) (ty : TileType) (value : Literal) : EntryM Value := do
  let dst ← freshValue hint ty
  emit (.const dst value)
  pure dst

def tileBlockId (axis : Nat) : EntryM Value := do
  let axis ← axisIndex axis
  let (x, y, z) ← ensureTileBlockIds
  pure <| match axis with
    | 0 => x
    | 1 => y
    | _ => z

def numTileBlocks (axis : Nat) : EntryM Value := do
  let axis ← axisIndex axis
  let (x, y, z) ← ensureNumTileBlocks
  pure <| match axis with
    | 0 => x
    | 1 => y
    | _ => z

def constInt (hint : String) (value : Int) (ty : TileType := scalarTileTy .i32) : EntryM Value :=
  const hint ty (.int value)

def constFloat (hint : String) (value : Float) (ty : TileType := scalarTileTy .f32) : EntryM Value :=
  const hint ty (.float value)

def constBool (hint : String) (value : Bool) (ty : TileType := scalarTileTy .i1) : EntryM Value :=
  const hint ty (.bool value)

def constScalarOf [ToTileLiteral α] (hint : String) (dtype : ScalarType) (value : α) : EntryM Value :=
  const hint (scalarTileTy dtype) (ToTileLiteral.toLiteral value)

def unary (hint : String) (op : UnaryOp) (src : Value) (dstTy? : Option TileType := none)
    : EntryM Value := do
  match op with
  | .copy =>
      discard <| tileScalarShape "TileIR unary `copy`" src
  | .exp | .exp2 | .log | .sqrt | .rsqrt =>
      discard <| ensureFloatScalarTile s!"TileIR unary `{op.render}`" src
  | .abs | .neg =>
      discard <| ensureNumericScalarTile s!"TileIR unary `{op.render}`" src
  let dst ← freshValue hint (dstTy?.getD src.ty)
  emit (.unary dst op src.name)
  pure dst

def iota (hint : String) (ty : TileType) : EntryM Value := do
  let dst ← freshValue hint ty
  emit (.iota dst)
  pure dst

def binary (hint : String) (op : BinaryOp) (lhs rhs : Value) (dstTy? : Option TileType := none)
    : EntryM Value := do
  let (lhsElem, lhsShape) ← tileScalarShape s!"TileIR binary `{op.render}`" lhs
  let (rhsElem, rhsShape) ← tileScalarShape s!"TileIR binary `{op.render}`" rhs
  unless lhsElem == rhsElem && lhsShape == rhsShape do
    throw s!"TileIR binary `{op.render}` expects matching operand types, but got {lhs.ty.render} and {rhs.ty.render}"
  unless lhsElem.isFloat do
    throw s!"TileIR binary `{op.render}` expects floating-point scalar tiles, but got {lhs.ty.render}"
  let dst ← freshValue hint (dstTy?.getD lhs.ty)
  emit (.binary dst op lhs.name rhs.name)
  pure dst

def cmpf
    (hint : String)
    (pred : ComparisonPredicate)
    (mode : FloatCompareMode)
    (lhs rhs : Value)
    (dstTy? : Option TileType := none)
    : EntryM Value := do
  let (lhsElem, lhsShape) ← ensureFloatScalarTile "TileIR `cmpf` lhs" lhs
  let (rhsElem, rhsShape) ← ensureFloatScalarTile "TileIR `cmpf` rhs" rhs
  unless lhsElem == rhsElem && lhsShape == rhsShape do
    throw s!"TileIR `cmpf` expects matching operand types, but got {lhs.ty.render} and {rhs.ty.render}"
  let dstTy ← comparisonResultTy "TileIR `cmpf`" lhsShape dstTy?
  let dst ← freshValue hint dstTy
  emit (.cmpf dst pred mode lhs.name rhs.name lhs.ty)
  pure dst

def cmpi
    (hint : String)
    (pred : ComparisonPredicate)
    (lhs rhs : Value)
    (signedness : Signedness := .signed)
    (dstTy? : Option TileType := none)
    : EntryM Value := do
  let (lhsElem, lhsShape) ← tileScalarShape "TileIR `cmpi` lhs" lhs
  let (rhsElem, rhsShape) ← tileScalarShape "TileIR `cmpi` rhs" rhs
  unless lhsElem == rhsElem && lhsShape == rhsShape do
    throw s!"TileIR `cmpi` expects matching operand types, but got {lhs.ty.render} and {rhs.ty.render}"
  unless lhsElem.isIntegral do
    throw s!"TileIR `cmpi` expects integral scalar tiles, but got {lhs.ty.render}"
  let dstTy ← comparisonResultTy "TileIR `cmpi`" lhsShape dstTy?
  let dst ← freshValue hint dstTy
  emit (.cmpi dst pred lhs.name rhs.name signedness lhs.ty)
  pure dst

def cat (hint : String) (lhs rhs : Value) (dim : Nat) : EntryM Value := do
  let (lhsElem, lhsShape) ← tileElemTypeShape "TileIR `ct.cat` lhs" lhs
  let (rhsElem, rhsShape) ← tileElemTypeShape "TileIR `ct.cat` rhs" rhs
  unless lhsElem == rhsElem do
    throw s!"TileIR `ct.cat` expects matching element types, but got {lhs.ty.render} and {rhs.ty.render}"
  unless lhsShape.size == rhsShape.size do
    throw s!"TileIR `ct.cat` expects matching tile rank, but got {lhs.ty.render} and {rhs.ty.render}"
  unless dim < lhsShape.size do
    throw s!"TileIR `ct.cat` axis {dim} is out of bounds for rank {lhsShape.size}"
  let mut outShape := lhsShape
  for i in [0:lhsShape.size] do
    if i != dim && lhsShape[i]! != rhsShape[i]! then
      throw s!"TileIR `ct.cat` expects matching non-concatenated dimensions, but got {renderNatShape lhsShape} and {renderNatShape rhsShape}"
    if i == dim then
      outShape := outShape.set! i (lhsShape[i]! + rhsShape[i]!)
  ensurePowerOfTwoShape "TileIR `ct.cat`" outShape
  let dstTy := tileTypeFromShape lhsElem outShape
  let dst ← freshValue hint dstTy
  emit (.cat dst lhs.name rhs.name dim lhs.ty rhs.ty)
  pure dst

def mmaf (hint : String) (a b c : Value) (dstTy? : Option TileType := none) : EntryM Value := do
  let (aElem, aShape) ← ensureFloatScalarTile "TileIR `ct.mma` lhs" a
  let (bElem, bShape) ← ensureFloatScalarTile "TileIR `ct.mma` rhs" b
  let (cElem, cShape) ← ensureFloatScalarTile "TileIR `ct.mma` accumulator" c
  unless aShape.size == 2 && bShape.size == 2 && cShape.size == 2 do
    throw s!"TileIR `ct.mma` expects rank-2 tile operands, but got {a.ty.render}, {b.ty.render}, and {c.ty.render}"
  unless aShape[1]! == bShape[0]! && aShape[0]! == cShape[0]! && bShape[1]! == cShape[1]! do
    throw s!"TileIR `ct.mma` shape mismatch: lhs={renderNatShape aShape}, rhs={renderNatShape bShape}, acc={renderNatShape cShape}"
  unless cElem.isFloat && aElem.isFloat && bElem.isFloat do
    throw s!"TileIR `ct.mma` expects floating-point tiles, but got {a.ty.render}, {b.ty.render}, and {c.ty.render}"
  let dst ← freshValue hint (dstTy?.getD c.ty)
  emit (.mmaf dst a.name b.name c.name a.ty b.ty c.ty)
  pure dst

def mmai
    (hint : String)
    (a b c : Value)
    (aSigned : Signedness := .signed)
    (bSigned : Signedness := .signed)
    (dstTy? : Option TileType := none)
    : EntryM Value := do
  let (aElem, aShape) ← ensureNumericScalarTile "TileIR `ct.mmai` lhs" a
  let (bElem, bShape) ← ensureNumericScalarTile "TileIR `ct.mmai` rhs" b
  let (cElem, cShape) ← ensureNumericScalarTile "TileIR `ct.mmai` accumulator" c
  unless aElem.isIntegral && bElem.isIntegral && cElem.isIntegral do
    throw s!"TileIR `ct.mmai` expects integral tiles, but got {a.ty.render}, {b.ty.render}, and {c.ty.render}"
  unless aShape.size == 2 && bShape.size == 2 && cShape.size == 2 do
    throw s!"TileIR `ct.mmai` expects rank-2 tile operands, but got {a.ty.render}, {b.ty.render}, and {c.ty.render}"
  unless aShape[1]! == bShape[0]! && aShape[0]! == cShape[0]! && bShape[1]! == cShape[1]! do
    throw s!"TileIR `ct.mmai` shape mismatch: lhs={renderNatShape aShape}, rhs={renderNatShape bShape}, acc={renderNatShape cShape}"
  let dst ← freshValue hint (dstTy?.getD c.ty)
  emit (.mmai dst a.name b.name c.name a.ty b.ty c.ty aSigned bSigned)
  pure dst

def broadcast (hint : String) (src : Value) (dstTy : TileType) : EntryM Value := do
  let some srcElem := src.ty.elemType?
    | throw s!"TileIR `ct.broadcast` expected a tile source, but got {src.ty.render}"
  let some dstElem := dstTy.elemType?
    | throw s!"TileIR `ct.broadcast` expected a tile destination type, but got {dstTy.render}"
  unless src.ty.staticShape? == some #[] do
    throw s!"TileIR `ct.broadcast` expects a scalar-tile source, but got {src.ty.render}"
  unless srcElem == dstElem do
    throw s!"TileIR `ct.broadcast` element mismatch: source {src.ty.render}, destination {dstTy.render}"
  let dst ← freshValue hint dstTy
  emit (.broadcast dst src.name src.ty)
  pure dst

def reshape (hint : String) (src : Value) (dstTy : TileType) : EntryM Value := do
  let some srcElem := src.ty.elemType?
    | throw s!"TileIR `ct.reshape` expected a tile source, but got {src.ty.render}"
  let some dstElem := dstTy.elemType?
    | throw s!"TileIR `ct.reshape` expected a tile destination type, but got {dstTy.render}"
  unless srcElem == dstElem do
    throw s!"TileIR `ct.reshape` element mismatch: source {src.ty.render}, destination {dstTy.render}"
  match src.ty.staticShape?, dstTy.staticShape? with
  | some srcShape, some dstShape =>
      ensureStaticElemCountEq "TileIR `ct.reshape`" srcShape dstShape
  | _, _ =>
      pure ()
  let dst ← freshValue hint dstTy
  emit (.reshape dst src.name src.ty)
  pure dst

def reshapeLike (hint : String) (src : Value) (shape : Array Nat) : EntryM Value := do
  let (elem, _) ← tileElemTypeShape "TileIR surface `ct.reshape`" src
  reshape hint src (tileTypeFromShape elem shape)

def permute (hint : String) (src : Value) (permutation : Array Nat) : EntryM Value := do
  let (elem, srcShape) ← tileElemTypeShape "TileIR `ct.permute`" src
  ensurePermutation "TileIR `ct.permute`" permutation srcShape.size
  let dstTy := tileTypeFromShape elem (permuteShape srcShape permutation)
  let dst ← freshValue hint dstTy
  emit (.permute dst src.name permutation src.ty)
  pure dst

def extract (hint : String) (src : Value) (indices : Array Value) (shape : Array Nat) : EntryM Value := do
  let (elem, srcShape) ← tileElemTypeShape "TileIR `ct.extract`" src
  unless indices.size == srcShape.size do
    throw s!"TileIR `ct.extract` expects {srcShape.size} index tile(s), but got {indices.size}"
  for idx in indices do
    let (idxElem, idxShape) ← tileScalarShape "TileIR `ct.extract` index" idx
    unless isIndexScalar idxElem do
      throw s!"TileIR `ct.extract` expects integral scalar indices, but got {idx.ty.render}"
    unless idxShape.isEmpty do
      throw s!"TileIR `ct.extract` expects scalar index tiles, but got {idx.ty.render}"
  unless shape.size == srcShape.size do
    throw s!"TileIR `ct.extract` expects a result shape with rank {srcShape.size}, but got {renderNatShape shape}"
  for i in [0:shape.size] do
    let dstDim := shape[i]!
    let srcDim := srcShape[i]!
    unless dstDim != 0 && srcDim % dstDim == 0 do
      throw s!"TileIR `ct.extract` requires each result dimension to evenly divide the source shape, but got source={renderNatShape srcShape} result={renderNatShape shape}"
  let dstTy := tileTypeFromShape elem shape
  let dst ← freshValue hint dstTy
  emit (.extract dst src.name (indices.map Value.name) src.ty)
  pure dst

def select (hint : String) (cond valIfTrue valIfFalse : Value) : EntryM Value := do
  let (condElem, condShape) ← tileScalarShape "TileIR `ct.where` condition" cond
  let (trueElem, trueShape) ← tileElemTypeShape "TileIR `ct.where` true branch" valIfTrue
  let (falseElem, falseShape) ← tileElemTypeShape "TileIR `ct.where` false branch" valIfFalse
  unless condElem == .i1 do
    throw s!"TileIR `ct.where` expects an i1 condition tile, but got {cond.ty.render}"
  unless trueElem == falseElem && trueShape == falseShape do
    throw s!"TileIR `ct.where` expects matching value tile types, but got {valIfTrue.ty.render} and {valIfFalse.ty.render}"
  unless condShape == trueShape do
    throw s!"TileIR `ct.where` expects the condition tile to match the value tile shape, but got cond={renderNatShape condShape} values={renderNatShape trueShape}"
  let dst ← freshValue hint valIfTrue.ty
  emit (.select dst cond.name valIfTrue.name valIfFalse.name cond.ty valIfTrue.ty)
  pure dst

def astype (hint : String) (src : Value) (dtype : ScalarType) : EntryM Value := do
  let (srcElem, shape) ← tileScalarShape "TileIR surface `ct.astype`" src
  if srcElem == dtype then
    pure src
  else
    let dstTy := scalarLikeTileTy dtype shape
    let op ← castOpFor srcElem dtype
    let dst ← freshValue hint dstTy
    emit (.cast dst op src.name src.ty)
    pure dst

namespace Value

def astype (value : Value) (dtype : ScalarType) : EntryM Value :=
  Tyr.GPU.Codegen.TileIR.astype "astype" value dtype

def reshape (value : Value) (shape : Array Nat) : EntryM Value :=
  Tyr.GPU.Codegen.TileIR.reshapeLike "reshape" value shape

def permute (value : Value) (axes : Array Nat) : EntryM Value :=
  Tyr.GPU.Codegen.TileIR.permute "permute" value axes

def extract (value : Value) (indices : Array Value) (shape : Array Nat) : EntryM Value :=
  Tyr.GPU.Codegen.TileIR.extract "extract" value indices shape

end Value

def fullConst [ToTileLiteral α]
    (hint : String)
    (shape : Array Nat)
    (value : α)
    (dtype : ScalarType)
    : EntryM Value := do
  let scalar ← constScalarOf (hint ++ "_scalar") dtype value
  if shape.isEmpty then
    pure scalar
  else
    broadcast hint scalar (scalarLikeTileTy dtype shape)

def zeros (hint : String) (shape : Array Nat) (dtype : ScalarType) : EntryM Value := do
  let scalar ← const (hint ++ "_scalar") (scalarTileTy dtype) (defaultZeroLiteral dtype)
  if shape.isEmpty then
    pure scalar
  else
    broadcast hint scalar (scalarLikeTileTy dtype shape)

def ones (hint : String) (shape : Array Nat) (dtype : ScalarType) : EntryM Value := do
  let scalar ← const (hint ++ "_scalar") (scalarTileTy dtype) (defaultOneLiteral dtype)
  if shape.isEmpty then
    pure scalar
  else
    broadcast hint scalar (scalarLikeTileTy dtype shape)

def fill (hint : String) (shape : Array Nat) (value : Value) (dtype : ScalarType) : EntryM Value := do
  let (elem, dims) ← tileScalarShape "TileIR surface `ct.full`" value
  unless dims.isEmpty do
    throw s!"TileIR surface `ct.full` expects a scalar fill value, but got {value.ty.render}"
  let scalar ←
    if elem == dtype then
      pure value
    else
      astype (hint ++ "_cast") value dtype
  if shape.isEmpty then
    pure scalar
  else
    broadcast hint scalar (scalarLikeTileTy dtype shape)

instance : FillValue Value where
  build := fill

instance [ToTileLiteral α] : FillValue α where
  build := fullConst

def full [FillValue α]
    (hint : String)
    (shape : Array Nat)
    (value : α)
    (dtype : ScalarType)
    : EntryM Value :=
  FillValue.build hint shape value dtype

def offset (hint : String) (ptr idx : Value) (dstTy : TileType) : EntryM Value := do
  let dst ← freshValue hint dstTy
  emit (.offset dst ptr.name idx.name ptr.ty idx.ty)
  pure dst

def makeTensorView (hint : String) (base : Value) (desc : TensorViewType) : EntryM Value := do
  let dstTy : TileType := .tensorView desc
  let dst ← freshValue hint dstTy
  emit (.makeTensorView dst base.name desc.shape desc.strides)
  pure dst

def makePartitionView (hint : String) (src : Value) (desc : PartitionViewType) : EntryM Value := do
  let dstTy : TileType := .partitionView desc
  let dst ← freshValue hint dstTy
  emit (.makePartitionView dst src.name)
  pure dst

def getGlobal (hint : String) (globalName : String) (ty : TileType) : EntryM Value := do
  let dst ← freshValue hint ty
  emit (.getGlobal dst globalName)
  pure dst

def loadPtrTko
    (hint : String)
    (ptr inputToken : Value)
    (valueTy : TileType)
    (order : MemoryOrder := .weak)
    : EntryM ValueAndToken := do
  let value ← freshValue hint valueTy
  let token ← freshValue "tok" .token
  emit (.loadPtrTko value token order ptr.name inputToken.name ptr.ty)
  pure { value, token }

def loadViewTko
    (hint : String)
    (view : Value)
    (indices : Array Value)
    (inputToken : Value)
    (valueTy : TileType)
    (order : MemoryOrder := .weak)
    : EntryM ValueAndToken := do
  let value ← freshValue hint valueTy
  let token ← freshValue "tok" .token
  emit (.loadViewTko value token order view.name (indices.map Value.name) inputToken.name view.ty)
  pure { value, token }

def load (hint : String) (ptr : Value) (valueTy : TileType) : EntryM Value := do
  let some ptrElem := ptrElementScalar? ptr
    | throw s!"TileIR `ct.load` expected a pointer-typed source, but got {ptr.ty.render}"
  let some valueElem := valueTy.literalScalar?
    | throw s!"TileIR `ct.load` expected a scalar-tile destination type, but got {valueTy.render}"
  unless ptrElem == valueElem do
    throw s!"TileIR `ct.load` element mismatch: pointer {ptr.ty.render}, value {valueTy.render}"
  let inputToken ← currentToken
  let { value, token } ← loadPtrTko hint ptr inputToken valueTy
  setCurrentToken token
  pure value

def loadView (hint : String) (view : Value) (indices : Array Value) (valueTy : TileType) : EntryM Value := do
  let some valueElem := valueTy.literalScalar?
    | throw s!"TileIR `ct.load_view` expected a scalar-tile destination type, but got {valueTy.render}"
  let viewElem :=
    match view.ty with
    | .tensorView desc => some desc.elem
    | .partitionView desc => some desc.tensor.elem
    | _ => none
  let some viewElem := viewElem
    | throw s!"TileIR `ct.load_view` expected a tensor or partition view, but got {view.ty.render}"
  unless viewElem == valueElem do
    throw s!"TileIR `ct.load_view` element mismatch: view {view.ty.render}, value {valueTy.render}"
  let inputToken ← currentToken
  let { value, token } ← loadViewTko hint view indices inputToken valueTy
  setCurrentToken token
  pure value

def loadShape (hint : String) (ptr : Value) (shape : Array Nat) : EntryM Value := do
  let some elem := ptrElementScalar? ptr
    | throw s!"TileIR surface `ct.load` expected a pointer-typed source, but got {ptr.ty.render}"
  load hint ptr (tileTy elem shape)

def loadAt (hint : String) (ptr idx : Value) (shape : Array Nat) : EntryM Value := do
  let shifted ← offset (hint ++ "_ptr") ptr idx ptr.ty
  loadShape hint shifted shape

def gather (hint : String) (base idx : Value) : EntryM Value := do
  let shiftedTy ← offsetResultPtrTy "TileIR surface `ct.gather`" base idx
  let valueTy ← gatherValueTy "TileIR surface `ct.gather`" base idx
  let shifted ← offset (hint ++ "_ptr") base idx shiftedTy
  load hint shifted valueTy

def loadTiled (hint : String) (base : Value) (indices : Array Value) (tileShape : Array Nat) : EntryM Value := do
  if tileShape.isEmpty then
    if h : indices.size = 1 then
      return (← loadAt hint base indices[0] #[])
    else
      throw s!"TileIR surface indexed `ct.load` expected exactly one scalar index for shape (), but got {indices.size}"
  if indices.size != tileShape.size then
    throw s!"TileIR surface indexed `ct.load` expected {tileShape.size} index value(s) for tile shape {tileShape}, but got {indices.size}"
  let tensor ← directTensorViewType base tileShape.size
  let view ← makeTensorView (hint ++ "_view") base tensor
  let partition ← makePartitionView (hint ++ "_partition") view {
    tileShape := tileShape
    tensor := tensor
    dimMap := #[]
  }
  let some elem := ptrElementScalar? base
    | throw s!"TileIR surface indexed `ct.load` expected a pointer-typed source, but got {base.ty.render}"
  loadView hint partition indices (tileTy elem tileShape)

def storePtrTko
    (ptr value inputToken : Value)
    (order : MemoryOrder := .weak)
    (hint : String := "tok")
    : EntryM Value := do
  let token ← freshValue hint .token
  emit (.storePtrTko token order ptr.name value.name inputToken.name ptr.ty value.ty)
  pure token

def storeViewTko
    (view : Value)
    (indices : Array Value)
    (value inputToken : Value)
    (order : MemoryOrder := .weak)
    (hint : String := "tok")
    : EntryM Value := do
  let token ← freshValue hint .token
  emit (.storeViewTko token order view.name (indices.map Value.name) value.name inputToken.name view.ty value.ty)
  pure token

def store (ptr value : Value) (order : MemoryOrder := .weak) : EntryM Unit := do
  let some ptrElem := ptrElementScalar? ptr
    | throw s!"TileIR `ct.store` expected a pointer-typed destination, but got {ptr.ty.render}"
  let some valueElem := value.ty.literalScalar?
    | throw s!"TileIR `ct.store` expected a scalar-tile value, but got {value.ty.render}"
  unless ptrElem == valueElem do
    throw s!"TileIR `ct.store` element mismatch: pointer {ptr.ty.render}, value {value.ty.render}"
  let inputToken ← currentToken
  let token ← storePtrTko ptr value inputToken order
  setCurrentToken token

def storeAt (ptr idx value : Value) (order : MemoryOrder := .weak) : EntryM Unit := do
  let shifted ← offset "store_ptr" ptr idx ptr.ty
  store shifted value order

def scatter (base idx value : Value) (order : MemoryOrder := .weak) : EntryM Unit := do
  let context := "TileIR surface `ct.scatter`"
  let shiftedTy ← offsetResultPtrTy context base idx
  let expectedTy ← gatherValueTy context base idx
  if value.ty != expectedTy then
    throw s!"{context} expected a value of type {expectedTy.render}, but got {value.ty.render}"
  let shifted ← offset "scatter_ptr" base idx shiftedTy
  store shifted value order

def storeTiled (base : Value) (indices : Array Value) (value : Value) (order : MemoryOrder := .weak) : EntryM Unit := do
  let shape :=
    match value.ty with
    | .tile dims (.scalar _) => dims
    | .tile dims (.ptr _) => dims
    | _ => #[]
  if shape.isEmpty then
    if h : indices.size = 1 then
      storeAt base indices[0] value order
    else
      throw s!"TileIR surface indexed `ct.store` expected exactly one scalar index for value type {value.ty.render}, but got {indices.size}"
    return ()
  if indices.size != shape.size then
    throw s!"TileIR surface indexed `ct.store` expected {shape.size} index value(s) for value type {value.ty.render}, but got {indices.size}"
  let mut tileShape : Array Nat := #[]
  for dim in shape do
    match dim with
    | .static n =>
        tileShape := tileShape.push n
    | .dynamic =>
        throw "TileIR surface indexed `ct.store` requires statically known tile shapes"
  let tensor ← directTensorViewType base tileShape.size
  let view ← makeTensorView "store_view" base tensor
  let partition ← makePartitionView "store_partition" view {
    tileShape := tileShape
    tensor := tensor
    dimMap := #[]
  }
  let inputToken ← currentToken
  let token ← storeViewTko partition indices value inputToken order
  setCurrentToken token

def storeView (view : Value) (indices : Array Value) (value : Value) (order : MemoryOrder := .weak) : EntryM Unit := do
  let some valueElem := value.ty.literalScalar?
    | throw s!"TileIR `ct.store_view` expected a scalar-tile value, but got {value.ty.render}"
  let viewElem :=
    match view.ty with
    | .tensorView desc => some desc.elem
    | .partitionView desc => some desc.tensor.elem
    | _ => none
  let some viewElem := viewElem
    | throw s!"TileIR `ct.store_view` expected a tensor or partition view, but got {view.ty.render}"
  unless viewElem == valueElem do
    throw s!"TileIR `ct.store_view` element mismatch: view {view.ty.render}, value {value.ty.render}"
  let inputToken ← currentToken
  let token ← storeViewTko view indices value inputToken order
  setCurrentToken token

def printTko (message : String) (hint : String := "tok") : EntryM Value := do
  let token ← freshValue hint .token
  emit (.printTko token message)
  pure token

def print (message : String) : EntryM Unit := do
  let token ← printTko message
  setCurrentToken token

def assert (cond : Value) (message : String) : EntryM Unit :=
  emit (.assertOp cond.name cond.ty message)

def staticAssert (cond : Bool) (mkMessage : Unit → String := fun _ => "") : EntryM Unit := do
  if cond then
    pure ()
  else
    let message := mkMessage ()
    if message.isEmpty then
      throw "TileIR static assertion failed"
    else
      throw s!"TileIR static assertion failed: {message}"

def loadGlobal (hint : String) (globalName : String) (valueTy : TileType) : EntryM Value := do
  let some elem := valueTy.literalScalar?
    | throw s!"TileIR surface `ct.load_global` expected a scalar/tile value type, but got {valueTy.render}"
  let ptr ← getGlobal (hint ++ "_ptr") globalName (ptrTileTy elem)
  load hint ptr valueTy

def if_
    (cond : Value)
    (thenBranch : EntryM (Array Value))
    (elseBranch : EntryM (Array Value))
    (hint : String := "if")
    : EntryM (Array Value) := do
  let inputToken? := (← get).currentToken?
  let (thenValues, thenStmts, thenToken?) ← captureBlock thenBranch
  let (elseValues, elseStmts, elseToken?) ← captureBlock elseBranch
  let userResults? := !thenValues.isEmpty
  let mergeToken? := inputToken?.isSome || (thenToken?.isSome && elseToken?.isSome)
  if !userResults? && !mergeToken? then
    throw "TileIR builder `if_` requires at least one yielded result"
  let tys ← ensureMatchingTypes "TileIR builder `if_`" thenValues elseValues
  let resultTys :=
    if mergeToken? then
      tys.push .token
    else
      tys
  let results ← freshResults hint resultTys
  let hiddenTokenResult? :=
    if mergeToken? then
      some results[tys.size]!
    else
      none
  let thenYieldValues := thenValues.map Value.name
  let elseYieldValues := elseValues.map Value.name
  let thenYieldValues :=
    match hiddenTokenResult? with
    | none => thenYieldValues
    | some _ =>
        let thenToken :=
          match thenToken?, inputToken? with
          | some token, _ => token.name
          | none, some token => token.name
          | none, none => panic! "TileIR builder token merge invariant violated in `if_`"
        thenYieldValues.push thenToken
  let elseYieldValues :=
    match hiddenTokenResult? with
    | none => elseYieldValues
    | some _ =>
        let elseToken :=
          match elseToken?, inputToken? with
          | some token, _ => token.name
          | none, some token => token.name
          | none, none => panic! "TileIR builder token merge invariant violated in `if_`"
        elseYieldValues.push elseToken
  emit <| .ifOp
    (results.map Value.binding)
    cond.name
    (thenStmts.push (.yieldOp thenYieldValues))
    (elseStmts.push (.yieldOp elseYieldValues))
  if let some token := hiddenTokenResult? then
    setCurrentToken token
  pure <| results.extract 0 tys.size

def if1
    (cond : Value)
    (thenBranch : EntryM Value)
    (elseBranch : EntryM Value)
    (hint : String := "if")
    : EntryM Value := do
  let results ← if_ cond (do pure #[← thenBranch]) (do pure #[← elseBranch]) hint
  pure results[0]!

def continue1 (value : Value) : LoopResult :=
  .continue_ #[value]

def break1 (value : Value) : LoopResult :=
  .break_ #[value]

def for_
    (lower upper step : Value)
    (iterInits : Array Value)
    (body : Value → Array Value → EntryM LoopResult)
    (ivHint : String := "iv")
    (hint : String := "loop")
    : EntryM (Array Value) := do
  let outerToken? := (← get).currentToken?
  let hiddenTokenBinder? ←
    match outerToken? with
    | some _ => some <$> freshValue "tok_iter" .token
    | none => pure none
  if iterInits.isEmpty && hiddenTokenBinder?.isNone then
    throw "TileIR builder `for_` requires at least one loop-carried value"
  let iv ← freshValue ivHint lower.ty
  let mut userCarries : Array Value := #[]
  for init in iterInits do
    userCarries := userCarries.push (← freshValue (init.name ++ "_iter") init.ty)
  let (loopResult, bodyStmts, bodyToken?) ←
    captureBlockWithToken? hiddenTokenBinder? (body iv userCarries)
  let yielded :=
    match loopResult with
    | .continue_ values => values
    | .break_ values => values
  let tys ← ensureMatchingTypes "TileIR builder `for_`" iterInits yielded
  let resultTys :=
    match hiddenTokenBinder? with
    | some _ => tys.push .token
    | none => tys
  let results ← freshResults hint resultTys
  let hiddenTokenResult? :=
    match hiddenTokenBinder? with
    | some _ => some results[tys.size]!
    | none => none
  let mut iterValues : Array LoopCarry := #[]
  for i in [0:iterInits.size] do
    iterValues := iterValues.push {
      binder := userCarries[i]!
      init := iterInits[i]!.name
    }
  if let some tokenBinder := hiddenTokenBinder? then
    let some tokenInit := outerToken?
      | panic! "TileIR builder token loop invariant violated in `for_`"
    iterValues := iterValues.push {
      binder := tokenBinder
      init := tokenInit.name
    }
  let yieldedValues :=
    match hiddenTokenBinder?, bodyToken? with
    | none, _ => yielded.map Value.name
    | some _, some token => (yielded.map Value.name).push token.name
    | some _tokenBinder, none => (yielded.map Value.name).push _tokenBinder.name
  let terminator :=
    match loopResult with
    | .continue_ _ => .continueOp yieldedValues
    | .break_ _ => .breakOp yieldedValues
  emit <| .forOp
    (results.map Value.binding)
    iv
    lower.name
    upper.name
    step.name
    iterValues
    (bodyStmts.push terminator)
  if let some token := hiddenTokenResult? then
    setCurrentToken token
  pure <| results.extract 0 tys.size

def for1
    (lower upper step init : Value)
    (body : Value → Value → EntryM LoopResult)
    (ivHint : String := "iv")
    (hint : String := "loop")
    : EntryM Value := do
  let results ← for_ lower upper step #[init] (fun iv carries => body iv carries[0]!) ivHint hint
  pure results[0]!

def buildEntry? (name : String) (params : Array Value) (body : EntryM Unit) : Except String Entry := do
  let (result, st) := body.run { currentToken? := initialToken? params }
  match result with
  | .error err => throw err
  | .ok _ =>
      pure {
        name := name
        params := params.map Value.param
        body := st.stmts
      }

def buildEntry! (name : String) (params : Array Value) (body : EntryM Unit) : Entry :=
  match buildEntry? name params body with
  | .ok entry => entry
  | .error err => panic! s!"Invalid TileIR entry '{name}': {err}"

def global (name : String) (ty : TileType) (value : Literal) : ModuleM Unit :=
  modify fun st => {
    st with globals := st.globals.push { name, ty, value }
  }

def entry (name : String) (params : Array Value) (body : EntryM Unit) : ModuleM Unit := do
  let built ←
    match buildEntry? name params body with
    | .ok result => pure result
    | .error err => throw err
  modify fun st => {
    st with entries := st.entries.push built
  }

def buildModule? (name : String) (body : ModuleM Unit) : Except String Module := do
  let (result, st) := body.run {}
  match result with
  | .error err => throw err
  | .ok _ =>
      pure {
        name := name
        globals := st.globals
        entries := st.entries
      }

def module_ (name : String) (body : ModuleM Unit) : Module :=
  match buildModule? name body with
  | .ok mod => mod
  | .error err => panic! s!"Invalid TileIR module '{name}': {err}"

end Tyr.GPU.Codegen.TileIR
