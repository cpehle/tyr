import Tyr.GPU.Codegen.TileIR.Type

/-!
# Tyr.GPU.Codegen.TileIR.Expr

Lean-side AST for public NVIDIA TileIR operations.

The intent is to stay close to the published MLIR syntax so that rendering to
`cuda_tile.*` textual IR is straightforward and predictable.
-/

namespace Tyr.GPU.Codegen.TileIR

structure Binding where
  name : String
  ty : TileType
  deriving Repr, Inhabited, BEq, DecidableEq

structure Param where
  name : String
  ty : TileType
  deriving Repr, Inhabited, BEq, DecidableEq

/-- Untyped literal payloads. The enclosing TileIR type determines the element type. -/
inductive Literal where
  | int (value : Int)
  | float (value : Float)
  | bool (value : Bool)
  | array (items : Array Literal)
  deriving Repr, Inhabited, BEq

structure Global where
  name : String
  ty : TileType
  value : Literal
  deriving Repr, Inhabited, BEq

inductive UnaryOp where
  | copy
  | exp
  | exp2
  | log
  | sqrt
  | rsqrt
  | abs
  | neg
  deriving Repr, Inhabited, BEq, DecidableEq

namespace UnaryOp

def render : UnaryOp → String
  | .copy => "copy"
  | .exp => "exp"
  | .exp2 => "exp2"
  | .log => "log"
  | .sqrt => "sqrt"
  | .rsqrt => "rsqrt"
  | .abs => "abs"
  | .neg => "neg"

end UnaryOp

inductive BinaryOp where
  | addf
  | subf
  | mulf
  | divf
  | maxf
  | minf
  deriving Repr, Inhabited, BEq, DecidableEq

namespace BinaryOp

def render : BinaryOp → String
  | .addf => "addf"
  | .subf => "subf"
  | .mulf => "mulf"
  | .divf => "divf"
  | .maxf => "maxf"
  | .minf => "minf"

end BinaryOp

inductive ComparisonPredicate where
  | equal
  | notEqual
  | lessThan
  | lessThanOrEqual
  | greaterThan
  | greaterThanOrEqual
  deriving Repr, Inhabited, BEq, DecidableEq

namespace ComparisonPredicate

def render : ComparisonPredicate → String
  | .equal => "equal"
  | .notEqual => "not_equal"
  | .lessThan => "less_than"
  | .lessThanOrEqual => "less_than_or_equal"
  | .greaterThan => "greater_than"
  | .greaterThanOrEqual => "greater_than_or_equal"

end ComparisonPredicate

inductive FloatCompareMode where
  | ordered
  | unordered
  deriving Repr, Inhabited, BEq, DecidableEq

namespace FloatCompareMode

def render : FloatCompareMode → String
  | .ordered => "ordered"
  | .unordered => "unordered"

end FloatCompareMode

inductive MemoryOrder where
  | weak
  | relaxed
  | acquire
  | release
  | acqRel
  | seqCst
  deriving Repr, Inhabited, BEq, DecidableEq

namespace MemoryOrder

def render : MemoryOrder → String
  | .weak => "weak"
  | .relaxed => "relaxed"
  | .acquire => "acquire"
  | .release => "release"
  | .acqRel => "acq_rel"
  | .seqCst => "seq_cst"

end MemoryOrder

inductive Signedness where
  | signed
  | unsigned
  deriving Repr, Inhabited, BEq, DecidableEq

namespace Signedness

def render : Signedness → String
  | .signed => "signed"
  | .unsigned => "unsigned"

end Signedness

inductive RoundingMode where
  | nearestEven
  | zero
  | negativeInf
  | positiveInf
  | approx
  | full
  | nearestIntToZero
  deriving Repr, Inhabited, BEq, DecidableEq

namespace RoundingMode

def render : RoundingMode → String
  | .nearestEven => "nearest_even"
  | .zero => "zero"
  | .negativeInf => "negative_inf"
  | .positiveInf => "positive_inf"
  | .approx => "approx"
  | .full => "full"
  | .nearestIntToZero => "nearest_int_to_zero"

end RoundingMode

inductive CastOp where
  | bitcast
  | exti (signedness : Signedness)
  | trunci
  | ftof (rounding : RoundingMode)
  | ftoi (signedness : Signedness) (rounding : RoundingMode)
  | itof (signedness : Signedness) (rounding : RoundingMode)
  deriving Repr, Inhabited, BEq, DecidableEq

namespace CastOp

def render : CastOp → String
  | .bitcast => "bitcast"
  | .exti _ => "exti"
  | .trunci => "trunci"
  | .ftof _ => "ftof"
  | .ftoi _ _ => "ftoi"
  | .itof _ _ => "itof"

end CastOp

structure LoopCarry where
  binder : Binding
  init : String
  deriving Repr, Inhabited, BEq, DecidableEq

inductive Stmt where
  | comment (text : String)
  | const (dst : Binding) (value : Literal)
  | getNumTileBlocks (x y z : Binding)
  | getTileBlockId (x y z : Binding)
  | iota (dst : Binding)
  | unary (dst : Binding) (op : UnaryOp) (src : String)
  | binary (dst : Binding) (op : BinaryOp) (lhs rhs : String)
  | cmpf
      (dst : Binding)
      (pred : ComparisonPredicate)
      (mode : FloatCompareMode)
      (lhs rhs : String)
      (lhsTy : TileType)
  | cmpi
      (dst : Binding)
      (pred : ComparisonPredicate)
      (lhs rhs : String)
      (signedness : Signedness)
      (lhsTy : TileType)
  | cat (dst : Binding) (lhs rhs : String) (dim : Nat) (lhsTy rhsTy : TileType)
  | mmaf (dst : Binding) (a b c : String) (aTy bTy cTy : TileType)
  | mmai
      (dst : Binding)
      (a b c : String)
      (aTy bTy cTy : TileType)
      (aSigned : Signedness)
      (bSigned : Signedness)
  | cast (dst : Binding) (op : CastOp) (src : String) (srcTy : TileType)
  | broadcast (dst : Binding) (src : String) (srcTy : TileType)
  | reshape (dst : Binding) (src : String) (srcTy : TileType)
  | permute (dst : Binding) (src : String) (permutation : Array Nat) (srcTy : TileType)
  | extract (dst : Binding) (src : String) (indices : Array String) (srcTy : TileType)
  | select (dst : Binding) (cond valIfTrue valIfFalse : String) (condTy valueTy : TileType)
  | offset (dst : Binding) (ptr : String) (idx : String) (ptrTy idxTy : TileType)
  | makeTensorView (dst : Binding) (base : String) (shape strides : Array ShapeDim)
  | makePartitionView (dst : Binding) (src : String)
  | loadPtrTko
      (value : Binding)
      (token : Binding)
      (order : MemoryOrder)
      (ptr : String)
      (inputToken : String)
      (ptrTy : TileType)
  | loadViewTko
      (value : Binding)
      (token : Binding)
      (order : MemoryOrder)
      (view : String)
      (indices : Array String)
      (inputToken : String)
      (viewTy : TileType)
  | storePtrTko
      (token : Binding)
      (order : MemoryOrder)
      (ptr : String)
      (value : String)
      (inputToken : String)
      (ptrTy valueTy : TileType)
  | storeViewTko
      (token : Binding)
      (order : MemoryOrder)
      (view : String)
      (indices : Array String)
      (value : String)
      (inputToken : String)
      (viewTy valueTy : TileType)
  | printTko (token : Binding) (message : String)
  | assertOp (cond : String) (condTy : TileType) (message : String)
  | getGlobal (dst : Binding) (globalName : String)
  | ifOp
      (results : Array Binding)
      (cond : String)
      (thenBody elseBody : Array Stmt)
  | forOp
      (results : Array Binding)
      (iv : Binding)
      (lower upper step : String)
      (iterValues : Array LoopCarry)
      (body : Array Stmt)
  | yieldOp (values : Array String)
  | continueOp (values : Array String)
  | breakOp (values : Array String)
  deriving Repr, Inhabited, BEq

structure Entry where
  name : String
  params : Array Param
  body : Array Stmt
  deriving Repr, Inhabited, BEq

structure Module where
  name : String
  globals : Array Global := #[]
  entries : Array Entry := #[]
  deriving Repr, Inhabited, BEq

instance : ToString UnaryOp where
  toString := UnaryOp.render

instance : ToString BinaryOp where
  toString := BinaryOp.render

instance : ToString ComparisonPredicate where
  toString := ComparisonPredicate.render

instance : ToString FloatCompareMode where
  toString := FloatCompareMode.render

instance : ToString MemoryOrder where
  toString := MemoryOrder.render

instance : ToString Signedness where
  toString := Signedness.render

instance : ToString RoundingMode where
  toString := RoundingMode.render

instance : ToString CastOp where
  toString := CastOp.render

end Tyr.GPU.Codegen.TileIR
