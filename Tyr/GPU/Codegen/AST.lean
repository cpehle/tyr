/-
  Tyr/GPU/Codegen/AST.lean

  Kernel expression AST matching ThunderKittens operations.
  Used for type-checked kernel construction and C++ code generation.
-/
import Tyr.GPU.Types

namespace Tyr.GPU.Codegen

/-- MMA transpose modes -/
inductive MMATranspose where
  | AB    -- A row, B col (standard)
  | ABt   -- A row, B row (B transposed)
  | AtB   -- A col, B col (A transposed)
  | AtBt  -- A col, B row (both transposed)
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert MMATranspose to C++ function suffix -/
def MMATranspose.toSuffix : MMATranspose → String
  | .AB => "AB"
  | .ABt => "ABt"
  | .AtB => "AtB"
  | .AtBt => "AtBt"

/-- Reduction operations -/
inductive ReduceOp where
  | Max | Min | Sum | Prod
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert ReduceOp to C++ -/
def ReduceOp.toCpp : ReduceOp → String
  | .Max => "max"
  | .Min => "min"
  | .Sum => "sum"
  | .Prod => "prod"

/-- Reduction axis -/
inductive ReduceAxis where
  | Row | Col | Full
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert ReduceAxis to C++ function prefix -/
def ReduceAxis.toPrefix : ReduceAxis → String
  | .Row => "row_"
  | .Col => "col_"
  | .Full => ""

/-- Unary element-wise operations -/
inductive UnaryOp where
  | Exp | Exp2 | Log | Log2 | Abs | Relu | Copy
  | Zero | One | PosInfty | NegInfty  -- Initialization
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert UnaryOp to C++ -/
def UnaryOp.toCpp : UnaryOp → String
  | .Exp => "exp"
  | .Exp2 => "exp2"
  | .Log => "log"
  | .Log2 => "log2"
  | .Abs => "abs"
  | .Relu => "relu"
  | .Copy => "copy"
  | .Zero => "zero"
  | .One => "one"
  | .PosInfty => "pos_infty"
  | .NegInfty => "neg_infty"

/-- Binary element-wise operations -/
inductive BinaryOp where
  | Add | Sub | Mul | Div | Max | Min
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert BinaryOp to C++ -/
def BinaryOp.toCpp : BinaryOp → String
  | .Add => "add"
  | .Sub => "sub"
  | .Mul => "mul"
  | .Div => "div"
  | .Max => "max"
  | .Min => "min"

/-- Broadcast axis -/
inductive BroadcastAxis where
  | Row | Col
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert BroadcastAxis to C++ suffix -/
def BroadcastAxis.toSuffix : BroadcastAxis → String
  | .Row => "_row"
  | .Col => "_col"

/-- Masking operations -/
inductive MaskOp where
  | Tril (diagonal : Int)
  | Triu (diagonal : Int)
  | MakeCausal
  | RightFill (colIdx : Nat)
  | LeftFill (colIdx : Nat)
  | UpperFill (rowIdx : Nat)
  | LowerFill (rowIdx : Nat)
  deriving Repr, BEq, Hashable, Inhabited

/-- Kernel expression AST -/
inductive KExpr where
  -- Tile declarations
  | declRT (name : String) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | declST (name : String) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | declRV (name : String) (dtype : GpuFloat) (len : Nat)  -- Register vector
  | declSV (name : String) (dtype : GpuFloat) (len : Nat)  -- Shared vector

  -- Memory operations
  | load (dst src : String)           -- load(dst, src) - S→R or G→S/R
  | store (dst src : String)          -- store(dst, src) - R→S or S/R→G
  | loadAsync (dst src : String)      -- load_async for TMA

  -- MMA operations
  | mma (trans : MMATranspose) (dst a b c : String)  -- d = a × b + c
  | mm (trans : MMATranspose) (dst a b : String)     -- d = a × b (no accumulate)
  | mmaFence (dst : String)           -- Hopper WGMMA fence
  | mmaCommitGroup                    -- Commit WGMMA group
  | mmaAsyncWait (n : Nat)            -- Wait for n groups

  -- Element-wise unary
  | unary (op : UnaryOp) (dst src : String)

  -- Element-wise binary
  | binary (op : BinaryOp) (dst a b : String)

  -- Broadcasting (row/col vector to tile)
  | broadcast (axis : BroadcastAxis) (dst vec : String)
  | binaryBroadcast (op : BinaryOp) (axis : BroadcastAxis) (dst tile vec : String)

  -- Reductions
  | reduce (op : ReduceOp) (axis : ReduceAxis) (dst src : String)
  | reduceAccum (op : ReduceOp) (axis : ReduceAxis) (dst src accum : String)

  -- Layout/type conversions
  | swapLayout (dst src : String)
  | transpose (dst src : String)
  | convert (dst src : String)        -- Type conversion (e.g., bf16 → float)

  -- Masking
  | mask (op : MaskOp) (dst src : String) (fillVal : Option Float)

  -- Synchronization
  | sync (barrierId : Nat)
  | arrive (barrierId : Nat)

  -- Control flow
  | seq (a b : KExpr)
  | forLoop (var : String) (lo hi : Nat) (body : KExpr)
  | comment (text : String)           -- C++ comment for readability

  deriving Repr, Inhabited

/-- Helper to sequence multiple expressions -/
def KExpr.seqAll : List KExpr → KExpr
  | [] => .comment "noop"
  | [e] => e
  | e :: es => .seq e (seqAll es)

/-- Kernel parameter specification -/
structure KernelParam where
  name : String
  cppType : String
  isPointer : Bool := false
  deriving Repr, Inhabited

/-- Kernel definition -/
structure KernelDef where
  name : String
  arch : GpuArch
  /-- Kernel parameters -/
  params : List KernelParam := []
  /-- Shared memory size requirement -/
  sharedMemBytes : Nat := 0
  /-- Kernel body -/
  body : KExpr
  deriving Repr, Inhabited

end Tyr.GPU.Codegen
