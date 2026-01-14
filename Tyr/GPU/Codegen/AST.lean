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
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

instance : ToString MMATranspose where
  toString
    | .AB => "AB"
    | .ABt => "ABt"
    | .AtB => "AtB"
    | .AtBt => "AtBt"

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
  | Neg | Sqrt | Rsqrt | Tanh | FastTanh | Sigmoid | Gelu  -- Activation functions
  | Silu | Swish                                -- Modern activations (SwiGLU)
  | Sin | Cos                                   -- Trig (for rotary embeddings)
  | Recip                                       -- 1/x
  | Square                                      -- x^2
  | Zero | One | PosInfty | NegInfty            -- Initialization
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

instance : ToString UnaryOp where
  toString
    | .Exp => "Exp" | .Exp2 => "Exp2" | .Log => "Log" | .Log2 => "Log2"
    | .Abs => "Abs" | .Relu => "Relu" | .Copy => "Copy" | .Neg => "Neg"
    | .Sqrt => "Sqrt" | .Rsqrt => "Rsqrt" | .Tanh => "Tanh" | .FastTanh => "FastTanh"
    | .Sigmoid => "Sigmoid" | .Gelu => "Gelu"
    | .Silu => "Silu" | .Swish => "Swish"
    | .Sin => "Sin" | .Cos => "Cos"
    | .Recip => "Recip" | .Square => "Square"
    | .Zero => "Zero" | .One => "One"
    | .PosInfty => "PosInfty" | .NegInfty => "NegInfty"

/-- Convert UnaryOp to C++ -/
def UnaryOp.toCpp : UnaryOp → String
  | .Exp => "exp"
  | .Exp2 => "exp2"
  | .Log => "log"
  | .Log2 => "log2"
  | .Abs => "abs"
  | .Relu => "relu"
  | .Copy => "copy"
  | .Neg => "neg"
  | .Sqrt => "sqrt"
  | .Rsqrt => "rsqrt"
  | .Tanh => "tanh"
  | .FastTanh => "fast_tanh"  -- Hardware-accelerated tanh (__nv_fast_tanh)
  | .Sigmoid => "sigmoid"
  | .Gelu => "gelu"
  | .Silu => "silu"
  | .Swish => "swish"
  | .Sin => "sin"
  | .Cos => "cos"
  | .Recip => "recip"
  | .Square => "square"
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
  | MakeCausalT      -- Transpose causal mask (for backward pass)
  | RightFill (colIdx : Nat)
  | LeftFill (colIdx : Nat)
  | UpperFill (rowIdx : Nat)
  | LowerFill (rowIdx : Nat)
  | UpperRightFill (row col : Nat)  -- Fill upper-right block
  deriving Repr, BEq, Hashable, Inhabited

/-- Ternary operations (FMA patterns) -/
inductive TernaryOp where
  | FMA           -- dst = a * b + c
  | FMAAxBtC      -- dst = A × B + C (matrix-style, like attention)
  | FMAAxCtB      -- dst = A × C + B (alternate pattern)
  deriving Repr, BEq, Hashable, Inhabited

/-- Convert TernaryOp to C++ -/
def TernaryOp.toCpp : TernaryOp → String
  | .FMA => "fma"
  | .FMAAxBtC => "fma_AxBtC"
  | .FMAAxCtB => "fma_AxCtB"

/-- Semaphore operations -/
inductive SemaphoreOp where
  | Init (count : Nat)    -- Initialize semaphore with count
  | Invalidate            -- Invalidate semaphore
  | Expect (bytes : Nat)  -- Expect bytes on semaphore
  | Wait                  -- Wait on semaphore
  | Arrive (count : Nat)  -- Arrive with transaction count
  | ArriveAndWait         -- Arrive and wait
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
  | storeAsync (dst src : String)     -- store_async for TMA
  | tmaExpect (barrier : String) (bytes : Nat)  -- expect_bytes for TMA

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

  -- Scalar operations
  | scalarMul (dst src : String) (scalar : Float)  -- dst = src * scalar

  -- Broadcasting (row/col vector to tile)
  | broadcast (axis : BroadcastAxis) (dst vec : String)
  | binaryBroadcast (op : BinaryOp) (axis : BroadcastAxis) (dst tile vec : String)

  -- Reductions
  | reduce (op : ReduceOp) (axis : ReduceAxis) (dst src : String)
  | reduceAccum (op : ReduceOp) (axis : ReduceAxis) (dst src accum : String)

  -- Scan/prefix operations (for state-space models)
  | cumsum (axis : ReduceAxis) (dst src : String)  -- Inclusive prefix sum

  -- Outer product (for state matrix updates)
  | outer (dst a b : String)  -- dst[i,j] = a[i] * b[j]

  -- Layout/type conversions
  | swapLayout (dst src : String)
  | transpose (dst src : String)
  | convert (dst src : String)        -- Type conversion (e.g., bf16 → float)

  -- Masking
  | mask (op : MaskOp) (dst src : String) (fillVal : Option Float)

  -- Tile slicing/splitting
  | sliceRows (dst src : String) (startRow numRows : Nat)  -- Extract row range
  | sliceCols (dst src : String) (startCol numCols : Nat)  -- Extract col range

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
