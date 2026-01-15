/-
  Tyr/GPU/Codegen/AST.lean

  Kernel operation types matching ThunderKittens operations.
  These enums are used by the VarId-based IR (IR.lean).
-/
import Tyr.GPU.Types

namespace Tyr.GPU.Codegen

/-- MMA transpose modes -/
inductive MMATranspose where
  | AB    -- A row, B col (standard)
  | ABt   -- A row, B row (B transposed)
  | AtB   -- A col, B col (A transposed)
  | AtBt  -- A col, B row (both transposed)
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

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
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

/-- Convert ReduceOp to C++ -/
def ReduceOp.toCpp : ReduceOp → String
  | .Max => "max"
  | .Min => "min"
  | .Sum => "sum"
  | .Prod => "prod"

/-- Reduction axis -/
inductive ReduceAxis where
  | Row | Col | Full
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

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
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

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
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

instance : ToString BinaryOp where
  toString
    | .Add => "Add"
    | .Sub => "Sub"
    | .Mul => "Mul"
    | .Div => "Div"
    | .Max => "Max"
    | .Min => "Min"

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
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

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
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

/-- Ternary operations (FMA patterns) -/
inductive TernaryOp where
  | FMA           -- dst = a * b + c
  | FMAAxBtC      -- dst = A × B + C (matrix-style, like attention)
  | FMAAxCtB      -- dst = A × C + B (alternate pattern)
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

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
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

end Tyr.GPU.Codegen