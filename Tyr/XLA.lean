-- XLA (Accelerated Linear Algebra) interface for Lean
-- Provides JIT compilation capabilities for tensor operations

import Tyr.Basic

/-!
# `Tyr.XLA`

XLA interop hooks and experimental integration points for compilation-oriented execution backends.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace xla

-- XLA opaque types following LLVM bindings pattern exactly
structure XlaBuilder where
  private mk :: ptr : USize
instance : Nonempty XlaBuilder := by exact ⟨{ ptr := default }⟩

structure XlaOp where
  private mk :: ptr : USize
instance : Nonempty XlaOp := by exact ⟨{ ptr := default }⟩

structure XlaShape where
  private mk :: ptr : USize
instance : Nonempty XlaShape := by exact ⟨{ ptr := default }⟩

structure XlaComputation where
  private mk :: ptr : USize
instance : Nonempty XlaComputation := by exact ⟨{ ptr := default }⟩

structure XlaClient where
  private mk :: ptr : USize
instance : Nonempty XlaClient := by exact ⟨{ ptr := default }⟩

structure XlaExecutable where
  private mk :: ptr : USize
instance : Nonempty XlaExecutable := by exact ⟨{ ptr := default }⟩

structure XlaBuffer where
  private mk :: ptr : USize
instance : Nonempty XlaBuffer := by exact ⟨{ ptr := default }⟩

-- XLA Builder Management
@[extern "lean_xla_new_builder"]
opaque newBuilder (name : String) : IO XlaBuilder

@[extern "lean_xla_build_computation"]
opaque buildComputation (builder : @& XlaBuilder) (root : @& XlaOp) : IO XlaComputation

-- Shape Creation
@[extern "lean_xla_make_shape_f32"]
opaque makeShapeF32 (dimensions : Array UInt64) : IO XlaShape

-- Parameter Creation
@[extern "lean_xla_parameter"]
opaque parameter (builder : @& XlaBuilder) (paramNum : UInt64) (shape : @& XlaShape) (name : String) : IO XlaOp

-- Constants
@[extern "lean_xla_constant_f32"]
opaque constantF32 (builder : @& XlaBuilder) (value : Float) : IO XlaOp

@[extern "lean_xla_constant_array_f32"]
opaque constantArrayF32 (builder : @& XlaBuilder) (values : Array Float) (shape : @& XlaShape) : IO XlaOp

-- Basic Arithmetic Operations
@[extern "lean_xla_add"]
opaque add (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_mul"]
opaque mul (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_sub"]
opaque sub (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

-- Matrix Operations
@[extern "lean_xla_dot"]
opaque dot (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

-- Client Management (following LLVM patterns)
@[extern "lean_xla_get_host_client"]
opaque getHostClient : IO XlaClient

@[extern "lean_xla_get_gpu_client"]
opaque getGpuClient (memoryFraction : Float) (preallocate : Bool) (kind : UInt8) : IO XlaClient

-- Compilation
@[extern "lean_xla_compile"]
opaque compile (client : @& XlaClient) (computation : @& XlaComputation) : IO XlaExecutable

-- Execution
@[extern "lean_xla_execute_direct_f32"]
opaque executeDirectF32 (executable : @& XlaExecutable) : IO Float

@[extern "lean_xla_execute_buffer"]
opaque executeBuffer (executable : @& XlaExecutable) : IO XlaBuffer

-- Buffer operations
@[extern "lean_xla_buffer_to_f32"]
opaque bufferToF32 (buffer : @& XlaBuffer) : IO Float

@[extern "lean_xla_buffer_to_array_f32"]
opaque bufferToArrayF32 (buffer : @& XlaBuffer) : IO (Array Float)

@[extern "lean_xla_buffer_shape"]
opaque bufferShape (buffer : @& XlaBuffer) : IO (Array UInt64)

-- Additional XLA Operations (modeled after EXLA)

-- Tensor creation functions
@[extern "lean_xla_zeros"]
opaque zeros (builder : @& XlaBuilder) (shape : @& XlaShape) : IO XlaOp

@[extern "lean_xla_ones"]
opaque ones (builder : @& XlaBuilder) (shape : @& XlaShape) : IO XlaOp

@[extern "lean_xla_iota"]
opaque iota (builder : @& XlaBuilder) (shape : @& XlaShape) (dimension : UInt64) : IO XlaOp

@[extern "lean_xla_constant_r1_f32"]
opaque constantR1F32 (builder : @& XlaBuilder) (values : Array Float) : IO XlaOp

@[extern "lean_xla_full"]
opaque full (builder : @& XlaBuilder) (shape : @& XlaShape) (value : Float) : IO XlaOp

@[extern "lean_xla_eye"]
opaque eye (builder : @& XlaBuilder) (rows : UInt64) (cols : UInt64) : IO XlaOp

-- Element-wise arithmetic operations  
@[extern "lean_xla_div"]
opaque div (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_pow"]
opaque pow (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_max_elementwise"]
opaque maxElementwise (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_min_elementwise"]
opaque minElementwise (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

-- Element-wise mathematical functions
@[extern "lean_xla_exp"]
opaque exp (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_log"]
opaque log (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_sqrt"]
opaque sqrt (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_tanh"]
opaque tanh (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_abs"]
opaque abs (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_neg"]
opaque neg (operand : @& XlaOp) : IO XlaOp

-- Trigonometric operations
@[extern "lean_xla_sin"]
opaque sin (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_cos"]
opaque cos (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_tan"]
opaque tan (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_asin"]
opaque asin (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_acos"]
opaque acos (operand : @& XlaOp) : IO XlaOp

@[extern "lean_xla_atan"]
opaque atan (operand : @& XlaOp) : IO XlaOp

-- Reduction operations (properly implemented for softmax)
@[extern "lean_xla_reduce_sum"]
opaque reduceSum (operand : @& XlaOp) (initValue : @& XlaOp) (dimensions : Array UInt64) : IO XlaOp

@[extern "lean_xla_reduce_max"]
opaque reduceMax (operand : @& XlaOp) (initValue : @& XlaOp) (dimensions : Array UInt64) : IO XlaOp

@[extern "lean_xla_reduce_min"]
opaque reduceMin (operand : @& XlaOp) (initValue : @& XlaOp) (dimensions : Array UInt64) : IO XlaOp

-- Broadcasting operations for softmax
@[extern "lean_xla_broadcast_in_dim"]
opaque broadcastInDim (operand : @& XlaOp) (shape : @& XlaShape) (broadcastDimensions : Array UInt64) : IO XlaOp

-- Concatenation for multi-head attention
@[extern "lean_xla_concatenate"]
opaque concatenate (operands : Array XlaOp) (dimension : UInt64) : IO XlaOp

-- Slicing for multi-head attention head splitting  
@[extern "lean_xla_slice"]
opaque slice (operand : @& XlaOp) (startIndices : Array UInt64) (limitIndices : Array UInt64) (strides : Array UInt64) : IO XlaOp

-- Shape operations
@[extern "lean_xla_reshape"]
opaque reshape (operand : @& XlaOp) (newShape : @& XlaShape) : IO XlaOp

@[extern "lean_xla_transpose"]
opaque transpose (operand : @& XlaOp) (permutation : Array UInt64) : IO XlaOp

-- Comparison operations
@[extern "lean_xla_eq"]
opaque eq (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_ne"]
opaque ne (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_lt"]
opaque lt (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_le"]
opaque le (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_gt"]
opaque gt (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

@[extern "lean_xla_ge"]
opaque ge (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

-- Select operation (conditional)
@[extern "lean_xla_select"]
opaque select (pred : @& XlaOp) (onTrue : @& XlaOp) (onFalse : @& XlaOp) : IO XlaOp

-- Note: Softmax will be implemented in Lean using the primitive operations above

-- Batch matrix multiplication
@[extern "lean_xla_batch_dot"]
opaque batchDot (lhs : @& XlaOp) (rhs : @& XlaOp) : IO XlaOp

end xla
