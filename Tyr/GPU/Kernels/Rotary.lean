/-
  Tyr/GPU/Kernels/Rotary.lean

  Rotary Positional Embedding (RoPE) kernel translated from ThunderKittens.
  Based on: ThunderKittens/kernels/rotary/rotary.cu

  RoPE applies rotation to pairs of elements in the hidden dimension:
    x1' = x1 * cos(θ) - x2 * sin(θ)
    x2' = x1 * sin(θ) + x2 * cos(θ)

  Where θ = position * base^(-2i/d) for each dimension i.

  Operations used:
  - load/store for shared memory
  - mul, add, sub element-wise
  - slicing for splitting dimensions
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Emit

namespace Tyr.GPU.Kernels

open Codegen

/-- Rotary Positional Embedding forward kernel

Applies rotation to Q and K tensors for attention.
The rotation is applied to pairs of adjacent elements.
-/
def rotaryFwd (headDim : Nat := 64) : KernelDef := {
  name := "rotary_fwd"
  arch := .SM90
  params := [
    { name := "q_ptr", cppType := "bf16", isPointer := true },
    { name := "k_ptr", cppType := "bf16", isPointer := true },
    { name := "q_out_ptr", cppType := "bf16", isPointer := true },
    { name := "k_out_ptr", cppType := "bf16", isPointer := true },
    { name := "sin_ptr", cppType := "float", isPointer := true },
    { name := "cos_ptr", cppType := "float", isPointer := true },
    { name := "seq_len", cppType := "int" },
    { name := "num_heads", cppType := "int" }
  ]
  sharedMemBytes := headDim * 2 * 4  -- q, k tiles + sin/cos
  body := KExpr.seqAll [
    .comment "Declare register tiles for query and key",
    .declRT "q" .BFloat16 16 headDim .Row,
    .declRT "k" .BFloat16 16 headDim .Row,

    .comment "Split into first and second halves for rotation",
    .declRT "q1" .BFloat16 16 (headDim/2) .Row,
    .declRT "q2" .BFloat16 16 (headDim/2) .Row,
    .declRT "k1" .BFloat16 16 (headDim/2) .Row,
    .declRT "k2" .BFloat16 16 (headDim/2) .Row,

    .comment "Temporary tiles for rotated values",
    .declRT "q1_rot" .BFloat16 16 (headDim/2) .Row,
    .declRT "q2_rot" .BFloat16 16 (headDim/2) .Row,
    .declRT "k1_rot" .BFloat16 16 (headDim/2) .Row,
    .declRT "k2_rot" .BFloat16 16 (headDim/2) .Row,

    .comment "Sin and cos tables (position-dependent)",
    .declRT "sin_table" .Float32 16 (headDim/2) .Row,
    .declRT "cos_table" .Float32 16 (headDim/2) .Row,

    .comment "Shared memory tiles",
    .declST "q_s" .BFloat16 16 headDim .Row,
    .declST "k_s" .BFloat16 16 headDim .Row,

    .comment "Load sin/cos tables for this position range",
    .load "sin_table" "sin_ptr",
    .load "cos_table" "cos_ptr",

    .comment "Process tokens in blocks of 16",
    .forLoop "token_idx" 0 64 (KExpr.seqAll [
      .comment "Load Q and K tiles",
      .load "q" "q_s",
      .load "k" "k_s",

      .comment "Split Q into halves: q1 = q[:, :d/2], q2 = q[:, d/2:]",
      .sliceCols "q1" "q" 0 (headDim/2),
      .sliceCols "q2" "q" (headDim/2) (headDim/2),

      .comment "Split K into halves: k1 = k[:, :d/2], k2 = k[:, d/2:]",
      .sliceCols "k1" "k" 0 (headDim/2),
      .sliceCols "k2" "k" (headDim/2) (headDim/2),

      .comment "Apply rotation to Q:",
      .comment "q1' = q1 * cos - q2 * sin",
      .binary .Mul "q1_rot" "q1" "cos_table",
      .binary .Mul "q2_rot" "q2" "sin_table",
      .binary .Sub "q1_rot" "q1_rot" "q2_rot",

      .comment "q2' = q1 * sin + q2 * cos",
      .binary .Mul "q1" "q1" "sin_table",
      .binary .Mul "q2" "q2" "cos_table",
      .binary .Add "q2_rot" "q1" "q2",

      .comment "Apply rotation to K:",
      .comment "k1' = k1 * cos - k2 * sin",
      .binary .Mul "k1_rot" "k1" "cos_table",
      .binary .Mul "k2_rot" "k2" "sin_table",
      .binary .Sub "k1_rot" "k1_rot" "k2_rot",

      .comment "k2' = k1 * sin + k2 * cos",
      .binary .Mul "k1" "k1" "sin_table",
      .binary .Mul "k2" "k2" "cos_table",
      .binary .Add "k2_rot" "k1" "k2",

      .comment "Store rotated Q and K",
      .store "q_out_ptr" "q",
      .store "k_out_ptr" "k",

      .sync 0
    ])
  ]
}

/-- YaRN-style rotary with NTK-aware scaling

YaRN extends RoPE with:
1. NTK-aware interpolation for sequence length extension
2. Linear interpolation in lower dimensions
3. Full NTK scaling in higher dimensions
-/
def yarnRotaryFwd (headDim : Nat := 64) : KernelDef := {
  name := "yarn_rotary_fwd"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "sin_ptr", cppType := "float", isPointer := true },
    { name := "cos_ptr", cppType := "float", isPointer := true },
    { name := "scale_ptr", cppType := "float", isPointer := true },
    { name := "seq_len", cppType := "int" },
    { name := "original_max_len", cppType := "int" }
  ]
  body := KExpr.seqAll [
    .comment "Declare register tiles",
    .declRT "x" .BFloat16 16 headDim .Row,
    .declRT "x1" .BFloat16 16 (headDim/2) .Row,
    .declRT "x2" .BFloat16 16 (headDim/2) .Row,
    .declRT "x1_rot" .BFloat16 16 (headDim/2) .Row,
    .declRT "x2_rot" .BFloat16 16 (headDim/2) .Row,

    .comment "Sin/cos with NTK scaling applied",
    .declRT "sin_scaled" .Float32 16 (headDim/2) .Row,
    .declRT "cos_scaled" .Float32 16 (headDim/2) .Row,

    .comment "Per-dimension attention scaling factor",
    .declRV "scale" .Float32 (headDim/2),

    .declST "x_s" .BFloat16 16 headDim .Row,

    .comment "Load precomputed NTK-scaled sin/cos and attention scales",
    .load "sin_scaled" "sin_ptr",
    .load "cos_scaled" "cos_ptr",
    .load "scale" "scale_ptr",

    .forLoop "token_idx" 0 64 (KExpr.seqAll [
      .load "x" "x_s",

      .comment "Split into halves",
      .sliceCols "x1" "x" 0 (headDim/2),
      .sliceCols "x2" "x" (headDim/2) (headDim/2),

      .comment "Apply scaled rotation",
      .binary .Mul "x1_rot" "x1" "cos_scaled",
      .binary .Mul "x2_rot" "x2" "sin_scaled",
      .binary .Sub "x1_rot" "x1_rot" "x2_rot",

      .binary .Mul "x1" "x1" "sin_scaled",
      .binary .Mul "x2" "x2" "cos_scaled",
      .binary .Add "x2_rot" "x1" "x2",

      .comment "Apply attention scaling",
      .binaryBroadcast .Mul .Col "x1_rot" "x1_rot" "scale",
      .binaryBroadcast .Mul .Col "x2_rot" "x2_rot" "scale",

      .store "out_ptr" "x",
      .sync 0
    ])
  ]
}

-- Generate C++ code for Rotary
#eval IO.println (generateCpp (rotaryFwd 64))

-- Generate C++ code for YaRN Rotary
#eval IO.println (generateCpp (yarnRotaryFwd 64))

end Tyr.GPU.Kernels
