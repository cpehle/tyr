/-
  Tyr/GPU/Kernels/GELU.lean

  GELU activation kernel translated from ThunderKittens.
  Based on: ThunderKittens/kernels/flux/flux_gelu.cu

  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

  This kernel fuses:
  - Bias addition
  - GELU activation
  - Optional gating (for SwiGLU/GeGLU variants)

  Operations used:
  - load/store
  - add, mul element-wise
  - tanh for GELU approximation
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Emit

namespace Tyr.GPU.Kernels

open Codegen

/-- GELU activation with bias fusion

Computes: GELU(x + bias)

The fast GELU approximation uses tanh:
  GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
-/
def geluFwd : KernelDef := {
  name := "gelu_fwd"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "bias_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "size", cppType := "int" }
  ]
  sharedMemBytes := 64 * 64 * 2 * 2
  body := KExpr.seqAll [
    .comment "Declare register tiles",
    .declRT "x" .BFloat16 64 64 .Row,
    .declRT "bias" .BFloat16 64 64 .Row,
    .declRT "temp" .Float32 64 64 .Row,

    .declST "x_s" .BFloat16 64 64 .Row,
    .declST "bias_s" .BFloat16 64 64 .Row,

    .comment "Load bias tile (long-resident if reused)",
    .load "bias" "bias_s",

    .forLoop "block_idx" 0 16 (KExpr.seqAll [
      .comment "Load input tile",
      .load "x" "x_s",

      .comment "Add bias",
      .binary .Add "x" "x" "bias",

      .comment "Apply GELU activation (built-in ThunderKittens op)",
      .unary .Gelu "x" "x",

      .comment "Store output",
      .store "x_s" "x",

      .sync 0
    ])
  ]
}

/-- SwiGLU activation: SwiGLU(x, gate) = x * sigmoid(gate)

Used in modern LLMs like LLaMA, Mistral.
Often the FFN computes: SwiGLU(W1(x), W_gate(x)) * W2
-/
def swiGluFwd : KernelDef := {
  name := "swiglu_fwd"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "gate_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "size", cppType := "int" }
  ]
  sharedMemBytes := 64 * 64 * 2 * 2
  body := KExpr.seqAll [
    .comment "Declare register tiles",
    .declRT "x" .BFloat16 64 64 .Row,
    .declRT "gate" .BFloat16 64 64 .Row,

    .declST "x_s" .BFloat16 64 64 .Row,
    .declST "gate_s" .BFloat16 64 64 .Row,

    .forLoop "block_idx" 0 16 (KExpr.seqAll [
      .comment "Load input and gate tiles",
      .load "x" "x_s",
      .load "gate" "gate_s",

      .comment "Apply sigmoid to gate: sigmoid(gate)",
      .unary .Sigmoid "gate" "gate",

      .comment "Multiply: x * sigmoid(gate)",
      .binary .Mul "x" "x" "gate",

      .comment "Store output",
      .store "x_s" "x",

      .sync 0
    ])
  ]
}

/-- GeGLU activation: GeGLU(x, gate) = x * GELU(gate)

Similar to SwiGLU but uses GELU instead of sigmoid.
-/
def geGluFwd : KernelDef := {
  name := "geglu_fwd"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "gate_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "size", cppType := "int" }
  ]
  sharedMemBytes := 64 * 64 * 2 * 2
  body := KExpr.seqAll [
    .comment "Declare register tiles",
    .declRT "x" .BFloat16 64 64 .Row,
    .declRT "gate" .BFloat16 64 64 .Row,

    .declST "x_s" .BFloat16 64 64 .Row,
    .declST "gate_s" .BFloat16 64 64 .Row,

    .forLoop "block_idx" 0 16 (KExpr.seqAll [
      .comment "Load input and gate tiles",
      .load "x" "x_s",
      .load "gate" "gate_s",

      .comment "Apply GELU to gate",
      .unary .Gelu "gate" "gate",

      .comment "Multiply: x * GELU(gate)",
      .binary .Mul "x" "x" "gate",

      .comment "Store output",
      .store "x_s" "x",

      .sync 0
    ])
  ]
}

/-- Fused FFN block: out = W2(SwiGLU(W1(x), W_gate(x)))

This is the common pattern in modern transformer FFN layers.
Fuses multiple operations for efficiency.
-/
def fusedFfnSwiGlu : KernelDef := {
  name := "fused_ffn_swiglu"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "w1_ptr", cppType := "bf16", isPointer := true },
    { name := "w_gate_ptr", cppType := "bf16", isPointer := true },
    { name := "w2_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "hidden_dim", cppType := "int" },
    { name := "ffn_dim", cppType := "int" }
  ]
  sharedMemBytes := 64 * 64 * 2 * 4
  body := KExpr.seqAll [
    .comment "Declare tiles for input, intermediate, and output",
    .declRT "x" .BFloat16 64 64 .Row,
    .declRT "w1_out" .BFloat16 64 64 .Row,
    .declRT "gate_out" .BFloat16 64 64 .Row,
    .declRT "hidden" .BFloat16 64 64 .Row,
    .declRT "out" .BFloat16 64 64 .Row,

    .comment "Accumulators for matmuls",
    .declRT "acc1" .Float32 64 64 .Row,
    .declRT "acc_gate" .Float32 64 64 .Row,
    .declRT "acc_out" .Float32 64 64 .Row,

    .comment "Weight tiles",
    .declRT "w1" .BFloat16 64 64 .Col,
    .declRT "w_gate" .BFloat16 64 64 .Col,
    .declRT "w2" .BFloat16 64 64 .Col,

    .declST "x_s" .BFloat16 64 64 .Row,
    .declST "w1_s" .BFloat16 64 64 .Row,
    .declST "w_gate_s" .BFloat16 64 64 .Row,
    .declST "w2_s" .BFloat16 64 64 .Row,
    .declST "out_s" .BFloat16 64 64 .Row,

    .comment "Initialize accumulators",
    .unary .Zero "acc1" "acc1",
    .unary .Zero "acc_gate" "acc_gate",
    .unary .Zero "acc_out" "acc_out",

    .comment "Step 1: Compute W1(x) and W_gate(x) in parallel via MMA",
    .forLoop "k_idx" 0 8 (KExpr.seqAll [
      .load "x" "x_s",
      .load "w1" "w1_s",
      .load "w_gate" "w_gate_s",

      .mma .AB "acc1" "x" "w1" "acc1",
      .mma .AB "acc_gate" "x" "w_gate" "acc_gate",

      .sync 0
    ]),

    .comment "Step 2: Apply SwiGLU activation",
    .convert "w1_out" "acc1",
    .convert "gate_out" "acc_gate",
    .unary .Sigmoid "gate_out" "gate_out",
    .binary .Mul "hidden" "w1_out" "gate_out",

    .comment "Step 3: Compute W2(hidden)",
    .forLoop "k_idx" 0 8 (KExpr.seqAll [
      .load "w2" "w2_s",

      .mma .AB "acc_out" "hidden" "w2" "acc_out",

      .sync 0
    ]),

    .comment "Store final output",
    .convert "out" "acc_out",
    .store "out_s" "out"
  ]
}

-- Generate C++ code for GELU
#eval IO.println (generateCpp geluFwd)

-- Generate C++ code for SwiGLU
#eval IO.println (generateCpp swiGluFwd)

-- Generate C++ code for GeGLU
#eval IO.println (generateCpp geGluFwd)

-- Generate C++ code for fused FFN
#eval IO.println (generateCpp fusedFfnSwiGlu)

end Tyr.GPU.Kernels
