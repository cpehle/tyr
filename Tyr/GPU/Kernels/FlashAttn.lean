/-
  Tyr/GPU/Kernels/FlashAttn.lean

  FlashAttention kernel definition using ThunderKittens abstractions.
  Demonstrates type-safe kernel construction and C++ code generation.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Emit

namespace Tyr.GPU.Kernels

open Codegen

/-- FlashAttention forward kernel (simplified single-block version) -/
def flashAttnFwd : KernelDef := {
  name := "flash_attn_fwd"
  arch := .SM90
  params := [
    { name := "Q_ptr", cppType := "bf16", isPointer := true },
    { name := "K_ptr", cppType := "bf16", isPointer := true },
    { name := "V_ptr", cppType := "bf16", isPointer := true },
    { name := "O_ptr", cppType := "bf16", isPointer := true },
    { name := "seq_len", cppType := "int", isPointer := false },
    { name := "head_dim", cppType := "int", isPointer := false }
  ]
  sharedMemBytes := 64 * 64 * 2 * 3  -- Q, K, V shared tiles
  body := KExpr.seqAll [
    .comment "Declare register tiles",
    .declRT "Q" .BFloat16 64 64 .Row,
    .declRT "K" .BFloat16 64 64 .Col,    -- K must be col-major for mma_AB
    .declRT "V" .BFloat16 64 64 .Row,
    .declRT "S" .BFloat16 64 64 .Row,    -- QK^T scores
    .declRT "P" .BFloat16 64 64 .Row,    -- Softmax output
    .declRT "O" .BFloat16 64 64 .Row,    -- Output accumulator
    .declRV "row_max" .Float32 64,       -- Row-wise max for softmax
    .declRV "row_sum" .Float32 64,       -- Row-wise sum for softmax

    .comment "Declare shared tiles for loading from global",
    .declST "Qs" .BFloat16 64 64 .Row,
    .declST "Ks" .BFloat16 64 64 .Row,
    .declST "Vs" .BFloat16 64 64 .Row,

    .comment "Initialize accumulators",
    .unary .NegInfty "row_max" "row_max",
    .unary .Zero "row_sum" "row_sum",
    .unary .Zero "O" "O",

    .comment "Load Q from shared to register",
    .load "Q" "Qs",

    .comment "Main loop over K, V blocks",
    .forLoop "kv_idx" 0 4 (KExpr.seqAll [
      .comment "Load K, V from shared",
      .load "K" "Ks",
      .load "V" "Vs",

      .comment "S = Q × K^T",
      .mma .AB "S" "Q" "K" "S",

      .comment "Apply causal mask",
      .mask .MakeCausal "S" "S" (some (-1e10)),

      .comment "Online softmax: update row_max",
      .reduceAccum .Max .Row "row_max" "S" "row_max",

      .comment "Subtract max and exponentiate",
      .binaryBroadcast .Sub .Col "S" "S" "row_max",
      .unary .Exp "P" "S",

      .comment "Update row_sum",
      .reduceAccum .Sum .Row "row_sum" "P" "row_sum",

      .comment "Accumulate O = O + P × V",
      .mma .AB "O" "P" "V" "O",

      .comment "Synchronize before next iteration",
      .sync 0
    ]),

    .comment "Final normalization: O = O / row_sum",
    .binaryBroadcast .Div .Col "O" "O" "row_sum"
  ]
}

/-- Simple GEMM kernel for testing -/
def simpleGemm : KernelDef := {
  name := "simple_gemm"
  arch := .SM90
  params := [
    { name := "A_ptr", cppType := "bf16", isPointer := true },
    { name := "B_ptr", cppType := "bf16", isPointer := true },
    { name := "C_ptr", cppType := "float", isPointer := true },
    { name := "M", cppType := "int" },
    { name := "N", cppType := "int" },
    { name := "K", cppType := "int" }
  ]
  body := KExpr.seqAll [
    .comment "Declare tiles",
    .declRT "A" .BFloat16 64 64 .Row,
    .declRT "B" .BFloat16 64 64 .Col,
    .declRT "C" .Float32 64 64 .Row,
    .declST "As" .BFloat16 64 64 .Row,
    .declST "Bs" .BFloat16 64 64 .Row,

    .comment "Initialize accumulator",
    .unary .Zero "C" "C",

    .comment "Main GEMM loop",
    .forLoop "k" 0 8 (KExpr.seqAll [
      .load "A" "As",
      .load "B" "Bs",
      .mma .AB "C" "A" "B" "C",
      .sync 0
    ])
  ]
}

-- Generate C++ code for FlashAttention
#eval IO.println (generateCpp flashAttnFwd)

-- Generate C++ code for simple GEMM
#eval IO.println (generateCpp simpleGemm)

end Tyr.GPU.Kernels
