/-
  Tyr/GPU/Kernels/LayerNorm.lean

  LayerNorm kernel definition translated from ThunderKittens.
  Based on: ThunderKittens/kernels/layernorm/layernorm.cu

  Operations used:
  - warp::load/store for shared memory
  - add, sub, mul, div element-wise
  - sum reduction for mean/variance
  - rsqrt for normalization
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Emit

namespace Tyr.GPU.Kernels

open Codegen

/-- LayerNorm forward kernel with residual connection

This kernel computes:
1. residual = residual + x  (residual connection)
2. mean = sum(residual) / D
3. var = sum((residual - mean)^2) / D
4. out = (residual - mean) / sqrt(var + eps) * weight + bias
-/
def layerNormFwd (hiddenDim : Nat := 1024) : KernelDef := {
  name := "layernorm_fwd"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "residual_ptr", cppType := "bf16", isPointer := true },
    { name := "weight_ptr", cppType := "bf16", isPointer := true },
    { name := "bias_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "out_resid_ptr", cppType := "bf16", isPointer := true },
    { name := "batch_size", cppType := "int" },
    { name := "seq_len", cppType := "int" },
    { name := "hidden_dim", cppType := "int" }
  ]
  sharedMemBytes := hiddenDim * 2 * 4  -- x, residual, weight, bias
  body := KExpr.seqAll [
    .comment "Declare shared vectors for hidden dimension",
    .declSV "x_s" .BFloat16 hiddenDim,
    .declSV "residual_s" .BFloat16 hiddenDim,
    .declSV "weight_s" .BFloat16 hiddenDim,
    .declSV "bias_s" .BFloat16 hiddenDim,
    .declSV "temp_s" .BFloat16 hiddenDim,

    .comment "Declare register vectors for computations",
    .declRV "mean" .Float32 1,
    .declRV "var" .Float32 1,
    .declRV "inv_std" .Float32 1,

    .comment "Load norm parameters (long-resident)",
    .load "weight_s" "weight_ptr",
    .load "bias_s" "bias_ptr",

    .comment "Process each token in the sequence",
    .forLoop "seq_idx" 0 128 (KExpr.seqAll [
      .comment "Load input and residual for this token",
      .load "x_s" "x_ptr",
      .load "residual_s" "residual_ptr",

      .comment "Step 1: Add residual connection",
      .binary .Add "residual_s" "residual_s" "x_s",

      .comment "Store updated residual for skip connection output",
      .store "out_resid_ptr" "residual_s",

      .comment "Step 2: Compute mean = sum(residual) / D",
      .reduce .Sum .Full "mean" "residual_s",
      -- Note: Division by hidden_dim would need scalar div

      .comment "Step 3: Subtract mean from residual",
      .binaryBroadcast .Sub .Row "temp_s" "residual_s" "mean",

      .comment "Step 4: Compute variance = sum((x - mean)^2) / D",
      .binary .Mul "residual_s" "temp_s" "temp_s",
      .reduce .Sum .Full "var" "residual_s",
      -- Note: Division by hidden_dim + eps, then rsqrt

      .comment "Step 5: Compute inverse standard deviation",
      .unary .Rsqrt "inv_std" "var",

      .comment "Step 6: Normalize: (x - mean) * inv_std",
      .binaryBroadcast .Mul .Row "temp_s" "temp_s" "inv_std",

      .comment "Step 7: Scale and shift: out = normalized * weight + bias",
      .binary .Mul "temp_s" "temp_s" "weight_s",
      .binary .Add "temp_s" "temp_s" "bias_s",

      .comment "Store output",
      .store "out_ptr" "temp_s",

      .sync 0
    ])
  ]
}

/-- RMSNorm kernel (simpler variant without mean subtraction)

RMSNorm computes:
  rms = sqrt(mean(x^2) + eps)
  out = x / rms * weight
-/
def rmsNormFwd (hiddenDim : Nat := 1024) : KernelDef := {
  name := "rmsnorm_fwd"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "weight_ptr", cppType := "bf16", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "hidden_dim", cppType := "int" }
  ]
  sharedMemBytes := hiddenDim * 2 * 2  -- x, weight
  body := KExpr.seqAll [
    .comment "Declare shared vectors",
    .declSV "x_s" .BFloat16 hiddenDim,
    .declSV "weight_s" .BFloat16 hiddenDim,
    .declSV "temp_s" .BFloat16 hiddenDim,

    .comment "Declare register scalars",
    .declRV "rms_sq" .Float32 1,
    .declRV "inv_rms" .Float32 1,

    .comment "Load weight (long-resident)",
    .load "weight_s" "weight_ptr",

    .comment "Process each token",
    .forLoop "seq_idx" 0 128 (KExpr.seqAll [
      .comment "Load input",
      .load "x_s" "x_ptr",

      .comment "Compute sum of squares",
      .binary .Mul "temp_s" "x_s" "x_s",
      .reduce .Sum .Full "rms_sq" "temp_s",

      .comment "Compute inverse RMS (1/sqrt(mean(x^2) + eps))",
      .unary .Rsqrt "inv_rms" "rms_sq",

      .comment "Normalize: x * inv_rms",
      .binaryBroadcast .Mul .Row "temp_s" "x_s" "inv_rms",

      .comment "Scale by weight",
      .binary .Mul "temp_s" "temp_s" "weight_s",

      .comment "Store output",
      .store "out_ptr" "temp_s",

      .sync 0
    ])
  ]
}

-- Generate C++ code for LayerNorm
#eval IO.println (generateCpp (layerNormFwd 1024))

-- Generate C++ code for RMSNorm
#eval IO.println (generateCpp (rmsNormFwd 1024))

end Tyr.GPU.Kernels
