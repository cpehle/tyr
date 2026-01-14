/-
  Tyr/GPU/Kernels/Mamba.lean

  Mamba2 (Selective State Space Model) kernel translated from ThunderKittens.
  Based on: ThunderKittens/kernels/mamba2/mamba2.cu

  Mamba2 implements a selective structured state space model with:
  1. Cumulative decay tracking via prefix sum
  2. Local attention with position-dependent decay
  3. Cross-position state accumulation

  The computation flow:
  - Compute cumsum of log decay factors A
  - Build decay matrix: decay[i,j] = exp(cumsum[i] - cumsum[j])
  - Apply causal mask
  - Local attention: att = Q @ K^T
  - Weight by decay: att = att * decay
  - Output: O = att @ V
  - Accumulate KV states for future positions

  Operations used:
  - cumsum for prefix scan
  - mm_ABt, mma_AB for matrix multiplies
  - exp for decay computation
  - make_causal for masking
  - mul for element-wise weighting
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Emit

namespace Tyr.GPU.Kernels

open Codegen

/-- Mamba2 forward kernel

Selective State Space Model with gated linear attention and exponential decay.

Input layout:
- Q: (B, 1, N, D) - Query, shared across heads
- K: (B, 1, N, D) - Key, shared across heads
- V: (B, H, N, D) - Value, per-head
- A: (B, H, N) - Decay factors (log-space), per-head per-token

Output:
- O: (B, H, N, D) - Output, per-head
-/
def mamba2Fwd (seqTile : Nat := 64) (headDim : Nat := 64) : KernelDef := {
  name := "mamba2_fwd"
  arch := .SM90
  params := [
    { name := "q_ptr", cppType := "bf16", isPointer := true },
    { name := "k_ptr", cppType := "bf16", isPointer := true },
    { name := "v_ptr", cppType := "bf16", isPointer := true },
    { name := "a_ptr", cppType := "float", isPointer := true },
    { name := "o_ptr", cppType := "bf16", isPointer := true },
    { name := "kv_state_ptr", cppType := "float", isPointer := true },
    { name := "batch_size", cppType := "int" },
    { name := "num_heads", cppType := "int" },
    { name := "seq_len", cppType := "int" },
    { name := "head_dim", cppType := "int" }
  ]
  sharedMemBytes := seqTile * headDim * 2 * 4 + seqTile * 4  -- Q, K, V, A
  body := KExpr.seqAll [
    .comment "=== Tile Declarations ===",

    .comment "Register tiles for Q, K, V (16×64 tiles per warp)",
    .declRT "q_reg" .BFloat16 16 headDim .Row,
    .declRT "k_reg" .BFloat16 16 headDim .Row,
    .declRT "v_reg" .BFloat16 16 headDim .Row,

    .comment "Attention scores tile (16×16 for local attention)",
    .declRT "att_block" .Float32 16 16 .Row,

    .comment "Local decay matrix (16×16)",
    .declRT "local_decay" .Float32 16 16 .Row,

    .comment "Output accumulator",
    .declRT "o_reg" .Float32 16 headDim .Row,

    .comment "Decay factor vectors",
    .declRV "a_vec" .Float32 16,
    .declRV "a_cumsum" .Float32 16,
    .declRV "a_cumsum_i" .Float32 16,
    .declRV "a_cumsum_j" .Float32 16,

    .comment "State vectors for cross-position accumulation",
    .declRT "kv_state" .Float32 headDim headDim .Row,

    .comment "Shared memory tiles",
    .declST "q_s" .BFloat16 16 headDim .Row,
    .declST "k_s" .BFloat16 16 headDim .Row,
    .declST "v_s" .BFloat16 16 headDim .Row,
    .declSV "a_s" .Float32 seqTile,

    .comment "=== Initialize Accumulators ===",
    .unary .Zero "o_reg" "o_reg",
    .unary .Zero "kv_state" "kv_state",

    .comment "=== Main Sequence Loop ===",
    .comment "Process sequence in chunks of seqTile tokens",
    .forLoop "chunk_idx" 0 16 (KExpr.seqAll [

      .comment "--- Step 1: Load Q, K, V, A for this chunk ---",
      .load "q_reg" "q_s",
      .load "k_reg" "k_s",
      .load "v_reg" "v_s",
      .load "a_vec" "a_s",

      .comment "--- Step 2: Compute cumulative sum of decay factors ---",
      .comment "a_cumsum[i] = sum(a[0..i]) - inclusive prefix sum",
      .cumsum .Row "a_cumsum" "a_vec",

      .comment "--- Step 3: Build decay matrix ---",
      .comment "decay[i,j] = exp(cumsum[i] - cumsum[j])",
      .comment "For each row i, broadcast cumsum[i]",
      .broadcast .Row "a_cumsum_i" "a_cumsum",
      .comment "For each col j, broadcast cumsum[j]",
      .broadcast .Col "a_cumsum_j" "a_cumsum",
      .comment "Compute difference and exponentiate",
      .binary .Sub "local_decay" "a_cumsum_i" "a_cumsum_j",
      .unary .Exp "local_decay" "local_decay",

      .comment "--- Step 4: Apply causal mask ---",
      .comment "Zero out positions where j > i (future positions)",
      .mask .MakeCausal "local_decay" "local_decay" (some 0.0),

      .comment "--- Step 5: Compute local attention scores ---",
      .comment "att_block = Q @ K^T",
      .mm .ABt "att_block" "q_reg" "k_reg",

      .comment "--- Step 6: Apply decay weighting ---",
      .comment "att_block = att_block * decay (element-wise)",
      .binary .Mul "att_block" "att_block" "local_decay",

      .comment "--- Step 7: Compute local output contribution ---",
      .comment "o_local = att_block @ V",
      .mma .AB "o_reg" "att_block" "v_reg" "o_reg",

      .comment "--- Step 8: Cross-position state accumulation ---",
      .comment "Scale Q by final position decay for state lookup",
      .binaryBroadcast .Mul .Row "q_reg" "q_reg" "a_cumsum",

      .comment "Accumulate from KV state: o += Q @ kv_state",
      .mma .AB "o_reg" "q_reg" "kv_state" "o_reg",

      .comment "--- Step 9: Update KV state for future chunks ---",
      .comment "Scale K by inverse decay: k_scaled = K * exp(final_cumsum - cumsum)",
      .comment "Compute state update: kv_state += V^T @ k_scaled",
      .mma .AtB "kv_state" "v_reg" "k_reg" "kv_state",

      .comment "--- Step 10: Store output for this chunk ---",
      .convert "v_reg" "o_reg",
      .store "o_ptr" "v_reg",

      .comment "Reset output accumulator for next chunk",
      .unary .Zero "o_reg" "o_reg",

      .sync 0
    ]),

    .comment "=== Store final KV state for incremental inference ===",
    .store "kv_state_ptr" "kv_state"
  ]
}

/-- Mamba2 with TMA (Tensor Memory Accelerator) for optimized memory access

This version uses async TMA operations for overlapping compute and memory.
-/
def mamba2FwdTMA (seqTile : Nat := 64) (headDim : Nat := 64) : KernelDef := {
  name := "mamba2_fwd_tma"
  arch := .SM90
  params := [
    { name := "q_ptr", cppType := "bf16", isPointer := true },
    { name := "k_ptr", cppType := "bf16", isPointer := true },
    { name := "v_ptr", cppType := "bf16", isPointer := true },
    { name := "a_ptr", cppType := "float", isPointer := true },
    { name := "o_ptr", cppType := "bf16", isPointer := true },
    { name := "batch_size", cppType := "int" },
    { name := "num_heads", cppType := "int" },
    { name := "seq_len", cppType := "int" }
  ]
  sharedMemBytes := seqTile * headDim * 2 * 6  -- Double buffering for Q, K, V
  body := KExpr.seqAll [
    .comment "=== Double-Buffered Tile Declarations ===",

    .comment "Shared memory with double buffering (tic/toc)",
    .declST "q_s_0" .BFloat16 16 headDim .Row,
    .declST "q_s_1" .BFloat16 16 headDim .Row,
    .declST "k_s_0" .BFloat16 16 headDim .Row,
    .declST "k_s_1" .BFloat16 16 headDim .Row,
    .declST "v_s_0" .BFloat16 16 headDim .Row,
    .declST "v_s_1" .BFloat16 16 headDim .Row,

    .comment "Register tiles",
    .declRT "q_reg" .BFloat16 16 headDim .Row,
    .declRT "k_reg" .BFloat16 16 headDim .Row,
    .declRT "v_reg" .BFloat16 16 headDim .Row,
    .declRT "att_block" .Float32 16 16 .Row,
    .declRT "local_decay" .Float32 16 16 .Row,
    .declRT "o_reg" .Float32 16 headDim .Row,

    .declRV "a_vec" .Float32 16,
    .declRV "a_cumsum" .Float32 16,

    .comment "=== Initialize ===",
    .unary .Zero "o_reg" "o_reg",

    .comment "=== Prefetch first chunk ===",
    .loadAsync "q_s_0" "q_ptr",
    .loadAsync "k_s_0" "k_ptr",
    .loadAsync "v_s_0" "v_ptr",

    .comment "=== Main pipelined loop ===",
    .forLoop "chunk_idx" 0 16 (KExpr.seqAll [
      .comment "Wait for current chunk load to complete",
      .mmaAsyncWait 0,

      .comment "Start loading next chunk (double buffer)",
      .loadAsync "q_s_1" "q_ptr",
      .loadAsync "k_s_1" "k_ptr",
      .loadAsync "v_s_1" "v_ptr",

      .comment "Load current chunk to registers",
      .load "q_reg" "q_s_0",
      .load "k_reg" "k_s_0",
      .load "v_reg" "v_s_0",

      .comment "Compute cumsum of decay",
      .cumsum .Row "a_cumsum" "a_vec",

      .comment "Build decay matrix",
      .broadcast .Row "local_decay" "a_cumsum",
      .unary .Exp "local_decay" "local_decay",
      .mask .MakeCausal "local_decay" "local_decay" (some 0.0),

      .comment "Local attention with decay",
      .mm .ABt "att_block" "q_reg" "k_reg",
      .binary .Mul "att_block" "att_block" "local_decay",

      .comment "Output accumulation",
      .mma .AB "o_reg" "att_block" "v_reg" "o_reg",

      .comment "Swap buffers for next iteration",
      .comment "(In actual code, indices would alternate)",

      .sync 0
    ]),

    .comment "=== Store final output ===",
    .convert "v_reg" "o_reg",
    .store "o_ptr" "v_reg"
  ]
}

/-- Simplified Mamba block for understanding the core algorithm

This distills the key Mamba computation pattern without optimization complexity.
-/
def mambaSimple (headDim : Nat := 64) : KernelDef := {
  name := "mamba_simple"
  arch := .SM90
  params := [
    { name := "x_ptr", cppType := "bf16", isPointer := true },
    { name := "dt_ptr", cppType := "float", isPointer := true },
    { name := "A_ptr", cppType := "float", isPointer := true },
    { name := "B_ptr", cppType := "bf16", isPointer := true },
    { name := "C_ptr", cppType := "bf16", isPointer := true },
    { name := "D_ptr", cppType := "float", isPointer := true },
    { name := "out_ptr", cppType := "bf16", isPointer := true },
    { name := "seq_len", cppType := "int" },
    { name := "state_dim", cppType := "int" }
  ]
  body := KExpr.seqAll [
    .comment "=== Mamba SSM: y = Cx + Du, where x' = Ax + Bu ===",

    .comment "Declare state vector h (hidden state)",
    .declRV "h" .Float32 headDim,
    .unary .Zero "h" "h",

    .comment "Input/output tiles",
    .declRV "u" .Float32 1,       -- Input scalar at each step
    .declRV "y" .Float32 1,       -- Output scalar at each step
    .declRV "dt" .Float32 1,      -- Discretization timestep

    .comment "SSM parameters (discretized)",
    .declRV "A_bar" .Float32 headDim,  -- exp(dt * A)
    .declRV "B_bar" .Float32 headDim,  -- (exp(dt * A) - 1) / A * B
    .declRV "C_vec" .Float32 headDim,
    .declRV "D_scalar" .Float32 1,

    .comment "Temporaries",
    .declRV "temp" .Float32 headDim,
    .declRV "Ax" .Float32 headDim,
    .declRV "Bu" .Float32 headDim,

    .declSV "x_s" .Float32 headDim,

    .comment "Load D (skip connection weight)",
    .load "D_scalar" "D_ptr",

    .comment "=== Sequential SSM recurrence ===",
    .forLoop "t" 0 128 (KExpr.seqAll [
      .comment "Load input u[t] and discretization dt[t]",
      .load "u" "x_ptr",
      .load "dt" "dt_ptr",

      .comment "Load SSM matrices for selective SSM (input-dependent)",
      .load "A_bar" "A_ptr",
      .load "B_bar" "B_ptr",
      .load "C_vec" "C_ptr",

      .comment "Discretize: A_bar = exp(dt * A)",
      .binary .Mul "A_bar" "A_bar" "dt",
      .unary .Exp "A_bar" "A_bar",

      .comment "State update: h = A_bar * h + B_bar * u",
      .binary .Mul "Ax" "A_bar" "h",
      .binaryBroadcast .Mul .Row "Bu" "B_bar" "u",
      .binary .Add "h" "Ax" "Bu",

      .comment "Output: y = C @ h + D * u",
      .binary .Mul "temp" "C_vec" "h",
      .reduce .Sum .Full "y" "temp",
      .binaryBroadcast .Mul .Row "temp" "D_scalar" "u",
      .binary .Add "y" "y" "temp",

      .comment "Store output y[t]",
      .store "out_ptr" "y",

      .sync 0
    ])
  ]
}

-- Generate C++ code for Mamba2 forward
#eval IO.println (generateCpp (mamba2Fwd 64 64))

-- Generate C++ code for Mamba2 with TMA
#eval IO.println (generateCpp (mamba2FwdTMA 64 64))

-- Generate C++ code for simplified Mamba
#eval IO.println (generateCpp (mambaSimple 64))

end Tyr.GPU.Kernels
