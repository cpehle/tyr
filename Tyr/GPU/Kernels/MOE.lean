/-
  Tyr/GPU/Kernels/MOE.lean

  Mixture of Experts (MOE) dispatch kernel implementation.
  Based on ThunderKittens patterns.

  Key features:
  - Token-to-expert routing based on gating scores
  - Top-K expert selection
  - Efficient token permutation for expert processing
  - Expert load balancing
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels.MOE

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Mixture of Experts

MOE layers route different tokens to different "expert" networks:
1. **Gating**: Compute routing scores for each token-expert pair
2. **Top-K Selection**: Select K experts per token (typically K=1 or K=2)
3. **Dispatch**: Permute tokens to their assigned experts
4. **Expert Compute**: Each expert processes its assigned tokens
5. **Combine**: Gather results and weight by gating scores

Common patterns:
- Token-choice (each token chooses experts)
- Expert-choice (each expert chooses tokens)
- Capacity factor limits tokens per expert
-/

/-- MOE Gating - compute routing probabilities -/
@[gpu_kernel .SM90]
def moeGatingFwd : KernelM Unit := do
  comment "=== MOE Gating ==="
  comment "Compute routing scores: scores = softmax(x @ W_gate)"

  -- Input tokens
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Gating weights (projects to num_experts)
  let wGate : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col  -- hidden_dim x num_experts

  -- Gating scores (64 tokens x 64 experts, assuming 64 experts)
  let scores : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Softmax normalization
  let maxVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Output (routing probabilities)
  let probs : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wGateShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load gating weights (long-resident)"
  load wGate wGateShared

  comment "Process token batches"
  forLoop 0 8 do
    comment "Load tokens"
    load x xShared

    comment "Compute gating scores: scores = x @ W_gate"
    mma scores x wGate scores

    comment "Softmax over experts (row-wise)"
    rowMax maxVec scores
    subCol scores scores maxVec
    exp scores scores
    rowSum sumVec scores
    divCol probs scores sumVec

    comment "Store routing probabilities"
    convert out probs
    store outShared out
    sync

def moeGatingFwdKernel : Kernel :=
  buildKernelM "moe_gating_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "w_gate", dtype := .BFloat16, isPointer := true },
    { name := "routing_probs", dtype := .BFloat16, isPointer := true },
    { name := "num_tokens", dtype := .Float32, isPointer := false },
    { name := "num_experts", dtype := .Float32, isPointer := false },
    { name := "hidden_dim", dtype := .Float32, isPointer := false }
  ] moeGatingFwd

/-- MOE Dispatch - route tokens to experts -/
@[gpu_kernel .SM90]
def moeDispatchFwd : KernelM Unit := do
  comment "=== MOE Dispatch ==="
  comment "Permute tokens based on routing decisions"

  -- Input tokens
  let tokens : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Routing decisions (top-K expert indices per token)
  let expertIds : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64  -- Storing indices as float (simplified)

  -- Routing weights (softmax scores for selected experts)
  let routingWeights : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Output (permuted tokens ready for expert processing)
  let dispatched : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Token indices for gathering results later
  let tokenIndices : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Shared memory
  let tokensShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let expertIdsShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let routingWeightsShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let dispatchedShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load inputs"
  load tokens tokensShared
  load expertIds expertIdsShared
  load routingWeights routingWeightsShared

  comment "Dispatch: permute tokens to expert buffers"
  comment "(Actual implementation would use scatter operations)"
  -- For each token, copy to the buffer of its assigned expert
  -- Simplified: just copy through
  copy dispatched tokens

  comment "Store dispatched tokens"
  store dispatchedShared dispatched
  sync

def moeDispatchFwdKernel : Kernel :=
  buildKernelM "moe_dispatch_fwd" .SM90 #[
    { name := "tokens", dtype := .BFloat16, isPointer := true },
    { name := "expert_ids", dtype := .Float32, isPointer := true },
    { name := "routing_weights", dtype := .Float32, isPointer := true },
    { name := "dispatched", dtype := .BFloat16, isPointer := true },
    { name := "token_indices", dtype := .Float32, isPointer := true },
    { name := "num_tokens", dtype := .Float32, isPointer := false },
    { name := "top_k", dtype := .Float32, isPointer := false }
  ] moeDispatchFwd

/-- MOE Combine - gather expert outputs and weight -/
@[gpu_kernel .SM90]
def moeCombineFwd : KernelM Unit := do
  comment "=== MOE Combine ==="
  comment "Gather expert outputs and combine with routing weights"

  -- Expert outputs (after expert computation)
  let expertOut : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Routing weights
  let routingWeights : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Working tiles
  let expertOutF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let combined : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let expertOutShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let routingWeightsShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load routing weights"
  loadVec routingWeights routingWeightsShared

  comment "Process expert outputs"
  forLoop 0 8 do  -- Loop over experts (assuming top-k = 2 or so)
    comment "Load expert output"
    load expertOut expertOutShared

    comment "Convert to float"
    convert expertOutF expertOut

    comment "Weight by routing score"
    mulCol expertOutF expertOutF routingWeights

    comment "Accumulate"
    add combined combined expertOutF

    sync

  comment "Store combined output"
  convert out combined
  store outShared out

def moeCombineFwdKernel : Kernel :=
  buildKernelM "moe_combine_fwd" .SM90 #[
    { name := "expert_outputs", dtype := .BFloat16, isPointer := true },
    { name := "routing_weights", dtype := .Float32, isPointer := true },
    { name := "combined_output", dtype := .BFloat16, isPointer := true },
    { name := "num_tokens", dtype := .Float32, isPointer := false },
    { name := "num_experts", dtype := .Float32, isPointer := false },
    { name := "top_k", dtype := .Float32, isPointer := false }
  ] moeCombineFwd

/-! ## Expert-Choice MOE

Alternative routing where experts choose tokens instead of vice versa.
Better load balancing but different semantics.
-/

/-- Expert-Choice MOE gating -/
@[gpu_kernel .SM90]
def moeExpertChoiceFwd : KernelM Unit := do
  comment "=== Expert-Choice MOE ==="
  comment "Each expert selects its top-C tokens (capacity C)"

  -- Gating scores (tokens x experts)
  let scores : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Transposed scores for expert-wise selection
  let scoresT : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Expert selections (top-C token indices per expert)
  let selections : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Softmax over tokens (column-wise)
  let maxVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Normalized routing weights
  let routingWeights : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let scoresShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load gating scores"
  load scores scoresShared

  comment "Transpose for expert-wise processing"
  transpose scoresT scores

  comment "Softmax over tokens for each expert (column-wise on original)"
  colMax maxVec scores
  subRow scores scores maxVec
  exp scores scores
  colSum sumVec scores
  divRow routingWeights scores sumVec

  comment "Each expert selects top-C tokens"
  comment "(In actual implementation: use argmax/top-k selection)"
  copy selections routingWeights  -- Placeholder

  comment "Store routing decisions"
  convert out routingWeights
  store outShared out
  sync

def moeExpertChoiceFwdKernel : Kernel :=
  buildKernelM "moe_expert_choice_fwd" .SM90 #[
    { name := "scores", dtype := .Float32, isPointer := true },
    { name := "routing_weights", dtype := .BFloat16, isPointer := true },
    { name := "selections", dtype := .Float32, isPointer := true },
    { name := "num_tokens", dtype := .Float32, isPointer := false },
    { name := "num_experts", dtype := .Float32, isPointer := false },
    { name := "capacity", dtype := .Float32, isPointer := false }
  ] moeExpertChoiceFwd

/-! ## Full MOE Block

Fused MOE computation: gating + dispatch + expert FFN + combine.
-/

/-- Full MOE FFN block -/
@[gpu_kernel .SM90]
def moeFfnFwd : KernelM Unit := do
  comment "=== MOE FFN Block ==="
  comment "Full MOE: gate -> dispatch -> expert FFN -> combine"

  -- Input tokens
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Gating weights
  let wGate : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Expert FFN weights (simplified: shared across experts)
  let wUp : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let wDown : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Intermediate tiles
  let gatingScores : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let routingProbs : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let hidden : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let expertOut : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let combined : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Softmax tracking
  let maxVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wGateShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let wUpShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let wDownShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load weights (long-resident)"
  load wGate wGateShared
  load wUp wUpShared
  load wDown wDownShared

  comment "Process token batches"
  forLoop 0 4 do
    comment "Load tokens"
    load x xShared

    comment "Step 1: Compute gating scores"
    mma gatingScores x wGate gatingScores

    comment "Step 2: Softmax routing"
    rowMax maxVec gatingScores
    subCol gatingScores gatingScores maxVec
    exp gatingScores gatingScores
    rowSum sumVec gatingScores
    divCol routingProbs gatingScores sumVec

    comment "Step 3: Expert FFN (simplified: all tokens through same expert)"
    -- Up projection
    mma hidden x wUp hidden
    -- Activation
    gelu hidden hidden
    -- Down projection
    let hiddenBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert hiddenBf hidden
    let hiddenCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
    swapLayout hiddenCol hiddenBf
    mma expertOut hiddenBf wDown expertOut

    comment "Step 4: Weight by routing probability and accumulate"
    -- In full implementation: loop over top-K experts
    mul combined expertOut routingProbs
    add combined combined expertOut

    comment "Store output"
    convert out combined
    store outShared out

    sync

def moeFfnFwdKernel : Kernel :=
  buildKernelM "moe_ffn_fwd" .SM90 #[
    { name := "x", dtype := .BFloat16, isPointer := true },
    { name := "w_gate", dtype := .BFloat16, isPointer := true },
    { name := "w_up", dtype := .BFloat16, isPointer := true },
    { name := "w_down", dtype := .BFloat16, isPointer := true },
    { name := "out", dtype := .BFloat16, isPointer := true },
    { name := "num_tokens", dtype := .Float32, isPointer := false },
    { name := "hidden_dim", dtype := .Float32, isPointer := false },
    { name := "num_experts", dtype := .Float32, isPointer := false }
  ] moeFfnFwd

-- Print generated kernels
#eval IO.println "=== MOE Gating ===" *> IO.println (generateKernel moeGatingFwdKernel)
#eval IO.println "\n=== MOE Dispatch ===" *> IO.println (generateKernel moeDispatchFwdKernel)
#eval IO.println "\n=== MOE Combine ===" *> IO.println (generateKernel moeCombineFwdKernel)
#eval IO.println "\n=== Expert-Choice MOE ===" *> IO.println (generateKernel moeExpertChoiceFwdKernel)
#eval IO.println "\n=== MOE FFN ===" *> IO.println (generateKernel moeFfnFwdKernel)

end Tyr.GPU.Kernels.MOE
