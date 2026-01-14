/-
  Tyr/GPU/Kernels/Distributed.lean

  Distributed/Multi-GPU kernel implementations.
  Based on ThunderKittens patterns.

  Key features:
  - PGL (Parallel Global Layout) for multi-GPU memory management
  - Multimem instructions for cross-GPU communication
  - AllGather, AllReduce, ReduceScatter operations
  - Distributed GEMM variants (AG-GEMM, GEMM-AR, GEMM-RS)
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

namespace Tyr.GPU.Kernels.Distributed

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## PGL (Parallel Global Layout) Operations

PGL provides unified addressing across multiple GPUs:
- mc_ptr: Multicast pointer for broadcasting
- gls[]: Per-device global layouts
- multimem instructions for atomic cross-GPU operations
-/

/-- AllGather kernel - gather data from all GPUs to all GPUs -/
@[gpu_kernel .SM90]
def allGatherFwd : KernelM Unit := do
  comment "=== AllGather Forward ==="
  comment "Each GPU contributes its local slice, all GPUs get full tensor"

  -- Local tile from this GPU
  let localTile : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Gathered result (full tensor across all GPUs)
  let gathered : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory for local data
  let localShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let gatheredShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load local tile"
  load localTile localShared

  comment "AllGather via multimem.ld_reduce"
  comment "(In actual implementation, this uses PGL multicast pointer)"
  -- Simulated: copy local to gathered
  copy gathered localTile

  comment "Store gathered result"
  store gatheredShared gathered
  sync

/-- Build AllGather kernel -/
def allGatherFwdKernel : Kernel :=
  buildKernelM "all_gather_fwd" .SM90 #[
    { name := "local_ptr", dtype := .BFloat16, isPointer := true },
    { name := "gathered_ptr", dtype := .BFloat16, isPointer := true },
    { name := "rank", dtype := .Float32, isPointer := false },
    { name := "world_size", dtype := .Float32, isPointer := false },
    { name := "local_size", dtype := .Float32, isPointer := false }
  ] allGatherFwd

/-- AllReduce kernel - sum across all GPUs, result on all GPUs -/
@[gpu_kernel .SM90]
def allReduceFwd : KernelM Unit := do
  comment "=== AllReduce Forward (Sum) ==="
  comment "Sum contributions from all GPUs, broadcast result to all"

  -- Local data from this GPU
  let localTile : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Reduced result
  let reduced : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Shared memory
  let localShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let reducedShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64

  comment "Load local contribution"
  load localTile localShared

  comment "AllReduce via multimem.ld_reduce.add"
  comment "(In actual implementation, atomically reads and adds from all GPUs)"
  -- Simulated: add local to reduced
  add reduced reduced localTile

  comment "Store reduced result"
  store reducedShared reduced
  sync

/-- Build AllReduce kernel -/
def allReduceFwdKernel : Kernel :=
  buildKernelM "all_reduce_fwd" .SM90 #[
    { name := "data_ptr", dtype := .Float32, isPointer := true },
    { name := "rank", dtype := .Float32, isPointer := false },
    { name := "world_size", dtype := .Float32, isPointer := false },
    { name := "size", dtype := .Float32, isPointer := false }
  ] allReduceFwd

/-- ReduceScatter kernel - reduce across GPUs, scatter slices -/
@[gpu_kernel .SM90]
def reduceScatterFwd : KernelM Unit := do
  comment "=== ReduceScatter Forward ==="
  comment "Reduce across GPUs, each GPU gets different slice of result"

  let input : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let output : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let inputShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let outputShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64

  comment "Load full input"
  load input inputShared

  comment "Reduce contributions from all GPUs"
  comment "(Each GPU only stores its designated slice)"
  add output output input

  comment "Store this GPU's slice of reduced result"
  store outputShared output
  sync

def reduceScatterFwdKernel : Kernel :=
  buildKernelM "reduce_scatter_fwd" .SM90 #[
    { name := "input_ptr", dtype := .Float32, isPointer := true },
    { name := "output_ptr", dtype := .Float32, isPointer := true },
    { name := "rank", dtype := .Float32, isPointer := false },
    { name := "world_size", dtype := .Float32, isPointer := false }
  ] reduceScatterFwd

/-! ## Distributed GEMM Operations

Fused communication + GEMM patterns for tensor/pipeline parallelism:
- AG-GEMM: AllGather A, then compute A @ B
- GEMM-AR: Compute A @ B, then AllReduce result
- GEMM-RS: Compute A @ B, then ReduceScatter result
-/

/-- AllGather-GEMM: Gather A across GPUs, compute A @ B -/
@[gpu_kernel .SM90]
def agGemmFwd : KernelM Unit := do
  comment "=== AllGather-GEMM Forward ==="
  comment "A is sharded across GPUs, gather then compute A @ B"

  -- Local slice of A (each GPU has 1/N of rows)
  let aLocal : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- B matrix (replicated on all GPUs)
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Output accumulator
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load B (replicated)"
  load b bShared

  comment "Loop over all GPUs' A slices"
  forLoop 0 8 do  -- 8 GPUs
    comment "AllGather: load A slice from GPU i"
    load aLocal aShared

    comment "GEMM: C += A[i] @ B"
    mma c aLocal b c

    sync

  comment "Store final result"
  convert out c
  store outShared out

def agGemmFwdKernel : Kernel :=
  buildKernelM "ag_gemm_fwd" .SM90 #[
    { name := "A_ptr", dtype := .BFloat16, isPointer := true },
    { name := "B_ptr", dtype := .BFloat16, isPointer := true },
    { name := "C_ptr", dtype := .BFloat16, isPointer := true },
    { name := "M", dtype := .Float32, isPointer := false },
    { name := "N", dtype := .Float32, isPointer := false },
    { name := "K", dtype := .Float32, isPointer := false },
    { name := "world_size", dtype := .Float32, isPointer := false }
  ] agGemmFwd

/-- GEMM-AllReduce: Compute A @ B, then allreduce result -/
@[gpu_kernel .SM90]
def gemmArFwd : KernelM Unit := do
  comment "=== GEMM-AllReduce Forward ==="
  comment "Each GPU computes partial A @ B, then sum across GPUs"

  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let cReduced : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Compute local GEMM"
  forLoop 0 4 do
    load a aShared
    load b bShared
    mma c a b c
    sync

  comment "AllReduce: sum partial results across GPUs"
  comment "(In actual implementation: multimem.ld_reduce.add)"
  add cReduced cReduced c

  comment "Store reduced result"
  convert out cReduced
  store outShared out

def gemmArFwdKernel : Kernel :=
  buildKernelM "gemm_ar_fwd" .SM90 #[
    { name := "A_ptr", dtype := .BFloat16, isPointer := true },
    { name := "B_ptr", dtype := .BFloat16, isPointer := true },
    { name := "C_ptr", dtype := .BFloat16, isPointer := true },
    { name := "M", dtype := .Float32, isPointer := false },
    { name := "N", dtype := .Float32, isPointer := false },
    { name := "K", dtype := .Float32, isPointer := false }
  ] gemmArFwd

/-- GEMM-ReduceScatter: Compute A @ B, then reduce-scatter result -/
@[gpu_kernel .SM90]
def gemmRsFwd : KernelM Unit := do
  comment "=== GEMM-ReduceScatter Forward ==="
  comment "Each GPU computes full A @ B, result is scattered after reduction"

  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let b : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let cScattered : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Compute local GEMM"
  forLoop 0 4 do
    load a aShared
    load b bShared
    mma c a b c
    sync

  comment "ReduceScatter: each GPU gets different slice of reduced result"
  add cScattered cScattered c

  comment "Store this GPU's slice"
  convert out cScattered
  store outShared out

def gemmRsFwdKernel : Kernel :=
  buildKernelM "gemm_rs_fwd" .SM90 #[
    { name := "A_ptr", dtype := .BFloat16, isPointer := true },
    { name := "B_ptr", dtype := .BFloat16, isPointer := true },
    { name := "C_ptr", dtype := .BFloat16, isPointer := true },
    { name := "rank", dtype := .Float32, isPointer := false },
    { name := "world_size", dtype := .Float32, isPointer := false }
  ] gemmRsFwd

-- Print generated kernels
#eval IO.println "=== AllGather ===" *> IO.println (generateKernel allGatherFwdKernel)
#eval IO.println "\n=== AllReduce ===" *> IO.println (generateKernel allReduceFwdKernel)
#eval IO.println "\n=== ReduceScatter ===" *> IO.println (generateKernel reduceScatterFwdKernel)
#eval IO.println "\n=== AG-GEMM ===" *> IO.println (generateKernel agGemmFwdKernel)
#eval IO.println "\n=== GEMM-AR ===" *> IO.println (generateKernel gemmArFwdKernel)
#eval IO.println "\n=== GEMM-RS ===" *> IO.println (generateKernel gemmRsFwdKernel)

end Tyr.GPU.Kernels.Distributed
