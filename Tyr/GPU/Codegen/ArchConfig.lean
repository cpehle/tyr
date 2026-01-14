/-
  Tyr/GPU/Codegen/ArchConfig.lean

  Architecture-specific configuration for GPU kernel generation.
  Supports specialized kernels for SM80 (Ampere), SM90 (Hopper), SM100 (Blackwell).
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew

namespace Tyr.GPU

/-! ## Architecture Capabilities -/

/-- Capabilities available on each architecture -/
structure ArchCapabilities where
  /-- Supports TMA (Tensor Memory Accelerator) -/
  hasTMA : Bool
  /-- Supports WGMMA (Warpgroup MMA) -/
  hasWGMMA : Bool
  /-- Supports FP8 datatypes -/
  hasFP8 : Bool
  /-- Supports asynchronous MMA -/
  hasAsyncMMA : Bool
  /-- Supports distributed shared memory -/
  hasDSM : Bool
  /-- Maximum shared memory per SM (bytes) -/
  maxSharedMem : Nat
  /-- Number of SMs (varies by SKU, using common values) -/
  typicalSMs : Nat
  /-- Tensor core throughput (TFLOPS for bf16) -/
  tensorCoreTFLOPS : Nat
  /-- Optimal tile size for MMA -/
  optimalMmaTile : Nat × Nat × Nat  -- M, N, K
  /-- Supports barrier cluster operations -/
  hasClusterBarrier : Bool
  deriving Repr, Inhabited

/-- Get capabilities for an architecture -/
def GpuArch.capabilities : GpuArch → ArchCapabilities
  | .SM80 => {
      hasTMA := false
      hasWGMMA := false
      hasFP8 := false
      hasAsyncMMA := true
      hasDSM := false
      maxSharedMem := 164 * 1024  -- 164 KB
      typicalSMs := 108           -- A100
      tensorCoreTFLOPS := 312     -- A100 bf16
      optimalMmaTile := (16, 16, 16)
      hasClusterBarrier := false
    }
  | .SM90 => {
      hasTMA := true
      hasWGMMA := true
      hasFP8 := true
      hasAsyncMMA := true
      hasDSM := true
      maxSharedMem := 228 * 1024  -- 228 KB
      typicalSMs := 132           -- H100 SXM
      tensorCoreTFLOPS := 989     -- H100 bf16 (with sparsity: 1979)
      optimalMmaTile := (64, 64, 16)
      hasClusterBarrier := true
    }
  | .SM100 => {
      hasTMA := true
      hasWGMMA := true
      hasFP8 := true
      hasAsyncMMA := true
      hasDSM := true
      maxSharedMem := 256 * 1024  -- 256 KB (estimated)
      typicalSMs := 160           -- B200 (estimated)
      tensorCoreTFLOPS := 2250    -- B200 bf16 (estimated)
      optimalMmaTile := (64, 64, 32)
      hasClusterBarrier := true
    }

/-- Check if architecture supports a dtype -/
def GpuArch.supportsDtype (arch : GpuArch) (dtype : GpuFloat) : Bool :=
  match dtype with
  | .Float32 | .Float16 | .BFloat16 => true
  | .FP8E4M3 | .FP8E5M2 => arch.capabilities.hasFP8

end Tyr.GPU

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Architecture-Specific Kernel Configuration -/

/-- Configuration for architecture-specialized kernel generation -/
structure ArchKernelConfig where
  /-- Target architecture -/
  arch : GpuArch
  /-- Block size (threads per block) -/
  blockSize : Nat := 128
  /-- Number of warps per block -/
  warpsPerBlock : Nat := 4
  /-- Use TMA if available -/
  useTMA : Bool := true
  /-- Use WGMMA if available -/
  useWGMMA : Bool := true
  /-- Tile dimensions for MMA -/
  mmaTileM : Nat := 64
  mmaTileN : Nat := 64
  mmaTileK : Nat := 64
  /-- Number of pipeline stages -/
  pipelineStages : Nat := 2
  /-- Cluster size (for SM90+) -/
  clusterSize : Nat := 1
  deriving Repr, Inhabited

/-- Get default configuration for an architecture -/
def ArchKernelConfig.default (arch : GpuArch) : ArchKernelConfig :=
  let caps := arch.capabilities
  let (optM, optN, _optK) := caps.optimalMmaTile
  {
    arch := arch
    blockSize := if caps.hasWGMMA then 128 else 256
    warpsPerBlock := if caps.hasWGMMA then 4 else 8
    useTMA := caps.hasTMA
    useWGMMA := caps.hasWGMMA
    mmaTileM := optM
    mmaTileN := optN
    mmaTileK := if caps.hasWGMMA then 64 else 32
    pipelineStages := if caps.hasTMA then 4 else 2
    clusterSize := if caps.hasClusterBarrier then 2 else 1
  }

/-! ## Multi-Architecture Kernel Generation -/

/-- Result of multi-arch kernel generation -/
structure MultiArchKernel where
  /-- Kernels for each target architecture -/
  kernels : Array (GpuArch × Kernel)
  /-- Combined source code with arch guards -/
  combinedSource : String
  deriving Repr

/-- Generate kernel for multiple architectures -/
def generateMultiArch (name : String) (params : Array KParam)
    (archs : Array GpuArch)
    (genKernel : ArchKernelConfig → KernelM Unit) : MultiArchKernel :=
  let kernels := archs.map fun arch =>
    let config := ArchKernelConfig.default arch
    let kernel := buildKernelM name arch params (genKernel config)
    (arch, kernel)

  let combinedSource := kernels.foldl (fun acc (_, kernel) =>
    let archCode := generateKernel kernel
    -- Remove duplicate headers for combined output
    let bodyOnly := archCode.splitOn "\n" |>.filter (fun line =>
      !line.startsWith "#include" && !line.startsWith "using namespace")
    let cleanCode := String.intercalate "\n" bodyOnly
    acc ++ cleanCode ++ "\n"
  ) "#include <kittens.cuh>\nusing namespace kittens;\n\n"

  { kernels, combinedSource }

/-- Generate kernel with automatic architecture selection -/
def generateAutoArch (name : String) (params : Array KParam)
    (genKernel : ArchKernelConfig → KernelM Unit) : MultiArchKernel :=
  generateMultiArch name params #[.SM80, .SM90, .SM100] genKernel

/-! ## Architecture-Conditional Operations -/

/-- Emit operation only if architecture supports TMA -/
def ifTMA (config : ArchKernelConfig) (tmaOp : KernelM Unit) (fallback : KernelM Unit)
    : KernelM Unit := do
  if config.useTMA && config.arch.capabilities.hasTMA then
    tmaOp
  else
    fallback

/-- Emit operation only if architecture supports WGMMA -/
def ifWGMMA (config : ArchKernelConfig) (wgmmaOp : KernelM Unit) (fallback : KernelM Unit)
    : KernelM Unit := do
  if config.useWGMMA && config.arch.capabilities.hasWGMMA then
    wgmmaOp
  else
    fallback

/-- Emit operation only if architecture supports FP8 -/
def ifFP8 (config : ArchKernelConfig) (fp8Op : KernelM Unit) (fallback : KernelM Unit)
    : KernelM Unit := do
  if config.arch.capabilities.hasFP8 then
    fp8Op
  else
    fallback

/-! ## Architecture-Aware Tile Selection -/

/-- Get optimal tile size for architecture -/
def getOptimalTileSize (arch : GpuArch) (operation : String := "mma") : Nat × Nat :=
  match arch with
  | .SM80 =>
    match operation with
    | "mma" => (64, 64)
    | "load" => (64, 64)
    | _ => (64, 64)
  | .SM90 =>
    match operation with
    | "mma" => (64, 128)  -- Larger tiles for WGMMA
    | "load" => (64, 64)
    | _ => (64, 64)
  | .SM100 =>
    match operation with
    | "mma" => (128, 128)  -- Even larger for Blackwell
    | "load" => (64, 128)
    | _ => (64, 64)

/-! ## Kernel Variant Generation -/

/-- Generate optimized kernel variant based on config -/
def optimizedKernelTemplate (config : ArchKernelConfig)
    (body : ArchKernelConfig → KernelM Unit) : KernelM Unit := do
  setArch config.arch

  -- Add architecture info as comment
  comment s!"Target: {config.arch} (TMA: {config.useTMA}, WGMMA: {config.useWGMMA})"
  comment s!"Tile: {config.mmaTileM}x{config.mmaTileN}x{config.mmaTileK}, Stages: {config.pipelineStages}"

  body config

/-! ## Example: Architecture-Specialized FlashAttention -/

/-- FlashAttention kernel generator that adapts to architecture -/
def flashAttnMultiArch (config : ArchKernelConfig) : KernelM Unit := do
  setArch config.arch
  comment s!"=== FlashAttention for {config.arch} ==="

  let tileM := config.mmaTileM
  let tileN := config.mmaTileN

  comment s!"Tiles: {tileM}x{tileN}"

  -- Allocate tiles with architecture-optimal sizes
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let s : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let rowMax : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let rowSum : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col

  load q qShared

  forLoop 0 config.pipelineStages do
    load k kShared
    load v vShared
    mmaT s q k s
    makeCausal s s (some (-1e10))
    rowMaxAccum rowMax s rowMax
    subCol s s rowMax
    exp s s
    rowSumAccum rowSum s rowSum

    let p : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert p s
    mma o p v o
    sync

  divCol o o rowSum

/-- Generate FlashAttention for all architectures -/
def flashAttnAllArchs : MultiArchKernel :=
  generateAutoArch "flash_attn" #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true }
  ] flashAttnMultiArch

-- Print multi-arch kernel
#eval IO.println flashAttnAllArchs.combinedSource

end Tyr.GPU.Codegen
