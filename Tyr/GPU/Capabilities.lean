/-
  Tyr/GPU/Capabilities.lean

  Architecture capability system for compile-time feature gating.
  Operations unavailable on older GPUs won't compile.
-/
import Tyr.GPU.Types

namespace Tyr.GPU

/-- Architecture-specific capabilities -/
class GpuCapabilities (arch : GpuArch) where
  /-- Maximum shared memory in bytes -/
  maxSharedMem : Nat
  /-- Supported floating point types -/
  supportedTypes : List GpuFloat
  /-- Has Tensor Memory Accelerator (TMA) -/
  hasTMA : Bool
  /-- Has Warpgroup Matrix Multiply Accumulate (WGMMA) -/
  hasWGMMA : Bool
  /-- Supports async copy -/
  hasAsyncCopy : Bool
  /-- Maximum registers per thread -/
  maxRegistersPerThread : Nat
  /-- Base tile size for MMA operations -/
  mmaTileSize : Nat × Nat × Nat  -- M, N, K

instance : GpuCapabilities .SM80 where
  maxSharedMem := 164 * 1024  -- 164 KB
  supportedTypes := [.Float32, .Float16, .BFloat16]
  hasTMA := false
  hasWGMMA := false
  hasAsyncCopy := true
  maxRegistersPerThread := 255
  mmaTileSize := (16, 8, 16)  -- mma.sync shape

instance : GpuCapabilities .SM90 where
  maxSharedMem := 228 * 1024  -- 228 KB
  supportedTypes := [.Float32, .Float16, .BFloat16, .FP8E4M3, .FP8E5M2]
  hasTMA := true
  hasWGMMA := true
  hasAsyncCopy := true
  maxRegistersPerThread := 255
  mmaTileSize := (64, 256, 16)  -- WGMMA shape

instance : GpuCapabilities .SM100 where
  maxSharedMem := 256 * 1024  -- 256 KB (estimated)
  supportedTypes := [.Float32, .Float16, .BFloat16, .FP8E4M3, .FP8E5M2]
  hasTMA := true
  hasWGMMA := true
  hasAsyncCopy := true
  maxRegistersPerThread := 255
  mmaTileSize := (64, 256, 16)  -- tcgen05 shape

/-- Require TMA support for certain operations -/
class RequiresTMA (arch : GpuArch) [GpuCapabilities arch] where
  h_tma : GpuCapabilities.hasTMA arch = true

instance : RequiresTMA .SM90 where
  h_tma := rfl

instance : RequiresTMA .SM100 where
  h_tma := rfl

/-- Require WGMMA support for warpgroup MMA -/
class RequiresWGMMA (arch : GpuArch) [GpuCapabilities arch] where
  h_wgmma : GpuCapabilities.hasWGMMA arch = true

instance : RequiresWGMMA .SM90 where
  h_wgmma := rfl

instance : RequiresWGMMA .SM100 where
  h_wgmma := rfl

/-- Check if a dtype is supported on an architecture -/
def GpuCapabilities.supportsType [caps : GpuCapabilities arch] (dtype : GpuFloat) : Bool :=
  caps.supportedTypes.contains dtype

/-- Compute shared memory required for a tile -/
def sharedMemForTile (dtype : GpuFloat) (rows cols : Nat) : Nat :=
  rows * cols * dtype.bytes

/-- Check if a tile fits in shared memory -/
def tileFitsShared [caps : GpuCapabilities arch] (dtype : GpuFloat) (rows cols : Nat) : Bool :=
  sharedMemForTile dtype rows cols ≤ caps.maxSharedMem

end Tyr.GPU
