/-
  Tyr/GPU/Types.lean

  Core types for ThunderKittens GPU kernel abstraction.
  These types map directly to ThunderKittens C++ template parameters.
-/
namespace Tyr.GPU

/-- GPU floating point types, matching ThunderKittens supported dtypes -/
inductive GpuFloat where
  | Float32   -- float
  | Float16   -- half
  | BFloat16  -- bf16
  | FP8E4M3   -- fp8e4m3 (Hopper/Blackwell)
  | FP8E5M2   -- fp8e5m2 (Hopper/Blackwell)
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

instance : ToString GpuFloat where
  toString
    | .Float32 => "Float32"
    | .Float16 => "Float16"
    | .BFloat16 => "BFloat16"
    | .FP8E4M3 => "FP8E4M3"
    | .FP8E5M2 => "FP8E5M2"

/-- Convert GpuFloat to C++ type string -/
def GpuFloat.toCpp : GpuFloat → String
  | .Float32 => "float"
  | .Float16 => "half"
  | .BFloat16 => "bf16"
  | .FP8E4M3 => "fp8e4m3"
  | .FP8E5M2 => "fp8e5m2"

/-- Bytes per element for each dtype -/
def GpuFloat.bytes : GpuFloat → Nat
  | .Float32 => 4
  | .Float16 => 2
  | .BFloat16 => 2
  | .FP8E4M3 => 1
  | .FP8E5M2 => 1

/-- Tile memory location -/
inductive TileLoc where
  | Register   -- rt<T, rows, cols, layout> - per-thread
  | Shared     -- st<T, rows, cols, layout> - warp-cooperative
  | Global     -- gl<T, ...> - device memory
  | TensorCore -- tt - Blackwell TMA descriptors
  deriving Repr, BEq, Hashable, Inhabited

/-- Tile layout (row-major vs column-major) -/
inductive TileLayout where
  | Row  -- row_l in ThunderKittens
  | Col  -- col_l in ThunderKittens
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

instance : ToString TileLayout where
  toString
    | .Row => "Row"
    | .Col => "Col"

/-- Convert TileLayout to C++ layout string -/
def TileLayout.toCpp : TileLayout → String
  | .Row => "row_l"
  | .Col => "col_l"

/-- GPU architecture generations -/
inductive GpuArch where
  | SM80   -- Ampere (A100)
  | SM90   -- Hopper (H100)
  | SM100  -- Blackwell (B200)
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

instance : ToString GpuArch where
  toString
    | .SM80 => "SM80"
    | .SM90 => "SM90"
    | .SM100 => "SM100"

/-- Convert GpuArch to C++ preprocessor guard -/
def GpuArch.toGuard : GpuArch → String
  | .SM80 => "KITTENS_SM80"
  | .SM90 => "KITTENS_HOPPER"
  | .SM100 => "KITTENS_BLACKWELL"

/-- Convert GpuArch to nvcc arch flag -/
def GpuArch.toNvccArch : GpuArch → String
  | .SM80 => "sm_80"
  | .SM90 => "sm_90"
  | .SM100 => "sm_100"

/-- Shared memory swizzle mode for bank conflict avoidance -/
inductive SwizzleMode where
  | None
  | Swizzle128B  -- 128-byte swizzle
  | Swizzle64B   -- 64-byte swizzle
  | Swizzle32B   -- 32-byte swizzle
  deriving Repr, BEq, Hashable, Inhabited

/-- Execution scope for operations -/
inductive Scope where
  | Thread     -- Single thread operations
  | Warp       -- 32-thread warp operations
  | Warpgroup  -- 128-thread warpgroup (4 warps)
  | Block      -- Full thread block
  deriving Repr, BEq, Hashable, Inhabited

/-- Number of threads per scope -/
def Scope.threads : Scope → Nat
  | .Thread => 1
  | .Warp => 32
  | .Warpgroup => 128
  | .Block => 1024  -- max, varies by kernel

end Tyr.GPU
