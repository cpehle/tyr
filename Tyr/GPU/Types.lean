import Lean.ToExpr

/-!
# Tyr.GPU.Types

`Tyr.GPU.Types` defines the foundational enums and configuration types used
throughout Tyr's GPU DSL. These values are shared by:

- type-level kernel APIs (`TileLayout`, `GpuFloat`, `GpuArch`),
- capability checks and architecture dispatch,
- C++ emission (`toCpp`, guard names, byte sizes).

Most definitions here are intentionally "small but central": they are the
common language between Lean-side kernel construction and emitted
ThunderKittens/CUDA code.
-/

namespace Tyr.GPU

/-- GPU global-memory element types supported by the generated backend -/
inductive GpuFloat where
  | Int64     -- signed 64-bit integer storage (indices/targets)
  | Float32   -- float
  | Float16   -- half
  | BFloat16  -- bf16
  | FP8E4M3   -- fp8e4m3 (Hopper/Blackwell)
  | FP8E5M2   -- fp8e5m2 (Hopper/Blackwell)
  | FP8E8M0   -- fp8e8m0 (Blackwell MX scale tiles)
  | FP4E2M1X2 -- packed fp4e2m1_2 (Blackwell NVFP4 storage)
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

instance : ToString GpuFloat where
  toString
    | .Int64 => "Int64"
    | .Float32 => "Float32"
    | .Float16 => "Float16"
    | .BFloat16 => "BFloat16"
    | .FP8E4M3 => "FP8E4M3"
    | .FP8E5M2 => "FP8E5M2"
    | .FP8E8M0 => "FP8E8M0"
    | .FP4E2M1X2 => "FP4E2M1X2"

/-- Convert GpuFloat to C++ type string -/
def GpuFloat.toCpp : GpuFloat → String
  | .Int64 => "int64_t"
  | .Float32 => "float"
  | .Float16 => "half"
  | .BFloat16 => "bf16"
  | .FP8E4M3 => "fp8e4m3"
  | .FP8E5M2 => "fp8e5m2"
  | .FP8E8M0 => "fp8e8m0"
  | .FP4E2M1X2 => "fp4e2m1_2"

/-- Bytes per element for each dtype -/
def GpuFloat.bytes : GpuFloat → Nat
  | .Int64 => 8
  | .Float32 => 4
  | .Float16 => 2
  | .BFloat16 => 2
  | .FP8E4M3 => 1
  | .FP8E5M2 => 1
  | .FP8E8M0 => 1
  | .FP4E2M1X2 => 1

/-- Tile memory location -/
inductive TileLoc where
  | Register   -- rt<T, rows, cols, layout> - per-thread
  | Shared     -- st<T, rows, cols, layout> - warp-cooperative
  | Global     -- gl<T, ...> - device memory
  | TensorCore -- tt - Blackwell TMA descriptors
  deriving Repr, BEq, Hashable, Inhabited, Lean.ToExpr

/-- Tile layout (row-major vs column-major) -/
inductive TileLayout where
  | Row  -- row_l in ThunderKittens
  | Col  -- col_l in ThunderKittens
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

instance : ToString TileLayout where
  toString
    | .Row => "Row"
    | .Col => "Col"

/-- Convert TileLayout to C++ layout string -/
def TileLayout.toCpp : TileLayout → String
  | .Row => "row_l"
  | .Col => "col_l"

/-- Transpose layout (swap Row ↔ Col) -/
def TileLayout.transpose : TileLayout → TileLayout
  | .Row => .Col
  | .Col => .Row

/-- GPU architecture generations -/
inductive GpuArch where
  | SM80   -- Ampere (A100)
  | SM90   -- Hopper (H100)
  | SM100  -- Blackwell (B200)
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

instance : ToString GpuArch where
  toString
    | .SM80 => "SM80"
    | .SM90 => "SM90"
    | .SM100 => "SM100"

/-- Convert GpuArch to C++ preprocessor guard -/
def GpuArch.toGuard : GpuArch → String
  | .SM80 => "KITTENS_AMPERE"
  | .SM90 => "KITTENS_HOPPER"
  | .SM100 => "KITTENS_BLACKWELL"

/-- Convert GpuArch to nvcc arch flag -/
def GpuArch.toNvccArch : GpuArch → String
  | .SM80 => "sm_80a"
  | .SM90 => "sm_90a"
  | .SM100 => "sm_100a"

/-- GPU family used for emitted guard selection and build-family dispatch. -/
inductive GpuFamily where
  | Ampere
  | Hopper
  | Blackwell
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Lean.ToExpr

instance : ToString GpuFamily where
  toString
    | .Ampere => "Ampere"
    | .Hopper => "Hopper"
    | .Blackwell => "Blackwell"

/-- Convert a family to the C++ preprocessor guard used by the runtime build. -/
def GpuFamily.toGuard : GpuFamily → String
  | .Ampere => "KITTENS_AMPERE"
  | .Hopper => "KITTENS_HOPPER"
  | .Blackwell => "KITTENS_BLACKWELL"

/-- Default emitted/build family for a capability floor. -/
def GpuArch.toFamily : GpuArch → GpuFamily
  | .SM80 => .Ampere
  | .SM90 => .Hopper
  | .SM100 => .Blackwell

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
