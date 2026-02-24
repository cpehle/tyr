/-
  Tyr/GPU/Codegen/Arch/Level.lean

  Architecture capability levels forming a hierarchy with ordering proofs.
  SM80 (Ampere) ⊆ SM90 (Hopper) ⊆ SM100 (Blackwell)
-/
import Tyr.GPU.Types

/-!
# `Tyr.GPU.Codegen.Arch.Level`

Architecture-specific GPU code generation support for Level within the ThunderKittens backend.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Codegen.Arch

open Tyr.GPU

/-- Architecture capability levels forming a hierarchy -/
inductive ArchLevel where
  | Ampere     -- SM80: baseline (warp mma, cp.async, 164KB smem)
  | Hopper     -- SM90: +TMA, +WGMMA, +FP8, 228KB smem
  | Blackwell  -- SM100: +tcgen05, +2-CTA MMA, +FP4, 256KB smem
  deriving Repr, BEq, DecidableEq, Inhabited

instance : ToString ArchLevel where
  toString
    | .Ampere => "Ampere"
    | .Hopper => "Hopper"
    | .Blackwell => "Blackwell"

/-- Convert ArchLevel to numeric value for comparison -/
def ArchLevel.toNat : ArchLevel → Nat
  | .Ampere => 80
  | .Hopper => 90
  | .Blackwell => 100

/-- Proof that architecture `a` capabilities are subset of `b` -/
inductive ArchLe : ArchLevel → ArchLevel → Prop where
  | refl : ArchLe a a
  | ampere_hopper : ArchLe .Ampere .Hopper
  | hopper_blackwell : ArchLe .Hopper .Blackwell
  | ampere_blackwell : ArchLe .Ampere .Blackwell

instance : LE ArchLevel where
  le := ArchLe

/-- Transitivity proof for ArchLe -/
theorem ArchLe.trans : ArchLe a b → ArchLe b c → ArchLe a c := by
  intro hab hbc
  cases hab with
  | refl => exact hbc
  | ampere_hopper =>
    cases hbc with
    | refl => exact .ampere_hopper
    | hopper_blackwell => exact .ampere_blackwell
  | hopper_blackwell =>
    cases hbc with
    | refl => exact .hopper_blackwell
  | ampere_blackwell =>
    cases hbc with
    | refl => exact .ampere_blackwell

/-- Decidable ordering on ArchLevel -/
def ArchLevel.decLe (a b : ArchLevel) : Decidable (a ≤ b) :=
  match a, b with
  | .Ampere, .Ampere => isTrue .refl
  | .Ampere, .Hopper => isTrue .ampere_hopper
  | .Ampere, .Blackwell => isTrue .ampere_blackwell
  | .Hopper, .Ampere => isFalse (by intro h; cases h)
  | .Hopper, .Hopper => isTrue .refl
  | .Hopper, .Blackwell => isTrue .hopper_blackwell
  | .Blackwell, .Ampere => isFalse (by intro h; cases h)
  | .Blackwell, .Hopper => isFalse (by intro h; cases h)
  | .Blackwell, .Blackwell => isTrue .refl

instance (a b : ArchLevel) : Decidable (a ≤ b) := ArchLevel.decLe a b

/-- Strict ordering -/
def ArchLevel.lt (a b : ArchLevel) : Prop :=
  a.toNat < b.toNat

instance : LT ArchLevel where
  lt := ArchLevel.lt

instance (a b : ArchLevel) : Decidable (a < b) :=
  Nat.decLt a.toNat b.toNat

/-- Convert ArchLevel to GpuArch (the existing type in Types.lean) -/
def ArchLevel.toGpuArch : ArchLevel → GpuArch
  | .Ampere => .SM80
  | .Hopper => .SM90
  | .Blackwell => .SM100

/-- Convert GpuArch to ArchLevel -/
def GpuArch.toArchLevel : GpuArch → ArchLevel
  | .SM80 => .Ampere
  | .SM90 => .Hopper
  | .SM100 => .Blackwell

/-- Convert ArchLevel to C++ preprocessor guard -/
def ArchLevel.toGuard : ArchLevel → String
  | .Ampere => "KITTENS_AMPERE"
  | .Hopper => "KITTENS_HOPPER"
  | .Blackwell => "KITTENS_BLACKWELL"

/-- Convert ArchLevel to nvcc arch flag -/
def ArchLevel.toNvccArch : ArchLevel → String
  | .Ampere => "sm_80a"
  | .Hopper => "sm_90a"
  | .Blackwell => "sm_100a"

/-- Convert ArchLevel to suffix for generated names -/
def ArchLevel.toSuffix : ArchLevel → String
  | .Ampere => "_SM80"
  | .Hopper => "_SM90"
  | .Blackwell => "_SM100"

/-- Convert ArchLevel to Lean Name suffix -/
def ArchLevel.toNameSuffix : ArchLevel → Lean.Name
  | .Ampere => `SM80
  | .Hopper => `SM90
  | .Blackwell => `SM100

/-- All architecture levels -/
def ArchLevel.all : Array ArchLevel := #[.Ampere, .Hopper, .Blackwell]

/-- Check if operation available at given architecture -/
def ArchLevel.supports (arch minArch : ArchLevel) : Bool :=
  minArch.toNat ≤ arch.toNat

/-- Proof that Ampere is supported by all architectures -/
theorem ArchLevel.ampere_base (arch : ArchLevel) : ArchLevel.Ampere ≤ arch := by
  cases arch with
  | Ampere => exact .refl
  | Hopper => exact .ampere_hopper
  | Blackwell => exact .ampere_blackwell

end Tyr.GPU.Codegen.Arch
