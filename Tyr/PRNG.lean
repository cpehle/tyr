import Std

namespace torch

/-!
  PRNG utilities (JAX-style key splitting).

  This is a lightweight, deterministic PRNG for CPU-side sampling used by
  DiffEq utilities (e.g. Brownian paths). It is *not* cryptographically secure.
-/

structure PRNGKey where
  state : UInt64
  deriving Repr, BEq, Inhabited

namespace PRNGKey

private def lcgA : UInt64 := 6364136223846793005
private def lcgC : UInt64 := 1442695040888963407
private def pi : Float := 3.14159265358979323846

private def mix (x : UInt64) : UInt64 :=
  x * lcgA + lcgC

private def toUInt64 (x : UInt32) : UInt64 :=
  UInt64.ofNat x.toNat

def fromUInt64 (seed : UInt64) : PRNGKey :=
  { state := mix seed }

def foldIn (key : PRNGKey) (tag : UInt32) : PRNGKey :=
  { state := mix (key.state + toUInt64 tag + 0x9e3779b97f4a7c15) }

def split (key : PRNGKey) : PRNGKey × PRNGKey :=
  let k1 := { state := mix (key.state + 0x9e3779b97f4a7c15) }
  let k2 := { state := mix (k1.state + 0x9e3779b97f4a7c15) }
  (k1, k2)

private def uniform01From (x : UInt64) : Float :=
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  (Float.ofNat mant) / denom

def normal01 (key : PRNGKey) (tag : UInt32) : Float :=
  let base := key.state + toUInt64 tag
  let u1 := uniform01From (mix (base + 0x9e3779b97f4a7c15))
  let u2 := uniform01From (mix (base + 0xbf58476d1ce4e5b9))
  let u1 := if u1 <= 1e-12 then 1e-12 else u1
  let r := Float.sqrt (-2.0 * Float.log u1)
  let theta := 2.0 * pi * u2
  r * Float.cos theta

end PRNGKey
end torch
