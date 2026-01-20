import Tyr.DiffEq.Path
import Tyr.PRNG

namespace torch
namespace DiffEq

/-! ## Brownian Paths and Increments

Minimal data types for Brownian motion controls and Levy areas.
-/

structure BrownianIncrement (DT BM : Type) where
  dt : DT
  W : BM

structure SpaceTimeLevyArea (DT BM : Type) where
  dt : DT
  W : BM
  H : BM

structure SpaceTimeTimeLevyArea (DT BM : Type) where
  dt : DT
  W : BM
  H : BM
  K : BM

/-! ## Levy Area Interfaces -/

class BrownianIncrementLike (Control BM : Type) where
  dt : Control → Time
  W : Control → BM

class SpaceTimeLevyAreaLike (Control BM : Type) extends BrownianIncrementLike Control BM where
  H : Control → BM

class SpaceTimeTimeLevyAreaLike (Control BM : Type) extends SpaceTimeLevyAreaLike Control BM where
  K : Control → BM

class BrownianIncrementBuild (Control BM : Type) where
  build : Time → BM → Control

class SpaceTimeLevyAreaBuild (Control BM : Type) where
  build : Time → BM → BM → Control

class SpaceTimeTimeLevyAreaBuild (Control BM : Type) where
  build : Time → BM → BM → BM → Control

instance : BrownianIncrementLike (BrownianIncrement Time BM) BM where
  dt inc := inc.dt
  W inc := inc.W

instance : SpaceTimeLevyAreaLike (SpaceTimeLevyArea Time BM) BM where
  dt inc := inc.dt
  W inc := inc.W
  H inc := inc.H

instance : SpaceTimeTimeLevyAreaLike (SpaceTimeTimeLevyArea Time BM) BM where
  dt inc := inc.dt
  W inc := inc.W
  H inc := inc.H
  K inc := inc.K

instance : BrownianIncrementBuild (BrownianIncrement Time BM) BM where
  build dt w := { dt := dt, W := w }

instance : SpaceTimeLevyAreaBuild (SpaceTimeLevyArea Time BM) BM where
  build dt w h := { dt := dt, W := w, H := h }

instance : SpaceTimeTimeLevyAreaBuild (SpaceTimeTimeLevyArea Time BM) BM where
  build dt w h k := { dt := dt, W := w, H := h, K := k }

instance [DiffEqSpace BM] : DiffEqSpace (BrownianIncrement Time BM) where
  add a b := { dt := a.dt + b.dt, W := DiffEqSpace.add a.W b.W }
  sub a b := { dt := a.dt - b.dt, W := DiffEqSpace.sub a.W b.W }
  scale s a := { dt := s * a.dt, W := DiffEqSpace.scale s a.W }

instance [DiffEqSpace BM] : DiffEqSpace (SpaceTimeLevyArea Time BM) where
  add a b := { dt := a.dt + b.dt, W := DiffEqSpace.add a.W b.W, H := DiffEqSpace.add a.H b.H }
  sub a b := { dt := a.dt - b.dt, W := DiffEqSpace.sub a.W b.W, H := DiffEqSpace.sub a.H b.H }
  scale s a :=
    { dt := s * a.dt, W := DiffEqSpace.scale s a.W, H := DiffEqSpace.scale s a.H }

instance [DiffEqSpace BM] : DiffEqSpace (SpaceTimeTimeLevyArea Time BM) where
  add a b := {
    dt := a.dt + b.dt
    W := DiffEqSpace.add a.W b.W
    H := DiffEqSpace.add a.H b.H
    K := DiffEqSpace.add a.K b.K
  }
  sub a b := {
    dt := a.dt - b.dt
    W := DiffEqSpace.sub a.W b.W
    H := DiffEqSpace.sub a.H b.H
    K := DiffEqSpace.sub a.K b.K
  }
  scale s a := {
    dt := s * a.dt
    W := DiffEqSpace.scale s a.W
    H := DiffEqSpace.scale s a.H
    K := DiffEqSpace.scale s a.K
  }

structure AbstractBrownianPath (BM : Type) where
  t0 : Time
  t1 : Time
  evaluate : Time → Time → Bool → BM
  increment : Time → Time → BrownianIncrement Time BM

structure UnsafeBrownianPath (BM : Type) where
  t0 : Time
  t1 : Time
  seed : UInt64
  shape : BM

structure VirtualBrownianTree (BM : Type) where
  t0 : Time
  t1 : Time
  tol : Time
  maxDepth : Nat := 24
  seed : UInt64
  shape : BM

/-! ## Scalar Brownian Motion (Float) -/

private def timeToUInt32 (t : Time) : UInt32 :=
  (t * 1000000.0).toUInt64.toUInt32

private def baseKey (seed : UInt64) : PRNGKey :=
  PRNGKey.fromUInt64 seed

private def keyFoldTimes (key : PRNGKey) (t0 t1 : Time) : PRNGKey :=
  let key := PRNGKey.foldIn key (timeToUInt32 t0)
  PRNGKey.foldIn key (timeToUInt32 t1)

private def intervalKey (seed : UInt64) (t0 t1 : Time) : PRNGKey :=
  keyFoldTimes (PRNGKey.foldIn (baseKey seed) 0x9e3779b9) t0 t1

private def midpointKey (seed : UInt64) (t0 t1 : Time) : PRNGKey :=
  keyFoldTimes (PRNGKey.foldIn (baseKey seed) 0x243f6a88) t0 t1

private def pointKey (seed : UInt64) (t0 t1 t : Time) : PRNGKey :=
  let key := keyFoldTimes (PRNGKey.foldIn (baseKey seed) 0x13198a2e) t0 t1
  PRNGKey.foldIn key (timeToUInt32 t)

private def scalarIncrement (seed : UInt64) (t0 t1 : Time) : BrownianIncrement Time Float :=
  let dt := t1 - t0
  let scale := Float.sqrt (Float.abs dt)
  let key := intervalKey seed t0 t1
  { dt := dt, W := PRNGKey.normal01 key 0 * scale }

structure ScalarBrownianPath where
  t0 : Time
  t1 : Time
  seed : UInt64

namespace ScalarBrownianPath

def increment (path : ScalarBrownianPath) (t0 t1 : Time) : BrownianIncrement Time Float :=
  scalarIncrement path.seed t0 t1

def evaluate (path : ScalarBrownianPath) (t0 t1 : Time) (_left : Bool := true) : Float :=
  (increment path t0 t1).W

def toAbstract (path : ScalarBrownianPath) : AbstractBrownianPath Float := {
  t0 := path.t0
  t1 := path.t1
  evaluate := fun t0 t1 left => evaluate path t0 t1 left
  increment := fun t0 t1 => increment path t0 t1
}

end ScalarBrownianPath

namespace AbstractBrownianPath

def toPath (path : AbstractBrownianPath BM) : AbstractPath BM := {
  t0 := path.t0
  t1 := path.t1
  evaluate := fun t0 t1 left =>
    match t1 with
    | none => path.evaluate t0 t0 left
    | some t1 => path.evaluate t0 t1 left
}

end AbstractBrownianPath

namespace UnsafeBrownianPath

private def incrementSpaceTime (path : UnsafeBrownianPath Float) (t0 t1 : Time) :
    SpaceTimeLevyArea Time Float :=
  let dt := t1 - t0
  let scale := Float.sqrt (Float.abs dt)
  let key := intervalKey path.seed t0 t1
  let w := PRNGKey.normal01 (PRNGKey.foldIn key 1) 0 * scale
  let h := PRNGKey.normal01 (PRNGKey.foldIn key 2) 0 * (scale / Float.sqrt 12.0)
  { dt := dt, W := w, H := h }

private def incrementSpaceTimeTime (path : UnsafeBrownianPath Float) (t0 t1 : Time) :
    SpaceTimeTimeLevyArea Time Float :=
  let dt := t1 - t0
  let scale := Float.sqrt (Float.abs dt)
  let key := intervalKey path.seed t0 t1
  let w := PRNGKey.normal01 (PRNGKey.foldIn key 1) 0 * scale
  let h := PRNGKey.normal01 (PRNGKey.foldIn key 2) 0 * (scale / Float.sqrt 12.0)
  let k := PRNGKey.normal01 (PRNGKey.foldIn key 3) 0 * (scale / Float.sqrt 720.0)
  { dt := dt, W := w, H := h, K := k }

def toAbstractFloat (path : UnsafeBrownianPath Float) : AbstractBrownianPath Float :=
  ScalarBrownianPath.toAbstract { t0 := path.t0, t1 := path.t1, seed := path.seed }

def toAbstractSpaceTime (path : UnsafeBrownianPath Float) :
    AbstractBrownianPath (SpaceTimeLevyArea Time Float) :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun t0 t1 _left => incrementSpaceTime path t0 t1
    increment := fun t0 t1 => { dt := t1 - t0, W := incrementSpaceTime path t0 t1 }
  }

def toAbstractSpaceTimeTime (path : UnsafeBrownianPath Float) :
    AbstractBrownianPath (SpaceTimeTimeLevyArea Time Float) :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun t0 t1 _left => incrementSpaceTimeTime path t0 t1
    increment := fun t0 t1 => { dt := t1 - t0, W := incrementSpaceTimeTime path t0 t1 }
  }

end UnsafeBrownianPath

namespace VirtualBrownianTree

private def rootValue (seed : UInt64) (t0 t1 : Time) : Float :=
  let dt := t1 - t0
  let scale := Float.sqrt (Float.abs dt)
  PRNGKey.normal01 (intervalKey seed t0 t1) 0 * scale

private def bridgeValue (seed : UInt64) (t0 t1 : Time) (w0 w1 : Float) (t : Time) : Float :=
  let dt := t1 - t0
  if dt == 0.0 then
    w0
  else
    let dt0 := t - t0
    let dt1 := t1 - t
    let mean := w0 + (dt0 / dt) * (w1 - w0)
    let var := (dt0 * dt1) / dt
    mean + Float.sqrt (Float.abs var) * PRNGKey.normal01 (pointKey seed t0 t1 t) 0

private def midpointValue (seed : UInt64) (t0 t1 : Time) (w0 w1 : Float) : Float :=
  let dt := t1 - t0
  let mean := 0.5 * (w0 + w1)
  let var := dt / 4.0
  mean + Float.sqrt (Float.abs var) * PRNGKey.normal01 (midpointKey seed t0 t1) 0

private def valueAux (seed : UInt64) (t0 t1 : Time) (w0 w1 : Float) (tol : Time)
    (depth : Nat) (t : Time) : Float :=
  if t <= t0 then
    w0
  else if t >= t1 then
    w1
  else
    let dt := Float.abs (t1 - t0)
    match depth with
    | 0 =>
        bridgeValue seed t0 t1 w0 w1 t
    | Nat.succ depth =>
        if dt <= tol then
          bridgeValue seed t0 t1 w0 w1 t
        else
          let mid := (t0 + t1) / 2.0
          let wMid := midpointValue seed t0 t1 w0 w1
          if t == mid then
            wMid
          else if t < mid then
            valueAux seed t0 mid w0 wMid tol depth t
          else
            valueAux seed mid t1 wMid w1 tol depth t

def value (path : VirtualBrownianTree Float) (t : Time) : Float :=
  let w0 := 0.0
  let w1 := rootValue path.seed path.t0 path.t1
  valueAux path.seed path.t0 path.t1 w0 w1 path.tol path.maxDepth t

def increment (path : VirtualBrownianTree Float) (t0 t1 : Time) : BrownianIncrement Time Float :=
  let w0 := value path t0
  let w1 := value path t1
  { dt := t1 - t0, W := w1 - w0 }

def evaluate (path : VirtualBrownianTree Float) (t0 t1 : Time) (_left : Bool := true) : Float :=
  (increment path t0 t1).W

def toAbstractFloat (path : VirtualBrownianTree Float) : AbstractBrownianPath Float :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun t0 t1 left => evaluate path t0 t1 left
    increment := fun t0 t1 => increment path t0 t1
  }

private def pow2neg (n : Nat) : Float :=
  1.0 / Float.ofNat (Nat.pow 2 n)

private def safeInv (x : Float) : Float :=
  if Float.abs x < 1.0e-12 then 0.0 else 1.0 / x

private def safeInvSq (x : Float) : Float :=
  let inv := safeInv x
  inv * inv

private def safeSqrt (x : Float) : Float :=
  if x <= 0.0 then 0.0 else Float.sqrt x

private def keyWith (key : PRNGKey) (tag : UInt32) : PRNGKey :=
  PRNGKey.foldIn key tag

private def split3 (key : PRNGKey) : PRNGKey × PRNGKey × PRNGKey :=
  (keyWith key 0, keyWith key 1, keyWith key 2)

private def split4 (key : PRNGKey) : PRNGKey × PRNGKey × PRNGKey × PRNGKey :=
  (keyWith key 0, keyWith key 1, keyWith key 2, keyWith key 3)

private def normalKey (key : PRNGKey) (tag : UInt32) : Float :=
  PRNGKey.normal01 (keyWith key tag) 0

private structure LevyValW where
  dt : Time
  W : Float

private structure LevyValH where
  dt : Time
  W : Float
  H : Float
  barH : Float

private structure LevyValK where
  dt : Time
  W : Float
  H : Float
  barH : Float
  K : Float
  barK : Float

private structure StateW where
  level : Nat
  s : Time
  w_s : Float
  w_u : Float
  w_su : Float
  key : PRNGKey

private structure StateH where
  level : Nat
  s : Time
  w_s : Float
  w_u : Float
  w_su : Float
  key : PRNGKey
  bhh_s : Float
  bhh_u : Float
  bhh_su : Float

private structure StateK where
  level : Nat
  s : Time
  w_s : Float
  w_u : Float
  w_su : Float
  key : PRNGKey
  bhh_s : Float
  bhh_u : Float
  bhh_su : Float
  bkk_s : Float
  bkk_u : Float
  bkk_su : Float

private def stepW (r : Time) (state : StateW) : StateW :=
  let su := pow2neg state.level
  let st := su / 2.0
  let t := state.s + st
  let (key_st, key_mid, key_tu) := split3 state.key
  let z := normalKey key_mid 0
  let mean := state.w_su / 2.0
  let w_term2 := safeSqrt su / 2.0 * z
  let w_st := mean + w_term2
  let w_tu := mean - w_term2
  let w_t := state.w_s + w_st
  if r > t then
    {
      level := state.level + 1
      s := t
      w_s := w_t
      w_u := state.w_u
      w_su := w_tu
      key := key_st
    }
  else
    {
      level := state.level + 1
      s := state.s
      w_s := state.w_s
      w_u := w_t
      w_su := w_st
      key := key_tu
    }

private def stepH (r : Time) (state : StateH) : StateH :=
  let su := pow2neg state.level
  let st := su / 2.0
  let t := state.s + st
  let (key_st, key_mid, key_tu) := split3 state.key
  let z1 := normalKey key_mid 0
  let z2 := normalKey key_mid 1
  let root_su := safeSqrt su
  let z := z1 * (root_su / 4.0)
  let n := z2 * safeSqrt (su / 12.0)
  let w_term1 := state.w_su / 2.0
  let w_term2 := (3.0 / (2.0 * su)) * state.bhh_su + z
  let w_st := w_term1 + w_term2
  let w_tu := w_term1 - w_term2
  let bhh_term1 := state.bhh_su / 8.0 - su / 4.0 * z
  let bhh_term2 := (su / 4.0) * n
  let bhh_st := bhh_term1 + bhh_term2
  let bhh_tu := bhh_term1 - bhh_term2
  let w_t := state.w_s + w_st
  let t_bb_s := t * state.w_s - state.s * w_t
  let bhh_t := state.bhh_s + bhh_st + 0.5 * t_bb_s
  if r > t then
    {
      level := state.level + 1
      s := t
      w_s := w_t
      w_u := state.w_u
      w_su := w_tu
      key := key_st
      bhh_s := bhh_t
      bhh_u := state.bhh_u
      bhh_su := bhh_tu
    }
  else
    {
      level := state.level + 1
      s := state.s
      w_s := state.w_s
      w_u := w_t
      w_su := w_st
      key := key_tu
      bhh_s := state.bhh_s
      bhh_u := bhh_t
      bhh_su := bhh_st
    }

private def stepK (r : Time) (state : StateK) : StateK :=
  let su := pow2neg state.level
  let st := su / 2.0
  let t := state.s + st
  let (key_st, key_mid, key_tu) := split3 state.key
  let z1 := normalKey key_mid 0
  let z2 := normalKey key_mid 1
  let z3 := normalKey key_mid 2
  let z := z1 * safeSqrt (su / 16.0)
  let x1 := z2 * safeSqrt (su / 768.0)
  let x2 := z3 * safeSqrt (su / 2880.0)
  let su2 := su * su
  let w_term1 := state.w_su / 2.0
  let w_term2 := (3.0 / (2.0 * su)) * state.bhh_su + z
  let w_st := w_term1 + w_term2
  let w_tu := w_term1 - w_term2
  let bhh_term1 := state.bhh_su / 8.0 - (st / 2.0) * z
  let bhh_term2 := (15.0 / (8.0 * su)) * state.bkk_su + st * x1
  let bhh_st := bhh_term1 + bhh_term2
  let bhh_tu := bhh_term1 - bhh_term2
  let bkk_term1 := state.bkk_su / 32.0 - (su2 / 8.0) * x1
  let bkk_term2 := (su2 / 4.0) * x2
  let bkk_st := bkk_term1 + bkk_term2
  let bkk_tu := bkk_term1 - bkk_term2
  let w_t := state.w_s + w_st
  let t_bb_s := t * state.w_s - state.s * w_t
  let bhh_t := state.bhh_s + bhh_st + 0.5 * t_bb_s
  let bkk_t :=
    state.bkk_s + bkk_st + (st / 2.0) * state.bhh_s - (state.s / 2.0) * bhh_st +
      ((t - 2.0 * state.s) / 12.0) * t_bb_s
  if r > t then
    {
      level := state.level + 1
      s := t
      w_s := w_t
      w_u := state.w_u
      w_su := w_tu
      key := key_st
      bhh_s := bhh_t
      bhh_u := state.bhh_u
      bhh_su := bhh_tu
      bkk_s := bkk_t
      bkk_u := state.bkk_u
      bkk_su := bkk_tu
    }
  else
    {
      level := state.level + 1
      s := state.s
      w_s := state.w_s
      w_u := w_t
      w_su := w_st
      key := key_tu
      bhh_s := state.bhh_s
      bhh_u := bhh_t
      bhh_su := bhh_st
      bkk_s := state.bkk_s
      bkk_u := bkk_t
      bkk_su := bkk_st
    }

private def descendW (tol : Time) (maxDepth : Nat) (r : Time) (state : StateW) : StateW :=
  let rec loop (fuel : Nat) (st : StateW) : StateW :=
    match fuel with
    | 0 => st
    | Nat.succ fuel =>
        let su := pow2neg st.level
        if su <= tol then
          st
        else
          loop fuel (stepW r st)
  loop maxDepth state

private def descendH (tol : Time) (maxDepth : Nat) (r : Time) (state : StateH) : StateH :=
  let rec loop (fuel : Nat) (st : StateH) : StateH :=
    match fuel with
    | 0 => st
    | Nat.succ fuel =>
        let su := pow2neg st.level
        if su <= tol then
          st
        else
          loop fuel (stepH r st)
  loop maxDepth state

private def descendK (tol : Time) (maxDepth : Nat) (r : Time) (state : StateK) : StateK :=
  let rec loop (fuel : Nat) (st : StateK) : StateK :=
    match fuel with
    | 0 => st
    | Nat.succ fuel =>
        let su := pow2neg st.level
        if su <= tol then
          st
        else
          loop fuel (stepK r st)
  loop maxDepth state

private def evalW (path : VirtualBrownianTree Float) (r : Time) : LevyValW :=
  let len := path.t1 - path.t0
  let tol := path.tol / Float.abs len
  let key := keyWith (baseKey path.seed) 0x42524f57
  let (state_key, init_key_w, _) := split3 key
  let w1 := normalKey init_key_w 0
  let init : StateW := { level := 0, s := 0.0, w_s := 0.0, w_u := w1, w_su := w1, key := state_key }
  let st := descendW tol path.maxDepth r init
  let su := pow2neg st.level
  let sr := if r <= st.s then 0.0 else r - st.s
  let ru := if su <= sr then 0.0 else su - sr
  let w_mean := st.w_s + (sr * safeInv su) * st.w_su
  let z := normalKey st.key 0
  let bb := safeSqrt (sr * ru * safeInv su) * z
  { dt := r, W := w_mean + bb }

private def evalH (path : VirtualBrownianTree Float) (r : Time) : LevyValH :=
  let len := path.t1 - path.t0
  let tol := path.tol / Float.abs len
  let key := keyWith (baseKey path.seed) 0x484c4556
  let (state_key, init_key_w, init_key_hh) := split3 key
  let w1 := normalKey init_key_w 0
  let bhh_1 := normalKey init_key_hh 0 / Float.sqrt 12.0
  let init : StateH := {
    level := 0
    s := 0.0
    w_s := 0.0
    w_u := w1
    w_su := w1
    key := state_key
    bhh_s := 0.0
    bhh_u := bhh_1
    bhh_su := bhh_1
  }
  let st := descendH tol path.maxDepth r init
  let su := pow2neg st.level
  let sr := if r <= st.s then 0.0 else r - st.s
  let ru := if su <= sr then 0.0 else su - sr
  let sr3 := sr * sr * sr
  let ru3 := ru * ru * ru
  let su3 := su * su * su
  let x1 := normalKey st.key 0
  let x2 := normalKey st.key 1
  let sr_ru_half := safeSqrt (sr * ru)
  let d := safeSqrt (sr3 + ru3)
  let d_prime := safeInv (2.0 * su * d)
  let a := d_prime * sr3 * sr_ru_half
  let b := d_prime * ru3 * sr_ru_half
  let w_sr :=
    (sr * safeInv su) * st.w_su + (6.0 * sr * ru * safeInv su3) * st.bhh_su +
      (2.0 * (a + b) * safeInv su) * x1
  let w_r := st.w_s + w_sr
  let c := safeSqrt (3.0 * sr3 * ru3) * safeInv (6.0 * d)
  let bhh_sr := (sr3 * safeInv su3) * st.bhh_su - a * x1 + c * x2
  let bhh_r := st.bhh_s + bhh_sr + 0.5 * (r * st.w_s - st.s * w_r)
  let hh_r := safeInv r * bhh_r
  { dt := r, W := w_r, H := hh_r, barH := bhh_r }

private def sampleMVN3 (key : PRNGKey) (a b c d e f : Float) : Float × Float × Float :=
  let z1 := normalKey key 0
  let z2 := normalKey key 1
  let z3 := normalKey key 2
  let l11 := safeSqrt a
  let l21 := if l11 == 0.0 then 0.0 else b / l11
  let l31 := if l11 == 0.0 then 0.0 else c / l11
  let l22 := safeSqrt (d - l21 * l21)
  let l32 := if l22 == 0.0 then 0.0 else (e - l21 * l31) / l22
  let l33 := safeSqrt (f - l31 * l31 - l32 * l32)
  let x1 := l11 * z1
  let x2 := l21 * z1 + l22 * z2
  let x3 := l31 * z1 + l32 * z2 + l33 * z3
  (x1, x2, x3)

private def evalK (path : VirtualBrownianTree Float) (r : Time) : LevyValK :=
  let len := path.t1 - path.t0
  let tol := path.tol / Float.abs len
  let key := keyWith (baseKey path.seed) 0x4b4c4556
  let (state_key, init_key_w, init_key_hh, init_key_kk) := split4 key
  let w1 := normalKey init_key_w 0
  let bhh_1 := normalKey init_key_hh 0 / Float.sqrt 12.0
  let bkk_1 := normalKey init_key_kk 0 / Float.sqrt 720.0
  let init : StateK := {
    level := 0
    s := 0.0
    w_s := 0.0
    w_u := w1
    w_su := w1
    key := state_key
    bhh_s := 0.0
    bhh_u := bhh_1
    bhh_su := bhh_1
    bkk_s := 0.0
    bkk_u := bkk_1
    bkk_su := bkk_1
  }
  let st := descendK tol path.maxDepth r init
  let su := pow2neg st.level
  let sr := if r <= st.s then 0.0 else r - st.s
  let ru := if su <= sr then 0.0 else su - sr
  let su2 := su * su
  let sr2 := sr * sr
  let ru2 := ru * ru
  let su3 := su2 * su
  let sr_by_su := sr * safeInv su
  let sr_by_su_3 := sr_by_su * sr_by_su * sr_by_su
  let sr_by_su_5 := sr_by_su_3 * sr_by_su * sr_by_su
  let ru_by_su := ru * safeInv su
  let sr_ru_by_su2 := sr_by_su * ru_by_su
  let bb_mean :=
    (6.0 * sr_ru_by_su2 * safeInv su) * st.bhh_su +
      (120.0 * sr_ru_by_su2 * (0.5 - sr_by_su) * safeInv su2) * st.bkk_su
  let w_mean := sr_by_su * st.w_su + bb_mean
  let h_mean :=
    (sr_by_su * sr_by_su * safeInv su) * (st.bhh_su + (30.0 * ru_by_su * safeInv su) * st.bkk_su)
  let k_mean := (sr_by_su_3 * safeInv su2) * st.bkk_su
  let ww_cov :=
    sr_by_su * ru_by_su * (((sr - ru) * (sr - ru) * (sr - ru) * (sr - ru)) + 4.0 * (sr2 * ru2)) *
      safeInv su3
  let wh_cov :=
    -(sr_by_su_3 * ru_by_su * (sr2 - 3.0 * sr * ru + 6.0 * ru2)) * safeInv (2.0 * su)
  let wk_cov := (sr_by_su * sr_by_su * sr_by_su * sr_by_su) * ru_by_su * (sr - ru) / 12.0
  let hh_cov :=
    (sr / 12.0) * (1.0 - sr_by_su_3 * (sr2 + 2.0 * sr * ru + 16.0 * ru2) * safeInv su2)
  let hk_cov := -(ru / 24.0) * sr_by_su_5
  let kk_cov := (sr / 720.0) * (1.0 - sr_by_su_5)
  let (hat_w_sr, hat_hh_sr, hat_kk_sr) := sampleMVN3 st.key ww_cov wh_cov wk_cov hh_cov hk_cov kk_cov
  let w_sr := w_mean + hat_w_sr
  let w_r := st.w_s + w_sr
  let r_bb_s := r * st.w_s - st.s * w_r
  let bhh_sr := sr * (h_mean + hat_hh_sr)
  let bhh_r := st.bhh_s + bhh_sr + 0.5 * r_bb_s
  let bkk_sr := sr2 * (k_mean + hat_kk_sr)
  let bkk_r :=
    st.bkk_s + bkk_sr + (sr / 2.0) * st.bhh_s - (st.s / 2.0) * bhh_sr +
      ((r - 2.0 * st.s) / 12.0) * r_bb_s
  let hh_r := safeInv r * bhh_r
  let kk_r := safeInvSq r * bkk_r
  { dt := r, W := w_r, H := hh_r, barH := bhh_r, K := kk_r, barK := bkk_r }

private def diffW (x0 x1 : LevyValW) : BrownianIncrement Time Float :=
  { dt := x1.dt - x0.dt, W := x1.W - x0.W }

private def diffH (x0 x1 : LevyValH) : SpaceTimeLevyArea Time Float :=
  let su := x1.dt - x0.dt
  let w_su := x1.W - x0.W
  let u_bb_s := x1.dt * x0.W - x0.dt * x1.W
  let bhh_su := x1.barH - x0.barH - 0.5 * u_bb_s
  let hh_su := safeInv su * bhh_su
  { dt := su, W := w_su, H := hh_su }

private def diffK (x0 x1 : LevyValK) : SpaceTimeTimeLevyArea Time Float :=
  let su := x1.dt - x0.dt
  let w_su := x1.W - x0.W
  let u_bb_s := x1.dt * x0.W - x0.dt * x1.W
  let bhh_su := x1.barH - x0.barH - 0.5 * u_bb_s
  let hh_su := safeInv su * bhh_su
  let bkk_su :=
    x1.barK - x0.barK - (su / 2.0) * x0.barH + (x0.dt / 2.0) * bhh_su -
      ((x1.dt - 2.0 * x0.dt) / 12.0) * u_bb_s
  let kk_su := safeInvSq su * bkk_su
  { dt := su, W := w_su, H := hh_su, K := kk_su }

private def scaleW (len : Time) (inc : BrownianIncrement Time Float) : BrownianIncrement Time Float :=
  let scale := Float.sqrt (Float.abs len)
  { dt := inc.dt * len, W := inc.W * scale }

private def scaleH (len : Time) (inc : SpaceTimeLevyArea Time Float) :
    SpaceTimeLevyArea Time Float :=
  let scale := Float.sqrt (Float.abs len)
  { dt := inc.dt * len, W := inc.W * scale, H := inc.H * scale }

private def scaleK (len : Time) (inc : SpaceTimeTimeLevyArea Time Float) :
    SpaceTimeTimeLevyArea Time Float :=
  let scale := Float.sqrt (Float.abs len)
  { dt := inc.dt * len, W := inc.W * scale, H := inc.H * scale, K := inc.K * scale }

private def normalizeTime (path : VirtualBrownianTree Float) (t : Time) : Time :=
  (t - path.t0) / (path.t1 - path.t0)

def incrementSpaceTime (path : VirtualBrownianTree Float) (t0 t1 : Time) :
    SpaceTimeLevyArea Time Float :=
  if t0 == t1 then
    { dt := 0.0, W := 0.0, H := 0.0 }
  else
    let forward := t0 < t1
    let t0' := if forward then t0 else t1
    let t1' := if forward then t1 else t0
    let len := path.t1 - path.t0
    let r0 := normalizeTime path t0'
    let r1 := normalizeTime path t1'
    let v0 := evalH path r0
    let v1 := evalH path r1
    let inc := scaleH len (diffH v0 v1)
    if forward then inc else { dt := -inc.dt, W := -inc.W, H := -inc.H }

def incrementSpaceTimeTime (path : VirtualBrownianTree Float) (t0 t1 : Time) :
    SpaceTimeTimeLevyArea Time Float :=
  if t0 == t1 then
    { dt := 0.0, W := 0.0, H := 0.0, K := 0.0 }
  else
    let forward := t0 < t1
    let t0' := if forward then t0 else t1
    let t1' := if forward then t1 else t0
    let len := path.t1 - path.t0
    let r0 := normalizeTime path t0'
    let r1 := normalizeTime path t1'
    let v0 := evalK path r0
    let v1 := evalK path r1
    let inc := scaleK len (diffK v0 v1)
    if forward then inc else { dt := -inc.dt, W := -inc.W, H := -inc.H, K := -inc.K }

def toAbstractSpaceTime (path : VirtualBrownianTree Float) :
    AbstractBrownianPath (SpaceTimeLevyArea Time Float) :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun t0 t1 _left => incrementSpaceTime path t0 t1
    increment := fun t0 t1 => { dt := t1 - t0, W := incrementSpaceTime path t0 t1 }
  }

def toAbstractSpaceTimeTime (path : VirtualBrownianTree Float) :
    AbstractBrownianPath (SpaceTimeTimeLevyArea Time Float) :=
  {
    t0 := path.t0
    t1 := path.t1
    evaluate := fun t0 t1 _left => incrementSpaceTimeTime path t0 t1
    increment := fun t0 t1 => { dt := t1 - t0, W := incrementSpaceTimeTime path t0 t1 }
  }

end VirtualBrownianTree

end DiffEq
end torch
