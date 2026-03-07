import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqBrownianOptionParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

@[test] def testBrownianOptionNoneSemantics : IO Unit := do
  let tree : VirtualBrownianTree (Option Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 91001
    shape := none
  }

  let inc01 := VirtualBrownianTree.increment tree 0.0 1.0
  let inc05 := VirtualBrownianTree.increment tree 0.0 0.5
  let inc51 := VirtualBrownianTree.increment tree 0.5 1.0
  LeanTest.assertTrue (inc01.W == none && inc05.W == none && inc51.W == none)
    "Option none should stay none for Brownian increments"
  LeanTest.assertTrue (approx inc01.dt (inc05.dt + inc51.dt) 1e-12)
    s!"Option none dt additivity mismatch: {inc01.dt} vs {inc05.dt + inc51.dt}"

  let st := VirtualBrownianTree.incrementSpaceTime tree 0.1 0.8
  LeanTest.assertTrue (st.W == none && st.H == none)
    "Option none should stay none for space-time Levy areas"

  let stt := VirtualBrownianTree.incrementSpaceTimeTime tree 0.2 0.9
  LeanTest.assertTrue (stt.W == none && stt.H == none && stt.K == none)
    "Option none should stay none for space-time-time Levy areas"

@[test] def testBrownianOptionSomeAdditivityAndConsistency : IO Unit := do
  let tree : VirtualBrownianTree (Option Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 91002
    shape := some 0.0
  }

  let inc01 := VirtualBrownianTree.increment tree 0.0 1.0
  let inc05 := VirtualBrownianTree.increment tree 0.0 0.5
  let inc51 := VirtualBrownianTree.increment tree 0.5 1.0
  match inc01.W, inc05.W, inc51.W with
  | some w01, some w05, some w51 =>
      LeanTest.assertTrue (approx w01 (w05 + w51) 1e-6)
        s!"Option some Brownian additivity mismatch: {w01} vs {w05 + w51}"
  | _, _, _ =>
      LeanTest.fail "Option some should produce some Brownian increment"

  let incA := VirtualBrownianTree.increment tree 0.2 0.9
  let incB := VirtualBrownianTree.increment tree 0.2 0.9
  match incA.W, incB.W with
  | some wA, some wB =>
      LeanTest.assertTrue (approx wA wB 1e-12)
        s!"Option some Brownian increment should be deterministic: {wA} vs {wB}"
  | _, _ =>
      LeanTest.fail "Option some determinism check expected some increments"

  let stA := VirtualBrownianTree.incrementSpaceTime tree 0.15 0.85
  let stB := VirtualBrownianTree.incrementSpaceTime tree 0.15 0.85
  match stA.W, stA.H, stB.W, stB.H with
  | some wA, some hA, some wB, some hB =>
      LeanTest.assertTrue (approx wA wB 1e-12)
        s!"Option some ST Brownian deterministic mismatch: {wA} vs {wB}"
      LeanTest.assertTrue (approx hA hB 1e-12)
        s!"Option some ST Levy deterministic mismatch: {hA} vs {hB}"
  | _, _, _, _ =>
      LeanTest.fail "Option some should produce some ST increments"

  let sttForward := VirtualBrownianTree.incrementSpaceTimeTime tree 0.25 0.9
  let sttReverse := VirtualBrownianTree.incrementSpaceTimeTime tree 0.9 0.25
  match sttForward.W, sttForward.H, sttForward.K, sttReverse.W, sttReverse.H, sttReverse.K with
  | some wf, some hf, some kf, some wr, some hr, some kr =>
      LeanTest.assertTrue (approx wr (-wf) 1e-12)
        s!"Option some STT reverse W mismatch: {wr} vs {-wf}"
      LeanTest.assertTrue (approx hr (-hf) 1e-12)
        s!"Option some STT reverse H mismatch: {hr} vs {-hf}"
      LeanTest.assertTrue (approx kr (-kf) 1e-12)
        s!"Option some STT reverse K mismatch: {kr} vs {-kf}"
  | _, _, _, _, _, _ =>
      LeanTest.fail "Option some should produce some STT increments"

@[test] def testAbstractBrownianPathPointEvalAnchorsToPathStart : IO Unit := do
  /-
  Diffrax reference (`_path.py`): point evaluation of a control path should be anchored
  to the path's start, not interpreted as a zero-length increment from query time.
  -/
  let base : AbstractBrownianPath Float := {
    t0 := -2.0
    t1 := 3.0
    evaluate := fun t0 t1 _left => t0 + 10.0 * t1
    increment := fun t0 t1 => { dt := t1 - t0, W := t1 - t0 }
  }
  let asPath := AbstractBrownianPath.toPath base
  let point := asPath.evaluate 0.7 none true
  let pointExpected := base.evaluate base.t0 0.7 true
  LeanTest.assertTrue (approx point pointExpected 1.0e-12)
    s!"toPath point evaluation should anchor at path.t0: expected {pointExpected}, got {point}"

  let inc := asPath.evaluate 0.4 (some 0.9) true
  let incExpected := base.evaluate 0.4 0.9 true
  LeanTest.assertTrue (approx inc incExpected 1.0e-12)
    s!"toPath increment evaluation should preserve increment semantics: expected {incExpected}, got {inc}"

def run : IO Unit := do
  testBrownianOptionNoneSemantics
  testBrownianOptionSomeAdditivityAndConsistency
  testAbstractBrownianPathPointEvalAnchorsToPathStart

end Tests.DiffEqBrownianOptionParity
