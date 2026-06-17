import Tyr.Model.BranchingFlows.DiffEq
import Tyr.Model.BranchingFlows.Discrete

/-!
  Molecule-shaped helpers for BranchingFlows.

  QM9 elements are atom records: a continuous 3D coordinate and a discrete atom
  label.  This module keeps that state boundary explicit while reusing the
  generic BranchingFlows sampler and the scalar OU bridge from the DiffEq
  adapter module.
-/

namespace torch.branching

/-! ## 3D coordinates -/

structure Vec3 where
  x : Float
  y : Float
  z : Float
  deriving Repr, BEq, Inhabited

namespace Vec3

def zero : Vec3 := { x := 0.0, y := 0.0, z := 0.0 }

def map2 (f : Float → Float → Float) (a b : Vec3) : Vec3 :=
  { x := f a.x b.x, y := f a.y b.y, z := f a.z b.z }

def scale (c : Float) (v : Vec3) : Vec3 :=
  { x := c * v.x, y := c * v.y, z := c * v.z }

def add (a b : Vec3) : Vec3 :=
  map2 (fun x y => x + y) a b

def weightedAverage (a b : Vec3) (w1 w2 : Nat) : Vec3 :=
  let total := (w1 + w2).toFloat
  if total == 0.0 then
    a
  else
    scale (1.0 / total) (add (scale w1.toFloat a) (scale w2.toFloat b))

def ouBridge (cfg : OUBridgeConfig) (x0 anchor : Vec3) (t0 t : Float) : Vec3 :=
  { x := OUBridgeConfig.bridge cfg x0.x anchor.x t0 t,
    y := OUBridgeConfig.bridge cfg x0.y anchor.y t0 t,
    z := OUBridgeConfig.bridge cfg x0.z anchor.z t0 t }

def sampleOUBridge (cfg : OUBridgeConfig) (x0 anchor : Vec3) (t0 t : Float)
    (rng : Rng) : Vec3 × Rng :=
  let (x, rng) := OUBridgeConfig.sampleBridge cfg x0.x anchor.x t0 t rng
  let (y, rng) := OUBridgeConfig.sampleBridge cfg x0.y anchor.y t0 t rng
  let (z, rng) := OUBridgeConfig.sampleBridge cfg x0.z anchor.z t0 t rng
  ({ x, y, z }, rng)

end Vec3

/-! ## Atom records -/

structure MoleculeAtom where
  coord : Vec3
  label : Nat
  deriving Repr, BEq, Inhabited

structure MoleculeBridgeConfig where
  coordOU : OUBridgeConfig
  maskToken : Nat
  labelDFM : Option DistNoisyDiscreteConfig := none

namespace MoleculeBridgeConfig

/-- QM9 continuous/discrete base-process defaults from the BranchingFlows appendix. -/
def qm9 (vocabSize maskToken : Nat) : MoleculeBridgeConfig :=
  { coordOU := OUBridgeConfig.logLinearVariance 5.0 10.0 0.001,
    maskToken,
    labelDFM := some (DistNoisyDiscreteConfig.qm9 vocabSize maskToken) }

def maskedAtom (cfg : MoleculeBridgeConfig) (coord : Vec3 := Vec3.zero) : MoleculeAtom :=
  { coord, label := cfg.maskToken }

/--
Anchor merge for molecule atoms.

Coordinates use the continuous weighted average.  The discrete label is replaced
by the mask token, matching the Julia `canonical_anchor_merge` behavior for
discrete states and avoiding endpoint-label leakage through internal anchors.
-/
def anchorMerge (cfg : MoleculeBridgeConfig)
    (a b : MoleculeAtom) (w1 w2 : Nat) : MoleculeAtom :=
  { coord := Vec3.weightedAverage a.coord b.coord w1 w2,
    label := cfg.maskToken }

/--
Molecule atom bridge.

Coordinates use the endpoint-conditioned OU bridge.  Labels use the optional
Flowfusion-style DFM bridge when configured, otherwise they retain the source
label until the terminal endpoint.
-/
def bridge (cfg : MoleculeBridgeConfig)
    (x0 anchor : MoleculeAtom) (t0 t : Float) : MoleculeAtom :=
  let terminal := max cfg.coordOU.terminalTime t0
  let label :=
    if t >= terminal then
      anchor.label
    else
      match cfg.labelDFM with
      | some dfm => dfm.modeBridgeFrom x0.label anchor.label t0 t
      | none => x0.label
  { coord := Vec3.ouBridge cfg.coordOU x0.coord anchor.coord t0 t,
    label }

def sampleBridge (cfg : MoleculeBridgeConfig)
    (x0 anchor : MoleculeAtom) (t0 t : Float) (rng : Rng) : MoleculeAtom × Rng :=
  let terminal := max cfg.coordOU.terminalTime t0
  let (coord, rng) := Vec3.sampleOUBridge cfg.coordOU x0.coord anchor.coord t0 t rng
  if t >= terminal then
    ({ coord, label := anchor.label }, rng)
  else
    match cfg.labelDFM with
    | some dfm =>
        let (label, rng) := dfm.bridgeFrom x0.label anchor.label t0 t rng
        ({ coord, label }, rng)
    | none =>
        ({ coord, label := x0.label }, rng)

def sampleInitialAtom (cfg : MoleculeBridgeConfig) (rng : Rng) : MoleculeAtom × Rng :=
  let (x, rng) := randNormal rng
  let (y, rng) := randNormal rng
  let (z, rng) := randNormal rng
  ({ coord := { x, y, z }, label := cfg.maskToken }, rng)

private def modeOfLogits (logits : Array Float) (fallback : Nat) : Nat :=
  if logits.isEmpty then
    fallback
  else Id.run do
    let mut out := 0
    let mut best := logits[0]!
    for i in [:logits.size] do
      let x := logits[i]!
      if x > best then
        best := x
        out := i
    return out

def stepLabelFromLogits
    (cfg : MoleculeBridgeConfig)
    (currentLabel : Nat)
    (targetLogits : Array Float)
    (t0 t1 : Float) : Nat :=
  match cfg.labelDFM with
  | some dfm => dfm.stepLabelMode currentLabel targetLogits t0 t1
  | none => modeOfLogits targetLogits currentLabel

def stepAtomFromLogits
    (cfg : MoleculeBridgeConfig)
    (current : MoleculeAtom)
    (coordTarget : Vec3)
    (labelLogits : Array Float)
    (t0 t1 : Float) : MoleculeAtom :=
  { coord := Vec3.ouBridge cfg.coordOU current.coord coordTarget t0 t1,
    label := cfg.stepLabelFromLogits current.label labelLogits t0 t1 }

end MoleculeBridgeConfig

def maskDeletedMoleculeLabels
    (cfg : MoleculeBridgeConfig)
    (state : BranchingState MoleculeAtom) :
    BranchingState MoleculeAtom :=
  { state with
    state := state.state.mapIdx (fun i atom =>
      if state.del.getD i false then
        { atom with label := cfg.maskToken }
      else
        atom) }

structure MoleculeModelPrediction where
  coordTargets : Array Vec3
  labelLogits : Array (Array Float)
  splitLogits : Array Float
  delLogits : Array Float
  deriving Repr, Inhabited

namespace MoleculeModelPrediction

def toBranchingStepPrediction
    (prediction : MoleculeModelPrediction)
    (cfg : MoleculeBridgeConfig)
    (x : BranchingState MoleculeAtom)
    (s1 s2 : Float) : BranchingStepPrediction MoleculeAtom :=
  let targets := (Array.range x.state.size).map (fun i =>
    let current := x.state.getD i default
    let coordTarget := prediction.coordTargets.getD i current.coord
    let labelLogits := prediction.labelLogits.getD i #[]
    cfg.stepAtomFromLogits current coordTarget labelLogits s1 s2)
  { targets,
    splitLogits := prediction.splitLogits,
    delLogits := prediction.delLogits }

end MoleculeModelPrediction

/--
Base-step adapter for `MoleculeModelPrediction.toBranchingStepPrediction`.

The prediction already contains the one-step OU/DFM molecule state, so the
generic BranchingFlows forward step only needs to apply split/deletion events.
-/
def moleculePredictedBaseStep
    (_cfg : MoleculeBridgeConfig)
    (_x : BranchingState MoleculeAtom)
    (targets : Array MoleculeAtom)
    (_s1 _s2 : Float) : Array MoleculeAtom :=
  targets

def moleculeBranchingStep
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x : BranchingState MoleculeAtom)
    (prediction : MoleculeModelPrediction)
    (s1 s2 : Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := fun _ _ => true)
    (rng : Rng := { state := 0 }) :
    BranchingStepResult MoleculeAtom × Rng :=
  branchingStep moleculePredictedBaseStep flow x
    (prediction.toBranchingStepPrediction flow.base x s1 s2)
    s1 s2 splitAllowedAfterBaseStep rng

def moleculeBranchingGenerate
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x0 : BranchingState MoleculeAtom)
    (model : Float → BranchingState MoleculeAtom → MoleculeModelPrediction)
    (schedule : Array Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := fun _ _ => true)
    (rng : Rng := { state := 0 }) :
    BranchingGenerateResult MoleculeAtom × Rng := Id.run do
  if schedule.size <= 1 then
    return ({ finalState := x0, trajectory := #[x0], events := #[], times := schedule }, rng)
  let mut state := x0
  let mut trajectory : Array (BranchingState MoleculeAtom) := #[x0]
  let mut events : Array (Array BranchingStepEvent) := #[]
  let mut rng := rng
  for i in [:schedule.size - 1] do
    let s1 := schedule[i]!
    let s2 := schedule[i + 1]!
    let prediction := model s1 state
    let (result, rng') :=
      moleculeBranchingStep flow state prediction s1 s2 splitAllowedAfterBaseStep rng
    rng := rng'
    state := result.state
    trajectory := trajectory.push state
    events := events.push result.events
  ({ finalState := state, trajectory, events, times := schedule }, rng)

private def truncateBranchingState (limit : Nat)
    (state : BranchingState α) : BranchingState α :=
  let n := Nat.min limit state.state.size
  { state with
    state := state.state.extract 0 n
    groupings := state.groupings.extract 0 n
    del := state.del.extract 0 n
    ids := state.ids.extract 0 n
    branchmask := state.branchmask.extract 0 n
    flowmask := state.flowmask.extract 0 n
    padmask := state.padmask.extract 0 n }

def moleculeBranchingGenerateIO
    (flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom)
    (x0 : BranchingState MoleculeAtom)
    (model : Float → BranchingState MoleculeAtom → IO MoleculeModelPrediction)
    (schedule : Array Float)
    (splitAllowedAfterBaseStep : MoleculeAtom → MoleculeAtom → Bool := fun _ _ => true)
    (maxStateLen? : Option Nat := none)
    (rng : Rng := { state := 0 }) :
    IO (BranchingGenerateResult MoleculeAtom × Rng) := do
  if schedule.size <= 1 then
    return ({ finalState := x0, trajectory := #[x0], events := #[], times := schedule }, rng)
  let mut state := x0
  let mut trajectory : Array (BranchingState MoleculeAtom) := #[x0]
  let mut events : Array (Array BranchingStepEvent) := #[]
  let mut rng := rng
  for i in [:schedule.size - 1] do
    let s1 := schedule[i]!
    let s2 := schedule[i + 1]!
    let prediction ← model s1 state
    let (result, rng') :=
      moleculeBranchingStep flow state prediction s1 s2 splitAllowedAfterBaseStep rng
    rng := rng'
    state :=
      match maxStateLen? with
      | some limit => truncateBranchingState limit result.state
      | none => result.state
    trajectory := trajectory.push state
    events := events.push result.events
  return ({ finalState := state, trajectory, events, times := schedule }, rng)

end torch.branching
