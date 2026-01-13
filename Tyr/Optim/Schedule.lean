/-
  Tyr/Optim/Schedule.lean

  Composable training schedule typeclass and implementations.

  Replaces hard-coded if/else schedules with a generic Schedule typeclass.
  Supports:
  - PiecewiseConstant: step-function schedules
  - Linear: linear interpolation between values
  - Cosine: cosine annealing
  - Composed: combine multiple schedules

  Example usage:
  ```
  -- Old: hard-coded
  def getBatchSize (step : UInt64) : UInt64 :=
    if step < 200 then 8 else if step < 1000 then 16 else 24

  -- New: declarative
  def batchSizeSchedule : PiecewiseConstant UInt64 :=
    { breakpoints := #[(200, 8), (1000, 16)], default := 24 }

  let batchSize := Schedule.valueAt batchSizeSchedule step
  ```
-/
namespace torch.Optim.Schedule

/-! ## Schedule Typeclass -/

/-- Typeclass for training schedules.
    Maps step number to a value of type α. -/
class Schedule (S : Type) (α : Type) where
  /-- Get the value at a given step -/
  valueAt : S → Nat → α

/-! ## PiecewiseConstant Schedule -/

/-- A piecewise constant schedule (step function).
    Value changes at specific breakpoints.

    Example: batch size ramp-up
    - steps 0-199: 8
    - steps 200-999: 16
    - steps 1000+: 24

    Represented as:
    { breakpoints := #[(200, 16), (1000, 24)], initialValue := 8 }
-/
structure PiecewiseConstant (α : Type) where
  /-- Breakpoints as (step, newValue) pairs, sorted by step -/
  breakpoints : Array (Nat × α)
  /-- Initial value (before first breakpoint) -/
  initialValue : α
  deriving Repr

/-- Get value at step for piecewise constant schedule -/
def PiecewiseConstant.getValue (s : PiecewiseConstant α) (step : Nat) : α := Id.run do
  let mut value := s.initialValue
  for (breakpoint, newValue) in s.breakpoints do
    if step >= breakpoint then
      value := newValue
    else
      break  -- Breakpoints are sorted, no need to check further
  return value

instance {α : Type} : Schedule (PiecewiseConstant α) α where
  valueAt := PiecewiseConstant.getValue

/-! ## Linear Schedule -/

/-- Linear interpolation schedule.
    Linearly interpolates between start and end values over a range. -/
structure Linear where
  /-- Starting step -/
  startStep : Nat
  /-- Ending step -/
  endStep : Nat
  /-- Value at start -/
  startValue : Float
  /-- Value at end -/
  endValue : Float
  deriving Repr

/-- Get value at step for linear schedule -/
def Linear.getValue (s : Linear) (step : Nat) : Float :=
  if step <= s.startStep then
    s.startValue
  else if step >= s.endStep then
    s.endValue
  else
    let progress := (step - s.startStep).toFloat / (s.endStep - s.startStep).toFloat
    s.startValue + progress * (s.endValue - s.startValue)

instance : Schedule Linear Float where
  valueAt := Linear.getValue

/-! ## Cosine Annealing Schedule -/

/-- Cosine annealing schedule with optional warmup.
    Common for learning rate decay. -/
structure CosineAnnealing where
  /-- Total number of steps -/
  totalSteps : Nat
  /-- Maximum value (after warmup) -/
  maxValue : Float
  /-- Minimum value (at end) -/
  minValue : Float
  /-- Warmup steps (linear from 0 to max) -/
  warmupSteps : Nat := 0
  deriving Repr

/-- Get value at step for cosine annealing schedule -/
def CosineAnnealing.getValue (s : CosineAnnealing) (step : Nat) : Float :=
  if step < s.warmupSteps then
    -- Linear warmup
    s.maxValue * step.toFloat / s.warmupSteps.toFloat
  else if step >= s.totalSteps then
    s.minValue
  else
    -- Cosine decay
    let decaySteps := s.totalSteps - s.warmupSteps
    let progress := (step - s.warmupSteps).toFloat / decaySteps.toFloat
    let pi : Float := 3.14159265358979323846
    s.minValue + 0.5 * (s.maxValue - s.minValue) * (1.0 + Float.cos (pi * progress))

instance : Schedule CosineAnnealing Float where
  valueAt := CosineAnnealing.getValue

/-! ## Warmup + Plateau + Cooldown (WPC) Schedule -/

/-- Warmup-Plateau-Cooldown schedule (common in LLM training).
    - Warmup: linear from 0 to peak
    - Plateau: constant at peak
    - Cooldown: cosine decay to min -/
structure WarmupPlateauCooldown where
  /-- Peak learning rate -/
  peakLr : Float
  /-- Minimum learning rate -/
  minLr : Float
  /-- Warmup duration (steps) -/
  warmupSteps : Nat
  /-- Total steps before cooldown starts (includes warmup) -/
  plateauEnd : Nat
  /-- Total training steps -/
  totalSteps : Nat
  deriving Repr

/-- Get value at step for WPC schedule -/
def WarmupPlateauCooldown.getValue (s : WarmupPlateauCooldown) (step : Nat) : Float :=
  if step < s.warmupSteps then
    -- Linear warmup
    s.peakLr * step.toFloat / s.warmupSteps.toFloat
  else if step < s.plateauEnd then
    -- Plateau
    s.peakLr
  else if step >= s.totalSteps then
    s.minLr
  else
    -- Cosine cooldown
    let cooldownSteps := s.totalSteps - s.plateauEnd
    let progress := (step - s.plateauEnd).toFloat / cooldownSteps.toFloat
    let pi : Float := 3.14159265358979323846
    s.minLr + 0.5 * (s.peakLr - s.minLr) * (1.0 + Float.cos (pi * progress))

instance : Schedule WarmupPlateauCooldown Float where
  valueAt := WarmupPlateauCooldown.getValue

/-! ## Momentum Schedule -/

/-- Momentum schedule with warmup and cooldown.
    Common for Muon optimizer. -/
structure MomentumSchedule where
  /-- Base momentum (plateau value) -/
  baseMomentum : Float := 0.95
  /-- Minimum momentum (warmup start, cooldown end) -/
  minMomentum : Float := 0.85
  /-- Warmup steps -/
  warmupSteps : Nat := 300
  /-- Cooldown steps (before end) -/
  cooldownSteps : Nat := 50
  /-- Total steps -/
  totalSteps : Nat
  deriving Repr

/-- Get momentum at step -/
def MomentumSchedule.getValue (s : MomentumSchedule) (step : Nat) : Float :=
  if step < s.warmupSteps then
    -- Linear warmup
    let progress := step.toFloat / s.warmupSteps.toFloat
    s.minMomentum + progress * (s.baseMomentum - s.minMomentum)
  else if step > s.totalSteps - s.cooldownSteps then
    -- Linear cooldown
    let stepsFromEnd := s.totalSteps - step
    let progress := stepsFromEnd.toFloat / s.cooldownSteps.toFloat
    s.minMomentum + progress * (s.baseMomentum - s.minMomentum)
  else
    s.baseMomentum

instance : Schedule MomentumSchedule Float where
  valueAt := MomentumSchedule.getValue

/-! ## Composed Schedule -/

/-- A schedule that applies a function to another schedule's output -/
structure Mapped (S : Type) (α β : Type) where
  /-- Inner schedule -/
  inner : S
  /-- Mapping function -/
  f : α → β

instance [Schedule S α] : Schedule (Mapped S α β) β where
  valueAt s step := s.f (Schedule.valueAt s.inner step)

/-- Map a function over a schedule -/
def map [Schedule S α] (f : α → β) (s : S) : Mapped S α β :=
  { inner := s, f }

/-! ## Window Size Schedule (Pair) -/

/-- Window size schedule returning (shortWindow, longWindow) pairs -/
structure WindowSchedule where
  /-- Breakpoints: (step, (shortBlocks, longBlocks)) -/
  breakpoints : Array (Nat × (Nat × Nat))
  /-- Initial value -/
  initialValue : (Nat × Nat) := (3, 3)
  deriving Repr

/-- Get window sizes at step -/
def WindowSchedule.getValue (s : WindowSchedule) (step : Nat) : (Nat × Nat) := Id.run do
  let mut value := s.initialValue
  for (breakpoint, newValue) in s.breakpoints do
    if step >= breakpoint then
      value := newValue
    else
      break
  return value

instance : Schedule WindowSchedule (Nat × Nat) where
  valueAt := WindowSchedule.getValue

/-! ## Pre-defined Schedules -/

/-- Default batch size schedule (modded-nanogpt style) -/
def defaultBatchSchedule : PiecewiseConstant UInt64 :=
  { breakpoints := #[(200, 16), (1000, 24)], initialValue := 8 }

/-- Default window size schedule (modded-nanogpt style) -/
def defaultWindowSchedule : WindowSchedule :=
  { breakpoints := #[(200, (3, 7)), (1000, (3, 11))], initialValue := (3, 3) }

/-- Default learning rate schedule for 2090 steps -/
def defaultLrSchedule : WarmupPlateauCooldown :=
  { peakLr := 0.023
    minLr := 0.0023
    warmupSteps := 300
    plateauEnd := 940  -- 55% of 2090 ≈ 1150, minus warmup
    totalSteps := 2090
  }

/-- Default momentum schedule -/
def defaultMomentumSchedule (totalSteps : Nat) : MomentumSchedule :=
  { baseMomentum := 0.95
    minMomentum := 0.85
    warmupSteps := 300
    cooldownSteps := 50
    totalSteps
  }

/-! ## Utility Functions -/

/-- Convert block count to sequence length -/
def blocksToSeqLen (blocks : Nat) (blockSize : Nat := 128) : UInt64 :=
  (blocks * blockSize).toUInt64

/-- Get sequence length from window schedule -/
def getSeqLenFromWindow (ws : WindowSchedule) (step : Nat) (blockSize : Nat := 128) : UInt64 :=
  let pair := WindowSchedule.getValue ws step
  blocksToSeqLen pair.2 blockSize

end torch.Optim.Schedule
