/-
  Tyr/Optim/Scheduler.lean

  Learning rate schedulers following PyTorch patterns.
  All schedulers are implemented as pure functions: (step : Nat) → Float
  This enables easy resumption from checkpoints.
-/
namespace torch.Optim.Scheduler

private def pi : Float := 3.14159265358979323846

private def maxFloat (a b : Float) : Float := if a > b then a else b

/-! ## Cosine Annealing with Warmup

Standard cosine annealing schedule with optional linear warmup.
-/

/-- Configuration for cosine annealing schedule -/
structure CosineConfig where
  baseLr : Float        -- Peak learning rate
  minLr : Float         -- Minimum learning rate at end
  warmupSteps : Nat     -- Steps for linear warmup (0 = no warmup)
  totalSteps : Nat      -- Total training steps
  deriving Repr, Inhabited

/-- Cosine annealing with warmup.
    - Warmup: linear increase from 0 to baseLr
    - Annealing: cosine decay from baseLr to minLr -/
def cosineWithWarmup (cfg : CosineConfig) (step : Nat) : Float :=
  if step < cfg.warmupSteps then
    -- Linear warmup
    cfg.baseLr * (step.toFloat / maxFloat cfg.warmupSteps.toFloat 1.0)
  else if step >= cfg.totalSteps then
    cfg.minLr
  else
    -- Cosine annealing
    let progress := (step - cfg.warmupSteps).toFloat / maxFloat (cfg.totalSteps - cfg.warmupSteps).toFloat 1.0
    let coeff := 0.5 * (1.0 + Float.cos (pi * progress))
    cfg.minLr + coeff * (cfg.baseLr - cfg.minLr)

/-! ## Linear Schedule

Linear warmup followed by linear decay.
-/

/-- Configuration for linear schedule -/
structure LinearConfig where
  baseLr : Float
  minLr : Float
  warmupSteps : Nat
  totalSteps : Nat
  deriving Repr, Inhabited

/-- Linear warmup then linear decay -/
def linearWithWarmup (cfg : LinearConfig) (step : Nat) : Float :=
  if step < cfg.warmupSteps then
    -- Linear warmup
    cfg.baseLr * (step.toFloat / maxFloat cfg.warmupSteps.toFloat 1.0)
  else if step >= cfg.totalSteps then
    cfg.minLr
  else
    -- Linear decay
    let progress := (step - cfg.warmupSteps).toFloat / maxFloat (cfg.totalSteps - cfg.warmupSteps).toFloat 1.0
    cfg.baseLr - progress * (cfg.baseLr - cfg.minLr)

/-! ## Step Decay

Multiply learning rate by gamma every stepSize steps.
-/

/-- Configuration for step decay schedule -/
structure StepConfig where
  baseLr : Float
  gamma : Float := 0.1   -- Multiplicative factor
  stepSize : Nat := 30   -- Steps between decays
  deriving Repr, Inhabited

/-- Step decay: lr = baseLr * gamma^(step / stepSize) -/
def stepDecay (cfg : StepConfig) (step : Nat) : Float :=
  let numDecays := step / cfg.stepSize
  cfg.baseLr * Float.pow cfg.gamma numDecays.toFloat

/-! ## One Cycle Policy

Super-convergence via 1cycle learning rate policy (Smith & Topin, 2017).
- Phase 1 (30%): Linear increase from minLr to maxLr
- Phase 2 (70%): Cosine decrease from maxLr to minLr
-/

/-- Configuration for one-cycle policy -/
structure OneCycleConfig where
  maxLr : Float           -- Maximum learning rate (peak)
  minLr : Float           -- Starting and ending LR
  totalSteps : Nat
  pctStart : Float := 0.3 -- Fraction of steps for warmup phase
  deriving Repr, Inhabited

/-- One-cycle learning rate policy -/
def oneCycle (cfg : OneCycleConfig) (step : Nat) : Float :=
  if step >= cfg.totalSteps then
    cfg.minLr
  else
    let warmupSteps := (cfg.pctStart * cfg.totalSteps.toFloat).toUInt64.toNat
    if step < warmupSteps then
      -- Phase 1: Linear warmup to maxLr
      let progress := step.toFloat / maxFloat warmupSteps.toFloat 1.0
      cfg.minLr + progress * (cfg.maxLr - cfg.minLr)
    else
      -- Phase 2: Cosine decay to minLr
      let progress := (step - warmupSteps).toFloat / maxFloat (cfg.totalSteps - warmupSteps).toFloat 1.0
      let coeff := 0.5 * (1.0 + Float.cos (pi * progress))
      cfg.minLr + coeff * (cfg.maxLr - cfg.minLr)

/-! ## Warmup + Plateau + Cosine Decay

Common pattern: warmup → constant → cosine decay.
-/

/-- Configuration for warmup-plateau-decay schedule -/
structure WarmupPlateauConfig where
  baseLr : Float
  minLr : Float
  warmupSteps : Nat
  plateauSteps : Nat    -- Steps at constant baseLr after warmup
  totalSteps : Nat
  deriving Repr, Inhabited

/-- Warmup → Plateau → Cosine decay -/
def warmupPlateauCosine (cfg : WarmupPlateauConfig) (step : Nat) : Float :=
  let decayStart := cfg.warmupSteps + cfg.plateauSteps
  if step < cfg.warmupSteps then
    -- Warmup phase
    cfg.baseLr * (step.toFloat / maxFloat cfg.warmupSteps.toFloat 1.0)
  else if step < decayStart then
    -- Plateau phase
    cfg.baseLr
  else if step >= cfg.totalSteps then
    cfg.minLr
  else
    -- Cosine decay phase
    let progress := (step - decayStart).toFloat / maxFloat (cfg.totalSteps - decayStart).toFloat 1.0
    let coeff := 0.5 * (1.0 + Float.cos (pi * progress))
    cfg.minLr + coeff * (cfg.baseLr - cfg.minLr)

/-! ## Exponential Decay -/

/-- Configuration for exponential decay -/
structure ExponentialConfig where
  baseLr : Float
  gamma : Float := 0.99  -- Decay factor per step
  deriving Repr, Inhabited

/-- Exponential decay: lr = baseLr * gamma^step -/
def exponentialDecay (cfg : ExponentialConfig) (step : Nat) : Float :=
  cfg.baseLr * Float.pow cfg.gamma step.toFloat

/-! ## Polynomial Decay -/

/-- Configuration for polynomial decay -/
structure PolynomialConfig where
  baseLr : Float
  minLr : Float
  totalSteps : Nat
  power : Float := 1.0  -- 1.0 = linear, 2.0 = quadratic, etc.
  deriving Repr, Inhabited

/-- Polynomial decay: lr = (baseLr - minLr) * (1 - step/total)^power + minLr -/
def polynomialDecay (cfg : PolynomialConfig) (step : Nat) : Float :=
  if step >= cfg.totalSteps then
    cfg.minLr
  else
    let progress := step.toFloat / cfg.totalSteps.toFloat
    let decay := Float.pow (1.0 - progress) cfg.power
    cfg.minLr + decay * (cfg.baseLr - cfg.minLr)

/-! ## Constant Schedule -/

/-- Constant learning rate (useful as baseline) -/
def constant (lr : Float) (_step : Nat) : Float := lr

/-! ## Weight Decay Scheduling

nanochat uses linear decay to zero: wd = base_wd * (1 - step/total_steps)
-/

/-- Configuration for weight decay schedule -/
structure WeightDecayConfig where
  baseWd : Float        -- Base weight decay value
  totalSteps : Nat      -- Total training steps
  deriving Repr, Inhabited

/-- Linear weight decay: decays from baseWd to 0 over training.
    wd(step) = baseWd * (1 - step/totalSteps)
    Following nanochat's approach where weight decay goes to zero. -/
def linearWeightDecay (cfg : WeightDecayConfig) (step : Nat) : Float :=
  if step >= cfg.totalSteps then
    0.0
  else
    cfg.baseWd * (1.0 - step.toFloat / cfg.totalSteps.toFloat)

/-- Cosine weight decay: smooth decay from baseWd to 0.
    wd(step) = baseWd * 0.5 * (1 + cos(π * step/totalSteps)) -/
def cosineWeightDecay (cfg : WeightDecayConfig) (step : Nat) : Float :=
  if step >= cfg.totalSteps then
    0.0
  else
    let progress := step.toFloat / cfg.totalSteps.toFloat
    cfg.baseWd * 0.5 * (1.0 + Float.cos (pi * progress))

/-- Constant weight decay (no scheduling) -/
def constantWeightDecay (wd : Float) (_step : Nat) : Float := wd

/-! ## Helper: Create schedule from config -/

/-- Union type for schedule configurations -/
inductive ScheduleConfig where
  | cosine : CosineConfig → ScheduleConfig
  | linear : LinearConfig → ScheduleConfig
  | step : StepConfig → ScheduleConfig
  | oneCycle : OneCycleConfig → ScheduleConfig
  | warmupPlateau : WarmupPlateauConfig → ScheduleConfig
  | exponential : ExponentialConfig → ScheduleConfig
  | polynomial : PolynomialConfig → ScheduleConfig
  | const : Float → ScheduleConfig
  deriving Repr

/-- Get learning rate for any schedule type -/
def getLr (cfg : ScheduleConfig) (step : Nat) : Float :=
  match cfg with
  | .cosine c => cosineWithWarmup c step
  | .linear c => linearWithWarmup c step
  | .step c => stepDecay c step
  | .oneCycle c => oneCycle c step
  | .warmupPlateau c => warmupPlateauCosine c step
  | .exponential c => exponentialDecay c step
  | .polynomial c => polynomialDecay c step
  | .const lr => lr

end torch.Optim.Scheduler
