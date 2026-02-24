/-
  Tyr/Optim/PolarExpress.lean

  Polar Express orthogonalization for Muon/NorMuon optimizers.
  Implements the Newton-Schulz iteration for matrix sign approximation.

  Based on modded-nanogpt's Polar Express implementation:
  - Uses precomputed coefficients for numerical stability
  - Supports batched and unbatched matrices
  - Handles tall vs wide matrices differently
-/
import Tyr.Torch
import Tyr.Distributed

/-!
# `Tyr.Optim.PolarExpress`

Optimizer submodule for Polar Express, used in training-time parameter updates.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace torch.Optim.PolarExpress

open torch

/-- Precomputed Newton-Schulz coefficients for 5 iterations.
    These are optimized for stability with safety_factor=2e-2, cushion=2.
    Each tuple is (a, b, c) for: Y = a*X + b*X@X.T@X + c*X@X.T@X@X.T@X -/
def newtonSchulzCoeffs : Array (Float × Float × Float) := #[
  (8.156554524902461, -22.48329292557795, 15.878769915207462),
  (4.042929935166739, -2.808917465908714, 0.5000178451051316),
  (3.241553795498743, -1.758787407092089, 0.35970315917498755),
  (2.866073605966538, -1.394706279118519, 0.3190251879692427),
  (2.6379268658498756, -1.1816678706889182, 0.2925389239509298)
]

/-- Configuration for Polar Express -/
structure Config where
  /-- Number of Newton-Schulz iterations (typically 5) -/
  numIters : UInt64 := 5
  /-- Whether to use split baddbmm for memory efficiency (advanced) -/
  splitBaddbmm : Bool := false
  deriving Repr, Inhabited

/-- Apply Polar Express orthogonalization to a gradient matrix.

    This approximates the matrix sign function via Newton-Schulz iterations,
    effectively orthogonalizing the gradient for more stable optimization.

    For a matrix G:
    1. Normalize: X = G / ||G||_F
    2. Iterate: X = a*X + b*X@X.T@X + c*X@X.T@X@X.T@X
    3. Return orthogonalized X

    Args:
    - G: Gradient matrix [batch?, m, n]
    - cfg: Configuration with numIters

    Returns: Orthogonalized gradient of same shape
-/
def apply {s : Shape} (G : T s) (cfg : Config := {}) : IO (T s) := do
  dist.polarExpress G cfg.numIters

/-- Muon-style orthogonalization that handles tall vs wide matrices.

    For [out, in] gradient:
    - If out > in: orthogonalize along rows (G.T is used, then result transposed back)
    - If out <= in: orthogonalize directly

    This ensures the orthogonalization is always applied to the "short" dimension.
-/
def muonOrthogonalize {s : Shape} (G : T s) (numIters : UInt64 := 5) : IO (T s) := do
  dist.muonOrthogonalize G numIters

/-- Apply Polar Express to a batch of gradient matrices in parallel.

    For NorMuon, gradients from multiple parameters are batched together
    for efficient orthogonalization on GPU.

    Args:
    - grads: Array of gradient matrices (must have same shape)
    - cfg: Configuration

    Returns: Array of orthogonalized gradients
-/
def applyBatch {s : Shape} (grads : Array (T s)) (cfg : Config := {}) : IO (Array (T s)) := do
  grads.mapM (apply · cfg)

/-- In-place orthogonalization (for memory efficiency).

    Note: This returns a new tensor but can be assigned back to the same variable.
    The original tensor is not modified in-place at the memory level.
-/
def applyInPlace {s : Shape} (G : T s) (cfg : Config := {}) : IO (T s) := do
  apply G cfg

/-- Compute the XXT product efficiently.

    This is used in Newton-Schulz iterations: C = A @ A.T
    Optimized for symmetric output.
-/
def computeXXT {s : Shape} (A : T s) : IO (T #[]) := do
  dist.xxt A

/-- Fused operation for Newton-Schulz: C = beta * A + alpha * (A @ A.T)

    More efficient than separate matmul and add operations.
-/
def fusedBaPlusCaa {s : Shape} (A : T s) (alpha beta : Float) : IO (T s) := do
  dist.baPlusCaa A alpha beta

end torch.Optim.PolarExpress
