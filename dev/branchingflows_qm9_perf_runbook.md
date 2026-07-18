# BranchingFlows QM9 — Tyr overhead runbook (spark-e626)

Goal: verify that Tyr's per-step training overhead for the QM9 paper profile is
**not massive** — i.e. step throughput on GPU is within the same ballpark as a
tuned loop for this model size (order 2× of expectation, not 10×). If overhead
is large, find and fix the top causes, then re-measure.

This document is both the procedure and the progress log. Newest log entries
at the bottom.

## Target workload

- Profile: `paper-qm9-main` (`scripts/branchingflows/qm9_paper.env.example`)
- Full architecture: hidden 384, 12 heads × 64, MLP 1536, RFF 64, 12 layers,
  6 coord-update layers (~25–40M params), batch 128, seq ≤ ~32
- Step: `trainStepMoleculeMuon` (Julia-compatible Muon path), four losses
  (coord MSE, label CE, split Poisson, deletion BCE) with flow time-weighting
- Expected per-step FLOPs ≈ 1 TFLOP → on GB10 the step is overhead-bound, so
  per-step wall time is the metric that matters

## Environment

- Host: `spark-e626` (SSH alias, BatchMode works from this machine)
- Arch: aarch64 (Grace CPU) + NVIDIA GB10 (Blackwell, SM120-class)
- Repo state on spark-e626: see log entries below
- Local machine: macOS (Apple Silicon), used for commits; smoke tests CPU-only

## Procedure

1. Verify + commit the in-flight Julia-parity Muon work locally (build
   `BranchingFlowsMoleculeTrainGenerate` + `test_runner_experimental`, run
   `--filter BranchingFlows`, run `--profile smoke` on CPU).
2. Sync the branch to spark-e626; build there (GB10 target; check
   `TYR_GPU_*` env and vendored CUDA libtorch on aarch64).
3. Baseline: time N steps of the paper profile on GPU (batch 128, full arch).
   Report steps/sec and ms/step; compare with the ~1 TFLOP/step expectation
   (10–100 ms/step is sane for an overhead-bound small model; > 1 s/step is
   "massive overhead" territory).
4. If overhead is massive: profile the step (device placement of every op,
   CPU↔GPU transfers, `nn.item`/host reads per step, Newton–Schulz cost,
   loss reduction on host vs device, allocator churn). Fix the top cause,
   re-measure, iterate. One change per iteration, numbers logged below.
5. Write the conclusion: measured steps/sec → projected 800k-run wall time
   and cost on this host.

## Progress log

### 2026-07-17 — setup

- spark-e626 reachable via SSH: aarch64, 1× NVIDIA GB10.
- Local verification build of the in-flight BranchingFlows diff started
  (targets: `BranchingFlowsMoleculeTrainGenerate`, `test_runner_experimental`).
