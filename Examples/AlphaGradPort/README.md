# AlphaGrad Port Examples

This subdirectory contains staged AlphaGrad-style elimination planning ports in Tyr.

Current flow:
- `RoeFlux_1d` is the first concrete end-to-end target.
- Remaining benchmark names are targeted sequentially via the sweep runner.

Executables:
- `lake exe AlphaGradRoeFlux1dA0 [episodes]`
- `lake exe AlphaGradPortSweep`
- `lake exe AlphaGradPortSweep <episodes>`
- `lake exe AlphaGradPortSweep <task-name> [episodes]`
- `lake exe AlphaGradPolicyTrain ppo [task-name] [epochs] [episodes-per-epoch]`
- `lake exe AlphaGradPolicyTrain alphazero [task-name] [epochs] [episodes-per-epoch]`
- `lake exe AlphaGradPolicySweep [mode] [task-name|all] [epochs] [episodes-per-epoch]`

Task order used by the sweep:
1. `RoeFlux_1d`
2. `Perceptron`
3. `Encoder`
4. `RobotArm_6DOF`
5. `BlackScholes_Jacobian`
6. `HumanHeartDipole`
7. `PropaneCombustion`

Notes:
- `RoeFlux_1d` uses a hand-shaped graph inspired by Graphax dependencies.
- `Perceptron`, `Encoder`, `RobotArm_6DOF`, and `BlackScholes_Jacobian`
  are now lowered from real Tyr `KStmt` programs.
- `HumanHeartDipole` and `PropaneCombustion`
  are now lowered from real Tyr `KStmt` programs.
- All non-`RoeFlux_1d` tasks are lowered via:
  `buildAndExtractFromKStmts` + `registerKStmtAllSupportedSemanticsRules`
  (strict semantic local-Jac rules; no placeholder/hybrid fallback on this path).
- `A0Train`/`AlphaGradPortSweep` are planning/evaluation loops.
- `AlphaGradPolicyTrain ppo` is full network actor-critic PPO
  (clipped objective + value loss + entropy regularization).
- `AlphaGradPolicyTrain alphazero` is full network AlphaZero-style training
  (MCTS policy targets + value regression).
- `AlphaGradPolicySweep` runs PPO/AlphaZero/both across one task or all tasks
  with a single CLI entrypoint.
