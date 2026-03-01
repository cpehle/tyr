# F01 Evaluation: `lean_args` Adoption

Date: 2026-03-01
Decision: **Defer**

## Scope Reviewed
- Qwen3TTS CLI parsing in `Examples/Qwen3TTS/EndToEnd.lean`
- Current CLI parsing footprint across `Examples/` entrypoints

## Outcome
Defer migration to `lean_args` for the current pass.

## Rationale
- Existing Qwen3TTS CLI already enforces explicit unknown-option handling and value validation (`validateRawArgs` + typed checks), so immediate reliability gain from a parser swap is low for this path.
- CLI parsing is currently spread across multiple example entrypoints (at least 9 `parseArgs` definitions), so broad adoption is a non-trivial behavior-change surface and not a trivially safe refactor.
- Current priority was correctness/CI gating and targeted regression coverage; a parser framework migration is better handled as a dedicated follow-up with compatibility tests per CLI.

## Follow-up Trigger (Adopt Later)
Revisit adoption when doing a planned cross-CLI argument-surface cleanup, with explicit golden tests for unknown-flag, missing-value, and numeric parse failure behavior.
