# Tyr Code Review Tracker

Last updated: 2026-03-01
Scope: broad static review across `Tyr/`, `cc/src/`, `Examples/`, `Tests/`, CI/workflows, hooks, and scripts.

Status legend:
- `[ ]` open
- `[~]` in progress
- `[x]` completed

## Working Set (Current)

- [x] `H01` Tokenizer serialization now keeps header/payload vocabulary counts aligned.
- [x] `H02` Token/special UTF-8 length narrowing now fails cleanly instead of silent `UInt16` overflow.
- [x] `H08` Qwen3ASR cached snapshot reuse now requires complete shard presence and re-runs weight completion.
- [x] `H09` Qwen3ASR audio placeholder mismatch now hard-fails silent drops and row/count mismatches.
- [x] `H10` Qwen3TTS `decodeChunked` now guards `chunkSize = 0`.
- [x] `H11` Required CI now typechecks/builds core `Examples` surfaces.
- [x] `H12` DataLoader tests now avoid vacuous pass patterns and emit explicit fixture-skip reasons.
- [x] `H13` Required CI now includes `TestsExperimental` and `TestQwen3TTS`.
- [x] `H14` Pre-push commit-subject validation now scopes to pushed ref updates and target remote tracking refs.
- [x] `H03` Apple audio input lifecycle/callback shared state now uses synchronized teardown and atomic guards.
- [x] `H05` High-frequency torch FFI paths now release owned Lean arguments consistently.
- [x] `H06` SafeTensors loader now validates metadata/data bounds and rejects malformed/truncated inputs explicitly.
- [x] `H07` High-risk torch FFI IO entrypoints now catch `c10::Error`/`std::exception` and return Lean IO errors.
- [x] `M01` Tokenizer deserialize no longer panics on malformed UTF-8.
- [x] `M02` Batched generation in `Qwen3`/`Qwen35` now stops per row and keeps finished rows pinned to EOS.
- [x] `M03` Decode loops no longer perform per-token host sync (`tensorToUInt64Array`) for EOS stopping checks.
- [x] `M04` `Qwen35` decode now precomputes rotary frequencies once per generation window.
- [x] `M05` Sampling now enforces `temperature > 0`.
- [x] `M06` Tokenizer BPE merge selection now uses precomputed merge priorities instead of repeated linear merge-list scans.
- [x] `M07` Lake/Bazel executable target drift is reconciled; parity check is bidirectional and currently clean.
- [x] `M08` Build config now avoids hard-coded macOS SDK/framework paths by resolving SDK roots dynamically.
- [x] `M09` Script env-var expansions are hardened for `set -u`.
- [x] `M10` GPU scripts now use portable CPU-count fallback instead of `nproc`-only.
- [x] `M11` Distributed init now rejects out-of-range `master_port` instead of truncating to `UInt16`.
- [x] `M12` WAV writing paths now reject RIFF-overflow payload sizes (>4GiB) before corruption.
- [x] `M13` Qwen3ASR tests no longer rely on gitignored fixture WAV paths.
- [x] `M15` CI commit-subject lint no longer truncates first-push checks to `~50`.
- [x] `M14` Optional Qwen3TTS regressions now use explicit CI gates with skip/fail semantics (no success masking via `continue-on-error`).
- [x] `F02` Qwen3TTS regression coverage expanded with instruct-path, zero-frame, and explicit error-path tests.
- [x] `F01` `lean_args` migration evaluated and deferred for now (see `dev/f01_lean_args_evaluation.md`).
- [x] `L01` TODO/FIXME and `sorry` lint scope expanded beyond `Tyr/` to include `Examples/` and `Tests/`.
- [x] `L02` Lake/Bazel parity checker now reports drift in both directions.

## Open Issues

### High


### Medium

### Low


## Completed (This Pass)

- [x] `H01` Tokenizer serialization can no longer emit mismatched header/payload vocab counts.
  Refs: `Tyr/Tokenizer/IO.lean`
- [x] `H02` Token/special UTF-8 lengths are now checked before `UInt16` encoding; invalid inputs return serialization errors.
  Refs: `Tyr/Tokenizer/IO.lean`
- [x] `H08` `Qwen3ASR` cached snapshot resolution now requires complete shard sets and repairs weights before reusing a cached snapshot.
  Refs: `Tyr/Model/Qwen3ASR/Pretrained.lean`
- [x] `H09` `Qwen3ASR` placeholder/audio alignment now hard-fails silent-drop and mismatch cases.
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`
- [x] `H10` `Qwen3TTS` `decodeChunked` now normalizes `chunkSize = 0` to `1` to avoid infinite loop.
  Refs: `Tyr/Model/Qwen3TTS/SpeechTokenizer.lean`
- [x] `H11` Required CI now builds/typechecks key `Examples` surfaces.
  Refs: `.github/workflows/ci.yml`
- [x] `H12` DataLoader tests now replace vacuous assertions and emit explicit fixture-skip messages.
  Refs: `Tests/TestDataLoader.lean`
- [x] `H13` Required CI now includes `test_runner_experimental` and `TestQwen3TTS` in the default suite.
  Refs: `.github/workflows/ci.yml`, `Tests.lean`
- [x] `H14` Pre-push hook now validates unpushed commits per pushed ref update and target remote scope.
  Refs: `.githooks/pre-push`
- [x] `H03` Apple audio input now synchronizes queue lifecycle and callback state to avoid race-prone teardown/re-enqueue behavior.
  Refs: `cc/src/apple_audio_input.mm`
- [x] `H05` Torch FFI ownership fixes now add missing `lean_dec` calls in hot paths and sharded loader plumbing.
  Refs: `cc/src/tyr.cpp`
- [x] `H06` SafeTensors loader now enforces strict metadata parsing, dtype validation, offset/size checks, and truncation detection.
  Refs: `cc/src/tyr.cpp`
- [x] `H07` High-risk `extern "C"` IO entrypoints now guard `c10::Error` and `std::exception` to prevent process termination.
  Refs: `cc/src/tyr.cpp`
- [x] `M01` Tokenizer deserialize now uses `String.fromUTF8?` and fails cleanly instead of panicking.
  Refs: `Tyr/Tokenizer/IO.lean`
- [x] `M02` Batched decode in `Qwen3` and `Qwen35` now tracks EOS per row and holds finished rows at EOS.
  Refs: `Tyr/Model/Qwen3/Model.lean`, `Tyr/Model/Qwen35/Model.lean`, `Tests/TestQwen3Model.lean`, `Tests/TestQwen35Model.lean`
- [x] `M03` `Qwen3` and `Qwen35` decode loops now keep EOS stop logic on-device without per-step host token sync.
  Refs: `Tyr/Model/Qwen3/Model.lean`, `Tyr/Model/Qwen35/Model.lean`, `Tests/TestQwen3Model.lean`, `Tests/TestQwen35Model.lean`
- [x] `M04` `Qwen35` decode path now precomputes rotary tables for the full generation window.
  Refs: `Tyr/Model/Qwen35/Model.lean`, `Tests/TestQwen35Model.lean`
- [x] `M05` Sampling now validates `temperature > 0`, with safe penalty fallback for invalid repetition-penalty values.
  Refs: `Tyr/Model/Qwen35/Model.lean`, `Tyr/Model/Qwen3TTS/Talker.lean`
- [x] `M06` Tokenizer BPE best-merge selection now uses merge-priority lookup maps, removing repeated full merge-list scans.
  Refs: `Tyr/Tokenizer/Types.lean`, `Tyr/Tokenizer/Encode.lean`, `Tyr/Tokenizer/Training.lean`, `Tyr/Tokenizer/IO.lean`, `Tests/TestNanoChatTokens.lean`
- [x] `M07` Lake/Bazel executable targets are reconciled so parity checks pass without drift.
  Refs: `BUILD.bazel`, `lakefile.lean`, `scripts/check_target_parity.sh`
- [x] `M08` macOS linker/SDK settings now use dynamic SDK resolution instead of hard-coded CLT/Xcode paths.
  Refs: `lakefile.lean`
- [x] `M09` Script env-var expansion now avoids unset-variable failures under `set -u`.
  Refs: `scripts/nanochat/run_pipeline_torchrun.sh`, `scripts/nanochat/run_train_torchrun.sh`, `scripts/nanochat/test_distributed_resume.sh`, `scripts/gpu/run_e2e_kernel.sh`, `scripts/gpu/bench_mha_h100_train.sh`
- [x] `M10` GPU scripts now use portable CPU-count detection (`nproc` / `getconf` / `sysctl` fallback).
  Refs: `scripts/gpu/run_e2e_kernel.sh`, `scripts/gpu/bench_mha_h100_train.sh`
- [x] `M11` Distributed init now rejects out-of-range `master_port` values instead of truncating.
  Refs: `cc/src/tyr_distributed.cpp`
- [x] `M12` WAV save/append/finalize now guard RIFF 32-bit data-size overflow before writing/patching headers.
  Refs: `cc/src/tyr.cpp`
- [x] `M13` Qwen3ASR WAV-dependent tests now generate deterministic temporary fixtures instead of relying on gitignored files.
  Refs: `Tests/TestQwen3ASR.lean`
- [x] `M15` Commit-subject lint now checks complete commit sets when base SHA is unavailable (no fixed `~50` truncation).
  Refs: `.github/workflows/ci.yml`
- [x] `L01` TODO/FIXME + `sorry` lint now scans `Tyr/`, `Examples/`, and `Tests/`.
  Refs: `.github/workflows/ci.yml`
- [x] `L02` Parity checker now reports both `Lake -> Bazel` and `Bazel -> Lake` mismatches.
  Refs: `scripts/check_target_parity.sh`

## Imported From `qwen_review_issues_2026-02-28`

### Feature Backlog

- [x] `F01` Evaluate switching CLI parsing to `lean_args`: deferred this pass; see `dev/f01_lean_args_evaluation.md`.
- [x] `F02` Expand Qwen3TTS regression test coverage beyond current suite.

### Historical Completed Items

- [x] `QR01` ASR streaming truncation in long sessions addressed.
- [x] `QR02` `StreamModel.pushAudio` unbounded ring growth addressed.
- [x] `QR03` Qwen3 `lm_head` loader fallback/exception handling addressed.
- [x] `QR04` Qwen3 cached decode RoPE recomputation overhead addressed.
- [x] `QR05` Qwen3 `generateGreedy` vs `generateGreedyUncached` empty-prompt behavior aligned.
- [x] `QR06` ASR batched EOS handling behavior addressed.
- [x] `QR07` ASR pretrained partial sharded-cache acceptance addressed (see `H08`).
- [x] `QR08` ASR placeholder/audio cardinality guard addressed (see `H09`).
- [x] `QR09` Qwen3TTS CLI `top-k` clamping/validation addressed.
- [x] `QR10` Qwen3TTS CLI unknown-flag / parse-error handling addressed.
- [x] `QR11` CI TODO/sorry checks now fail on matches (see `L01`).
- [x] `QR12` CI optional-step behavior corrected.
- [x] `QR13` Commit-subject range fragility on first-push path addressed.
- [x] `QR14` `matrix_log` imaginary-component handling addressed.
- [x] `QR15` `qr_reduced` release-build debug-check behavior addressed.
- [x] `QR16` `dependencies_macos.sh` safety/OS-guard behavior addressed.
- [x] `QR17` README `elan` install URL corrected.
- [x] `QR18` README CI commit-subject enforcement statement corrected.
- [x] `QR19` ASR live-mic float parsing validation tightened.
- [x] `QR20` Key regression tests added for long-stream/loader/args/EOS paths.
