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
- [x] `M01` Tokenizer deserialize no longer panics on malformed UTF-8.
- [x] `M05` Sampling now enforces `temperature > 0`.
- [~] `M07` Parity checker is now bidirectional; concrete Lake/Bazel target drift remains to be reconciled.
- [x] `M09` Script env-var expansions are hardened for `set -u`.
- [x] `M10` GPU scripts now use portable CPU-count fallback instead of `nproc`-only.
- [x] `M11` Distributed init now rejects out-of-range `master_port` instead of truncating to `UInt16`.
- [x] `M12` WAV writing paths now reject RIFF-overflow payload sizes (>4GiB) before corruption.
- [x] `M13` Qwen3ASR tests no longer rely on gitignored fixture WAV paths.
- [x] `M15` CI commit-subject lint no longer truncates first-push checks to `~50`.
- [x] `L01` TODO/FIXME and `sorry` lint scope expanded beyond `Tyr/` to include `Examples/` and `Tests/`.
- [x] `L02` Lake/Bazel parity checker now reports drift in both directions.

## Open Issues

### High

- [ ] `H03` Audio capture runtime has race-prone shared state (`g_running`/queue lifecycle), risking UB/corruption.
  Refs: `cc/src/apple_audio_input.mm:17`, `cc/src/apple_audio_input.mm:46`, `cc/src/apple_audio_input.mm:53`, `cc/src/apple_audio_input.mm:131`
- [ ] `H05` Multiple FFI entrypoints appear to leak owned `lean_obj_arg` values due missing `lean_dec`/borrow mismatch in hot paths.
  Refs: `cc/src/tyr.cpp:779`, `cc/src/tyr.cpp:780`, `cc/src/tyr.cpp:781`, `cc/src/tyr.cpp:866`, `cc/src/tyr.cpp:879`, `cc/src/tyr.cpp:901`, `cc/src/tyr.cpp:3663`
- [ ] `H06` SafeTensors loader path can accept malformed/truncated inputs and silently decode wrong dtype/data.
  Refs: `cc/src/tyr.cpp:2651`, `cc/src/tyr.cpp:2895`, `cc/src/tyr.cpp:2905`, `cc/src/tyr.cpp:2948`, `cc/src/tyr.cpp:2958`, `cc/src/tyr.cpp:2962`
- [ ] `H07` Many `extern "C"` FFI entrypoints lack exception boundaries; uncaught `c10::Error` can terminate process.
  Refs: `cc/src/tyr.cpp:3406`, `cc/src/tyr.cpp:3918`, `cc/src/tyr.cpp:3928`

### Medium

- [ ] `M02` Batched generation in `Qwen3`/`Qwen35` still uses all-rows EOS stopping semantics; finished rows continue decoding.
  Refs: `Tyr/Model/Qwen3/Model.lean:27`, `Tyr/Model/Qwen3/Model.lean:199`, `Tyr/Model/Qwen35/Model.lean:31`, `Tyr/Model/Qwen35/Model.lean:1107`
- [ ] `M03` Per-token host synchronization in decode loops (`tensorToUInt64Array`) is a major perf drag, especially on GPU.
  Refs: `Tyr/Model/Qwen3/Model.lean:195`, `Tyr/Model/Qwen3/Model.lean:225`, `Tyr/Model/Qwen35/Model.lean:1103`, `Tyr/Model/Qwen35/Model.lean:1135`
- [ ] `M04` `Qwen35` decode path recomputes rotary frequencies each token (`O(T^2)` style overhead).
  Refs: `Tyr/Model/Qwen35/Model.lean:1018`, `Tyr/Model/Qwen35/Model.lean:1019`, `Tyr/Model/Qwen35/Model.lean:1020`
- [ ] `M06` Tokenizer BPE merge search is algorithmically expensive due repeated linear scans across merges.
  Refs: `Tyr/Tokenizer/Encode.lean:89`, `Tyr/Tokenizer/Encode.lean:97`, `Tyr/Tokenizer/Encode.lean:111`, `Tyr/Tokenizer/Encode.lean:140`
- [ ] `M07` Lake/Bazel target drift still exists after checker hardening; reconcile listed target mismatches.
  Refs: `scripts/check_target_parity.sh:28`, `lakefile.lean:226`, `lakefile.lean:232`, `lakefile.lean:256`, `lakefile.lean:262`, `lakefile.lean:268`, `lakefile.lean:274`, `BUILD.bazel:217`
- [ ] `M08` Build config uses hard-coded linker/SDK paths that reduce portability across hosts/toolchains.
  Refs: `lakefile.lean:7`, `lakefile.lean:23`, `lakefile.lean:25`, `lakefile.lean:27`, `lakefile.lean:47`, `lakefile.lean:58`
- [ ] `M14` Optional regression steps in CI are effectively false-green (`continue-on-error` and skip-if-missing semantics).
  Refs: `.github/workflows/ci.yml:109`, `.github/workflows/ci.yml:111`, `.github/workflows/ci.yml:114`, `scripts/qwen3tts_parity_regression.sh:22`, `scripts/qwen3tts_asr_regression.sh:16`

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
- [x] `M01` Tokenizer deserialize now uses `String.fromUTF8?` and fails cleanly instead of panicking.
  Refs: `Tyr/Tokenizer/IO.lean`
- [x] `M05` Sampling now validates `temperature > 0`, with safe penalty fallback for invalid repetition-penalty values.
  Refs: `Tyr/Model/Qwen35/Model.lean`, `Tyr/Model/Qwen3TTS/Talker.lean`
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

- [ ] `F01` Evaluate switching CLI parsing to `lean_args`: <https://github.com/cpehle/lean_args>
- [ ] `F02` Expand Qwen3TTS regression test coverage beyond current suite.

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
