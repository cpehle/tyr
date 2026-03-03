# Tyr Code Review Tracker

Last updated: 2026-03-03
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
- [x] `H15` Qwen3TTS decode no longer hard-fails on non-16 code groups; decode now auto-falls back to Python bridge in both offline and true-streaming paths.
- [x] `H16` Qwen3TTS speech-tokenizer variant support now runs Lean-native for 12Hz-family tokenizer configs with dynamic code-group counts (no Python encode fallback).
- [x] `H17` Qwen3TTS encode now supports separate Lean-native 25Hz tokenizer architecture via dedicated encoder module and bridge-time model-type dispatch.
- [x] `H24` ASR streaming prompt tokenization now uses append-only suffix caching to avoid full prompt re-encode on every decode hop.
- [x] `H25` ASR full-accumulation streaming now reuses cached audio-encoder projected prefixes and encodes only tail features when possible.
- [x] `H26` ASR streaming decode now supports generation from precomputed `inputs_embeds`, avoiding redundant embed construction in cache-aware paths.
- [x] `M24` Added a deterministic ASR streaming decode-cache benchmark regression test to measure prompt-cache-only vs full decode-cache performance.
- [x] `H27` Add incremental streaming Whisper-frontend cache so full-accumulation decode only computes new tail mel features each hop.
- [x] `H28` Replace full-sequence placeholder `masked_scatter` in streaming decode with cached placeholder-span writes to reduce O(seq*hidden) per-hop work.
- [x] `H29` Remove per-hop prompt-embed tensor diff check from prompt-cache reuse (`maxAll(abs(...))`) and rely on token-prefix/shape invariants.
- [x] `M25` Wrap ASR inference hot paths in `autograd.no_grad` to reduce autograd overhead and transient memory during decode.
- [ ] `M26` Add stream micro-batching scheduler for multi-session ASR throughput (shared decode batch across active sessions).
- [ ] `H30` Eliminate prompt-ID host→device rebuild each streaming hop by introducing device-resident prompt token buffers and append-only device assembly.
- [ ] `H31` Eliminate per-hop generated-token host pull/decode in hot path by supporting deferred text decode or token-level streaming buffers.

## Open Issues

### High
- [x] `H18` Add upstream-compatible ASR audio input normalization surface (URL/base64/typed in-memory + resample), not just local PCM16 WAV/16k-float entrypoints.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Frontend.lean`, `../Qwen3-ASR/qwen_asr/inference/utils.py`
- [x] `H19` Support upstream-style ASR timestamp flow with separate forced-aligner handle (`ASR + aligner`) instead of requiring a forced-aligner thinker checkpoint for transcription.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Model.lean`, `../Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`
- [x] `H20` Reconcile streaming decode semantics with upstream (full-accumulation chunk decode) or make behavior-selectable to reduce long-utterance parity drift.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `../Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`
- [x] `H21` Add true streaming incremental-cache decode path for ASR (audio/text cache reuse per hop) to avoid full recompute each streaming step.
  Refs: `Tyr/Model/Qwen3ASR/StreamModel.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/Model.lean`
- [x] `H22` Add native batch collation/inference path for offline ASR (batched prompt/audio tensors) instead of per-item decode loops.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Model.lean`
- [x] `H23` Reduce ASR host-device sync overhead in decode hot paths by keeping per-step EOS/finished-row handling on-device (no per-token `tensorToUInt64Array` round-trips in greedy loops).
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`, `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H24` Add append-only prompt-token cache for ASR streaming decode to avoid full prompt tokenization each hop.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H25` Add audio-encoder prefix cache reuse in ASR full-accumulation streaming decode to avoid re-encoding unchanged prefix frames.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H26` Add ASR decode path that accepts precomputed `inputs_embeds` and prompt-cache reuse to bypass redundant embed recomputation.
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H27` Add incremental streaming frontend cache that reuses prior waveform/mel overlap and appends only new feature frames per hop.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/Frontend.lean`
- [x] `H28` Replace streaming decode full-mask `masked_scatter` with cached placeholder span offsets + targeted slice writes for audio token insertion.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/Model.lean`
- [x] `H29` Eliminate prompt-cache reuse embed-prefix equality reduction (`maxAll(abs(...))`) from streaming path and gate reuse by prompt token prefix + shape only.
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`
- [ ] `H30` Remove per-hop prompt-ID CPU tensor materialization + `.to(device)` by caching device-side prompt token tensors and composing audio/suffix deltas on-device.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [ ] `H31` Remove per-hop generated-token host transfer (`tensorToUInt64Array`) in streaming decode hot path by introducing deferred decode/token-buffer API.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/Transcribe.lean`


### Medium
- [ ] `M16` Add higher-level Qwen3TTS mode API parity (`custom_voice`, `voice_design`, `voice_clone`) on top of current generic prompt-conditioning path.
  Refs: `Examples/Qwen3TTS/EndToEnd.lean`, `Tyr/Model/Qwen3TTS/Model.lean`
- [ ] `M17` Improve language parity for `Auto`/dialect handling to match upstream behavior (currently mostly token gating without dialect override logic).
  Refs: `Examples/Qwen3TTS/EndToEnd.lean`
- [x] `M18` Add standalone ASR forced-aligner wrapper parity (`from_pretrained` + `align(audio,text,language)` single/batch API) on top of existing low-level align helpers.
  Refs: `Tyr/Model/Qwen3ASR/ForcedAligner.lean`, `Tyr/Model/Qwen3ASR/Transcribe.lean`, `../Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`
- [x] `M19` Improve forced-alignment tokenization parity for JP/KR by adding language-specific segmentation strategy (upstream uses dedicated tokenizers).
  Refs: `Tyr/Model/Qwen3ASR/ForcedAligner.lean`, `../Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`
- [x] `M20` Reduce ASR prompt-template drift risk by deriving prompt construction from processor/chat-template semantics instead of hardcoded template strings.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `../Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`
- [x] `M21` Avoid duplicate ASR frontend extraction in separate-aligner timestamp flow by reusing decode frontend features when preprocessors are compatible.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `M22` Skip no-op audio resampling in ASR frontend/normalization paths when source sample rate already matches target sample rate.
  Refs: `Tyr/Model/Qwen3ASR/Frontend.lean`
- [x] `M23` Reduce long-audio chunk-boundary scan overhead in non-timestamp ASR by tightening silence-search window and supporting zero-window fast path.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `M25` Apply `autograd.no_grad` across ASR transcription/streaming inference entrypoints to avoid unnecessary graph tracking overhead.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`
- [ ] `M26` Add multi-session streaming micro-batching for ASR `StreamModel` so concurrent sessions share decoder batch steps.
  Refs: `Tyr/Model/Qwen3ASR/StreamModel.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`

### Low
- [ ] `L03` Expose direct speaker-name selection in CLI from loaded speaker maps.
  Refs: `Tyr/Model/Qwen3TTS/Config.lean`, `Examples/Qwen3TTS/EndToEnd.lean`
- [ ] `L04` Extend public demo/streaming entrypoints to ergonomic multi-sample batching (core tensors support batch, demo is mostly single-sample).
  Refs: `Examples/Qwen3TTS/EndToEnd.lean`, `Tyr/Model/Qwen3TTS/Streaming.lean`
- [x] `L05` Add ASR batch-throughput control parity (`max_inference_batch_size` style chunking knob) for large batch transcription/alignment workloads.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `../Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`
- [~] `L06` Remove URL/base64 temp-file materialization from ASR normalization by adding in-memory decode frontend path.
  Refs: `Tyr/Model/Qwen3ASR/Frontend.lean`


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
- [x] `H15` Qwen3TTS decode now supports non-16 talker code-group models via automatic Python-bridge fallback, including true-streaming post-decode fallback when Lean decode is unavailable.
  Refs: `Examples/Qwen3TTS/EndToEnd.lean`, `Tyr/Model/Qwen3TTS/Streaming.lean`
- [x] `H16` Qwen3TTS audio encode now supports 12Hz-family tokenizer variants natively in Lean by loading dynamic semantic/acoustic quantizer counts.
  Refs: `Tyr/Model/Qwen3TTS/SpeechTokenizerBridge.lean`, `Tyr/Model/Qwen3TTS/SpeechTokenizerEncoder.lean`
- [x] `H17` Qwen3TTS audio encode now supports tokenizer `model_type="qwen3_tts_tokenizer_25hz"` through a separate Lean-native encoder architecture (Whisper frontend + 25Hz transformer/VQ path) with bridge dispatch.
  Refs: `Tyr/Model/Qwen3TTS/SpeechTokenizer25HzEncoder.lean`, `Tyr/Model/Qwen3TTS/SpeechTokenizerBridge.lean`, `Tyr/Model/Qwen3TTS.lean`
- [x] `H18` Qwen3ASR now supports unified audio input normalization across local path, URL, base64 payload, and in-memory waveform+sample-rate entrypoints.
  Refs: `Tyr/Model/Qwen3ASR/Frontend.lean`, `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `H19` Qwen3ASR timestamp flow now accepts a separate forced-aligner handle and no longer requires forced-aligner thinker checkpoints for transcription.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `H20` Qwen3ASR streaming now supports selectable decode semantics (`rollingWindow` vs `fullAccumulation`) to reduce long-utterance parity drift.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/StreamModel.lean`
- [x] `M18` Added standalone `Qwen3ForcedAligner` parity API with `loadFromPretrained`, single/batch align entrypoints, and transcription integration.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `M19` Forced-alignment text encoding now applies language-specific segmentation for Japanese/Korean instead of generic fallback-only tokenization.
  Refs: `Tyr/Model/Qwen3ASR/ForcedAligner.lean`
- [x] `M20` ASR prompt building now routes through processor chat-template semantics to reduce hardcoded-template drift risk.
  Refs: `Tyr/Model/Qwen3ASR/Processor.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `L05` Added `maxInferenceBatchSize` chunking knobs across ASR transcription and forced-aligner batch entrypoints.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Examples/Qwen3ASR/Transcribe.lean`
- [x] `M21` Separate-aligner timestamp path now reuses ASR frontend features when preprocessor settings are compatible, avoiding a duplicate mel extraction pass.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `M22` ASR frontend/normalization now skips expensive resample calls when source and target sample rates already match.
  Refs: `Tyr/Model/Qwen3ASR/Frontend.lean`
- [x] `M23` Non-timestamp long-audio ASR now uses a smaller boundary-search window and supports a no-search cut fast path for chunk splitting.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `H21` Added true streaming incremental-cache decode plumbing for ASR prompt reuse across hops (cache-aware decode helpers + `StreamSession` cache persistence).
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/StreamModel.lean`, `Tests/TestQwen3ASR.lean`
- [x] `H22` Offline ASR now uses native bucketed batch collation/inference for single-chunk waveforms (`frames+seq` buckets), with fallback for non-collatable rows.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [x] `H23` Added batched token decode extraction path to reduce host sync frequency in offline ASR bucketed decode.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H23` Reduced scalar sync overhead by switching per-sample valid-frame extraction from `tensorToUInt64Array` to `nn.item` in both offline and streaming decode paths.
  Refs: `Tyr/Model/Qwen3ASR/Transcribe.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H23` ASR greedy decode loops now run EOS/finished-row masking fully on-device (cached + uncached + prompt-cache decode), removing per-step host token pulls in generation hot paths.
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`
- [x] `H24` Added append-only prompt tokenization cache (`StreamingPromptTokenCache`) so streaming decode can reuse encoded prompt prefixes and only tokenize suffix deltas.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H25` Added full-accumulation audio encoder prefix reuse cache (`StreamingAudioEncoderCache`) and tail-only audio projection updates in streaming decode.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H26` Added `generateGreedyFromInputsEmbedsWithPromptCache` and wired streaming decode to precompute/scatter audio embeddings once before generation.
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`, `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/StreamModel.lean`
- [x] `M24` Added `testQwen3ASRStreamingDecodeCacheBenchmark` to validate decode-cache parity and print measured speedup (`iters=20`, baseline `68ms`, optimized `66ms`, `1.03x`).
  Refs: `Tests/TestQwen3ASR.lean`
- [x] `H27` Added full-accumulation frontend-cache reuse in streaming decode: when audio grows append-only, recompute a suffix window and splice prior feature/mask prefixes instead of full-wave frontend recomputation.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H27` Added device-resident frontend cache reuse for full-accumulation streaming: when prefix frames are reusable, keep prior device prefix and transfer only suffix feature/mask frames.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H28` Streaming decode now replaces audio placeholders via contiguous token-span slice composition, removing full-sequence boolean-mask `masked_scatter` in the hot path.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`
- [x] `H29` Prompt-cache reuse now relies on prompt token-prefix and capacity invariants (removed per-hop embed-prefix tensor diff reduction).
  Refs: `Tyr/Model/Qwen3ASR/Model.lean`
- [x] `M25` Added `autograd.no_grad` guards around ASR streaming/offline generation and alignment inference calls.
  Refs: `Tyr/Model/Qwen3ASR/Streaming.lean`, `Tyr/Model/Qwen3ASR/Transcribe.lean`
- [~] `L06` Base64 ASR input path now decodes WAV directly in-memory (no temp file); URL path still uses temp-file materialization pending in-memory download path.
  Refs: `Tyr/Model/Qwen3ASR/Frontend.lean`

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
