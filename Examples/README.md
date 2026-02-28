# Examples

Runnable examples demonstrating Tyr's capabilities. Each example is a self-contained training or inference script.

Before running any example, set up the environment (see [Environment Setup](../README.md#environment-setup)) or use the Lake helper scripts where available.

## TrainGPT

Character-level GPT training on Shakespeare, matching [nanoGPT](https://github.com/karpathy/nanoGPT) configuration.

**Data:** `data/shakespeare_char/{train,val}.bin` (nanoGPT format). Falls back to random tokens if data is missing.

```bash
lake build TrainGPT
lake run train
# or manually: .lake/build/bin/TrainGPT
```

Configs available in `Examples/GPT/GPT.lean`: `nanogpt_cpu_shakespeare`, `gpt2_micro`, `gpt2_mini`, `gpt2_small`.

## TrainDiffusion

Discrete masked diffusion model on ASCII text. Supports animated terminal output during sampling, multi-block generation with overlap for continuity, and checkpoint save/load.

**Data:** `data/shakespeare_char/input.txt` (plain text, optional -- falls back to random tokens).

```bash
lake build TrainDiffusion
.lake/build/bin/TrainDiffusion                          # Train from scratch
.lake/build/bin/TrainDiffusion --generate               # Generate from checkpoint
.lake/build/bin/TrainDiffusion -g -p "ROMEO:" -t 0.7    # Generate with prompt
.lake/build/bin/TrainDiffusion -g -n 8                   # Generate 8 blocks
```

**CLI flags:**
- `--generate, -g [path]` -- Load checkpoint and generate (skip training)
- `--checkpoint, -c <path>` -- Checkpoint directory (default: `checkpoints/diffusion`)
- `--prompt, -p <text>` -- Prompt for generation
- `--blocks, -n <num>` -- Number of blocks to generate (default: 4)
- `--temperature, -t <val>` -- Sampling temperature (default: 0.9)

## TrainNanoChat

Modded-nanogpt style distributed GPT training with NorMuon + DistAdam optimizers, dynamic batch size / window size schedules, and multi-GPU support via `torchrun`.

**Data:** `data/fineweb10B/` (sharded binary), `data/fineweb_val/` (validation shards).

```bash
lake build TrainNanoChat

# Single GPU
.lake/build/bin/TrainNanoChat

# Multi-GPU
torchrun --nproc_per_node=8 .lake/build/bin/TrainNanoChat

# With options
.lake/build/bin/TrainNanoChat --data data/fineweb10B --val data/fineweb_val \
  --checkpoint-dir checkpoints/modded --debug
```

**CLI flags:**
- `--data <path>` -- Training data directory (default: `data/fineweb10B`)
- `--val <path>` -- Validation data directory (default: `data/fineweb_val`)
- `--checkpoint-dir <path>` -- Checkpoint save directory
- `--resume <path>` -- Resume from checkpoint
- `--debug` -- Run with small model for testing

## FluxDemo

End-to-end Flux Klein 4B image generation: text encoding (Qwen) -> diffusion -> VAE decoding.

**Weights:** Download Flux Klein 4B weights and place them at:
- `weights/flux.safetensors` -- Flux transformer weights
- `weights/ae.safetensors` -- VAE decoder weights
- `weights/flux-klein-4b/text_encoder/` -- Qwen text encoder (sharded)

```bash
lake build FluxDemo
.lake/build/bin/FluxDemo
```

**Output:** `output.ppm`

## Qwen3TTSEndToEnd

End-to-end Qwen3-TTS demo: Lean loads real talker/speaker weights, does Lean tokenization + generation, and uses thin Python bridge scripts only for speech-tokenizer/audio frontend tasks (decode/encode/mel prep).

**Model directory layout (local HuggingFace export):**
- `config.json`
- sharded `.safetensors` + index for `talker.*` and (base models) `speaker_encoder.*`
- `tokenizer.json` and `tokenizer_config.json`
- `speech_tokenizer/` subdirectory

```bash
lake build Qwen3TTSEndToEnd
lake exe Qwen3TTSEndToEnd \
  --model-dir weights/qwen3-tts \
  --text "Hello from Lean." \
  --wav-path output/qwen3tts.wav
```

Useful flags:
- `--skip-decode` to only emit codec tokens
- `--codes-path <path>` output codec token file
- `--decode-script scripts/qwen3tts_decode_codes.py` Python decoder bridge
- `--encode-audio-path <wav>` + `--encode-only` for audio -> codec IDs bridge
- `--ref-audio-path <wav>` to derive speaker embedding from reference audio
- `--speaker-mel-path <path>` intermediate speaker mel safetensors file
- `--encode-script scripts/qwen3tts_encode_audio.py` speech-tokenizer encoder bridge
- `--speaker-mel-script scripts/qwen3tts_prepare_speaker_mel.py` mel extraction bridge
- `--qwen-repo ../Qwen3-TTS` local repo fallback for importing `qwen_tts` if not installed as a package

Bridge scripts are launched with `uv run python` by default (`--python uv`).

## Qwen35RunHF

Run Qwen3.5 text generation from either a local model directory or a Hugging Face repo ID. If you pass a repo ID, Tyr resolves a local HF snapshot if present, otherwise downloads `config.json` and model safetensors to cache.

Supported collection coverage:
- The loader path is wired for all models in `https://huggingface.co/collections/Qwen/qwen35` (dense/MoE, base, FP8 variants) via the same `Qwen35` config+weight resolution path.

```bash
lake build Qwen35RunHF

# Small smoke-test model
lake exe Qwen35RunHF --source tiny-random/qwen3.5 --prompt "Hello from Lean."

# Official Qwen repo (large; requires enough VRAM/RAM)
lake exe Qwen35RunHF --source Qwen/Qwen3.5-4B --prompt "Summarize dependent types."

# Multimodal with Apple system media path (ImageIO/AVFoundation)
lake exe Qwen35RunHF --source Qwen/Qwen3.5-4B --multimodal \
  --image input.jpg --prompt "Describe this image."

# Multimodal video with temporal downsampling
lake exe Qwen35RunHF --source Qwen/Qwen3.5-4B --multimodal \
  --video clip.mp4 --video-max-frames 64 --video-frame-stride 4 \
  --prompt "Summarize this clip."

# Batched prompts + streaming
lake exe Qwen35RunHF --source tiny-random/qwen3.5 --prompt-file prompts.txt \
  --batch-size 4 --stream
```

Useful flags:
- `--revision <rev>` HF branch/tag/commit (default: `main`)
- `--cache-dir <path>` override model download cache directory
- `--prompt-file <path>` one prompt per line (batched decode input)
- `--batch-size <n>` prompts per decode batch
- `--max-new-tokens <n>` number of generated tokens
- `--stream` stream tokens as they are decoded
- `--multimodal` load `Qwen35ForConditionalGeneration`
- `--image <path>` image input for multimodal prefix-feature injection (Apple-only media path)
- `--video <path>` video input for multimodal prefix-feature injection (Apple-only media path)
- `--video-max-frames <n>` cap decoded video frames for preprocessing cost
- `--video-frame-stride <n>` keep every Nth decoded frame before patchification

## BranchingFlows

Combinatorial branching flow sampler -- a port of branching flow networks to Lean. Includes continuous, discrete, and mixed training demos. No external data needed.

```bash
lake build  # Part of Examples lib, no standalone executable yet
```

Demo files:
- `BranchingFlowsDemo.lean` -- Minimal deterministic demo
- `ContinuousTrainDemo.lean` -- Continuous state training
- `DiscreteTrainDemo.lean` -- Discrete state training
- `MixedTrainDemo.lean` -- Mixed continuous/discrete training

## NanoProof

Transformer architecture for theorem proving with dual policy/value heads for MCTS integration. Currently model-only; RL training loop depends on MCTS/Prover modules that need to be recreated.

Architecture features: rotary embeddings, RMSNorm, QK normalization, ReLU^2 activation, Group-Query Attention (GQA).
