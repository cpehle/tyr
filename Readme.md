# Tyr

A dependently-typed deep learning framework for Lean 4, providing compile-time tensor shape verification. Very much WIP.

## Overview

Tyr uses Lean 4's dependent type system to catch tensor dimension mismatches at compile time, not runtime. This eliminates a major source of bugs while providing access to optimized tensor operations.

```lean
-- Shapes are tracked in the type system
def linear {m n b : UInt64} (x : T #[b, m]) (M : T #[n, m]) : T #[b, n] := ...

-- Mismatched dimensions fail at compile time, not runtime!
let x : T #[32, 768] := ...
let w : T #[768, 512] := ...
let y := linear x w  -- Error: expected T #[n, 768], got T #[768, 512]
```

## Dependencies

### Lean 4

Install [elan](https://github.com/leanprover/elan) (the Lean version manager):

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
```

The correct Lean nightly is pinned in `lean-toolchain` and will be installed
automatically on first `lake build`.

### LibTorch (required)

CI currently pins `LIBTORCH_VERSION=2.7.1`. Use the same version locally for reproducible builds.

**macOS:**
```bash
LIBTORCH_VERSION=2.7.1
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  LIBTORCH_PKG="libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
else
  LIBTORCH_PKG="libtorch-macos-x86_64-${LIBTORCH_VERSION}.zip"
fi
cd external
curl -LO "https://download.pytorch.org/libtorch/cpu/${LIBTORCH_PKG}"
unzip "${LIBTORCH_PKG}" && rm "${LIBTORCH_PKG}"
cd ..
```

Or run the helper script:
```bash
bash dependencies_macos.sh
```

**Linux (CPU):**
```bash
LIBTORCH_VERSION=2.7.1
cd external
curl -L "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip" -o libtorch.zip
unzip libtorch.zip && rm libtorch.zip
cd ..
```

**Linux (CUDA 12.6):**
```bash
LIBTORCH_VERSION=2.7.1
cd external
curl -L "https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu126.zip" -o libtorch.zip
unzip libtorch.zip && rm libtorch.zip
cd ..
```

### OpenMP (required)

**macOS (Homebrew):**
```bash
brew install libomp
```

**Linux:**
OpenMP is typically included with GCC. Install if needed:
```bash
sudo apt install libomp-dev   # Debian/Ubuntu
```

### Apache Arrow & Parquet (required for data loading)

**macOS:**
```bash
brew install apache-arrow
```

**Linux:**
```bash
sudo apt install libarrow-dev libparquet-dev
```

### C++17 Compiler (required)

- macOS: Xcode command line tools (`xcode-select --install`)
- Linux: GCC 9+ or Clang 10+ (`sudo apt install build-essential`)

## Quick Start

### Building

```bash
# Build all targets with Lake
lake build

# Build specific executables
lake build test_runner
lake build TrainGPT
lake build TrainDiffusion
lake build TrainNanoChat
lake build FluxDemo
```

### Building with Bazel

```bash
# Build all Bazel targets
bazel build //...

# Keep Lake/Bazel executable targets in sync
./scripts/check_target_parity.sh
```

### Environment Setup

All executables need the native library paths set at runtime:

**macOS (Apple Silicon):**
```bash
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib:/opt/homebrew/lib
```

**macOS (Intel):**
```bash
export DYLD_LIBRARY_PATH=external/libtorch/lib:/usr/local/opt/libomp/lib:/usr/local/lib
```

**Linux:**
```bash
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
```

Or use the Lake helper scripts which set these automatically:
```bash
lake run           # runs test_runner
lake run train     # runs TrainGPT
```

### Running Tests

```bash
lake build test_runner
.lake/build/bin/test_runner

# Or use the helper script
lake run

# Experimental/in-progress suites
lake build test_runner_experimental
.lake/build/bin/test_runner_experimental
```

## Examples

See [Examples/README.md](Examples/README.md) for detailed per-example documentation.

| Example | Description | Build target |
|---------|-------------|--------------|
| **TrainGPT** | Character-level GPT on Shakespeare | `lake build TrainGPT` |
| **TrainDiffusion** | Discrete masked diffusion on ASCII text | `lake build TrainDiffusion` |
| **TrainNanoChat** | Modded-nanogpt distributed training | `lake build TrainNanoChat` |
| **FluxDemo** | Flux Klein 4B image generation | `lake build FluxDemo` |
| **BranchingFlows** | Combinatorial branching flow sampler | Part of `Examples` lib |
| **NanoProof** | Transformer theorem prover (model only) | Part of `Examples` lib |

### Distributed NanoChat (GPU Node)

Use the helper scripts to run `TrainNanoChat` under `torchrun` without pulling in a mismatched CUDA module stack:

```bash
# default: debug smoke run on 2 GPUs
./scripts/nanochat/run_train_torchrun.sh

# explicit 4-GPU run
NPROC_PER_NODE=4 ./scripts/nanochat/run_train_torchrun.sh \
  --debug --iterations 2 --data data/nanochat --val data/nanochat

# scaling check (1/2/4 GPUs by default)
./scripts/nanochat/bench_distributed.sh
```

Notes:
- `run_train_torchrun.sh` defaults `TORCHRUN_BIN` to `torchrun` from `PATH`, then falls back to `/grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/bin/torchrun`.
- Set `SKIP_MODULES=1` to skip cluster module loading on hosts without environment modules.
- Override launcher path with `TORCHRUN_BIN=/path/to/torchrun`.
- Override process counts in the benchmark script with `SIZES="2 4"` (or any space-separated list).

## Key Concepts

### Shape-Indexed Tensors

The core data type is `T s` - a tensor type indexed by its shape:

```lean
-- T is parameterized by shape (Array UInt64)
def T (s : Shape) : Type := TSpec.type

-- Shape mismatches are compile-time errors
def matmul {a b c : UInt64} (x : T #[a, b]) (y : T #[b, c]) : T #[a, c] := ...
```

### TensorStruct Typeclass

Generic traversal over structures containing tensors:

```lean
class TensorStruct (α : Type) where
  map     : (∀ {s}, T s → T s) → α → α
  mapM    : (∀ {s}, T s → m (T s)) → α → m α
  zipWith : (∀ {s}, T s → T s → T s) → α → α → α
  fold    : (∀ {s}, T s → β → β) → β → α → β
```

Use `Vector n α` instead of `Array α` for type-safe `zipWith` operations.

### Data Loading

Two patterns for different use cases:

```lean
-- Fixed dimensions (type-safe):
let iter := SequentialBatchIterator.new loader 8 256
let (batch, iter') := iter.next  -- Returns T #[8, 256]

-- Dynamic dimensions (modded-nanogpt style):
let iter := BatchIterator.new shard 8 256
let (batch, iter') ← iter.next   -- Returns T #[] (erased)
```

## FFI Reference Counting

The C++ bindings use careful reference counting. See `cc/src/tyr.cpp` header for details:

- `borrowTensor()`: Shared ownership, auto-cleanup
- `giveTensor()`: Transfer ownership to Lean
- `lean_dec()`: Required after extracting from `lean_obj_arg`, not for `b_lean_obj_arg`

Monitor tensor leaks via `get_live_tensors` which tracks outstanding C++ tensors.

## Development

### Adding New Tensor Operations

1. Add Lean declaration in `Tyr/Torch.lean` with `@[extern "lean_torch_xxx"]`
2. Implement in `cc/src/tyr.cpp` following reference counting conventions
3. Rebuild: `lake build`

### Project Structure

- `lakefile.lean` - Lake build configuration
- `lean-toolchain` - Lean version specification
- `cc/` - C++ FFI bindings (LibTorch wrapper)
- `Tyr/` - Core framework (tensors, modules, optimizers, distributed)
- `Examples/` - Training scripts and model implementations
- `Tests/` - Test suites

## License

TBD.
