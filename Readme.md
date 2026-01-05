# Tyr

A dependently-typed deep learning framework for Lean 4, providing compile-time tensor shape verification through LibTorch bindings.

## Overview

Tyr leverages Lean 4's dependent type system to catch tensor dimension mismatches at compile time, not runtime. This eliminates a major source of deep learning bugs while providing access to PyTorch's optimized tensor operations.

```lean
-- Shapes are tracked in the type system
def linear {m n b : UInt64} (x : T #[b, m]) (M : T #[n, m]) : T #[b, n] := ...

-- Mismatched dimensions fail at compile time, not runtime!
let x : T #[32, 768] := ...
let w : T #[768, 512] := ...
let y := linear x w  -- Error: expected T #[n, 768], got T #[768, 512]
```

## Architecture

```
Tyr/
├── Basic.lean          # Core types: Shape, DType, Device, T (tensor type)
├── Torch.lean          # Tensor operations (arange, zeros, matmul, etc.)
├── TensorStruct.lean   # Generic tensor tree traversal (like JAX PyTree)
├── Module/             # Neural network modules
│   ├── Core.lean       # Module typeclass
│   ├── Linear.lean     # Linear layers
│   └── LayerNorm.lean  # Layer normalization
├── Optim/              # Optimizers (AdamW, distributed variants)
├── GPT.lean            # GPT-2 implementation
├── NanoProof.lean      # Theorem-proving transformer
├── Train.lean          # Training loops
└── DataLoader.lean     # Data loading utilities

cc/                     # C++ FFI bindings
├── src/tyr.cpp         # Main LibTorch bindings
└── include/tyr.h       # Header file

external/
├── libtorch/           # PyTorch C++ library
└── xla_extension/      # Optional XLA compilation support
```

## Prerequisites

- **Lean 4** (v4.25.0-pre or compatible) - See `lean-toolchain`
- **LibTorch** (PyTorch C++ API) - Place in `external/libtorch/`
- **OpenMP** - For parallelization (`brew install libomp` on macOS)
- **XLA Extension** (optional) - For JIT compilation

## Quick Start

### Building

```bash
# Build all targets with Lake
lake build

# Build specific executables
lake build test
lake build TrainGPT
lake build TrainDiffusion
```

### Running Tests

```bash
# macOS
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
.lake/build/bin/test

# Linux
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
.lake/build/bin/test

# Or use the helper script
lake script run
```

### Training GPT

```bash
# macOS
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
.lake/build/bin/TrainGPT

# Linux
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
.lake/build/bin/TrainGPT

# Or use the helper script
lake script train
```

## Key Concepts

### Shape-Indexed Tensors

The core innovation is `T s` - a tensor type indexed by its shape:

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

## Model Implementations

- **GPT.lean**: GPT-2 architecture (124M parameters)
- **NanoProof.lean**: Theorem-proving transformer with rotary embeddings, RMSNorm, GQA
- **Diffusion.lean**: Discrete diffusion transformer for text generation

## Development

### Adding New Tensor Operations

1. Add Lean declaration in `Tyr/Torch.lean` with `@[extern "lean_torch_xxx"]`
2. Implement in `cc/src/tyr.cpp` following reference counting conventions
3. Rebuild: `make -C cc && ninja`

### Project Structure

- `lakefile.lean` - Lake build configuration
- `build.ninja` - Generated Ninja build rules
- `lean-toolchain` - Lean version specification

## License

[Add license information]

## Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) for training methodology
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) for optimization techniques
- [Equinox](https://github.com/patrick-kidger/equinox) for TensorStruct design inspiration
