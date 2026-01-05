# Build Instructions

## Building with Lake (Recommended)

```bash
# Build all targets
lake build

# Build specific executables
lake build test
lake build TrainGPT
lake build TrainDiffusion
lake build TestDataLoader
```

## Running Tests

```bash
# macOS
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
.lake/build/bin/test

# Linux
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
.lake/build/bin/test
```

Or use the helper script:
```bash
lake script run
```

## Running Training

```bash
# macOS
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
.lake/build/bin/TrainGPT

# Linux
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
.lake/build/bin/TrainGPT
```

Or use the helper script:
```bash
lake script train
```

## Building C++ Bindings Separately

```bash
make -C cc
```

## Legacy Build (Deprecated)

The `ninja` build system is deprecated. Use `lake build` instead.
