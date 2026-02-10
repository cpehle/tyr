# Build Instructions

## Building with Lake (Recommended)

```bash
# Build all targets
lake build

# Build specific executables
lake build test_runner
lake build TrainGPT
lake build TrainDiffusion
lake build TrainNanoChat
lake build FluxDemo
lake build TestDataLoader
```

## Running Tests

```bash
# macOS
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
.lake/build/bin/test_runner

# Linux
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
.lake/build/bin/test_runner
```

Or use the helper script:
```bash
lake run
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
lake run train
```

## Building C++ Bindings Separately

```bash
make -C cc
```
