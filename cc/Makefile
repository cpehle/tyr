# Detect Lean
# Lake tracks this file hash for extern_lib invalidation.

ifndef LEAN_HOME
LEAN ?= lean
LEAN_HOME := $(shell $(LEAN) --print-prefix)
endif

LEAN_INCLUDE := $(LEAN_HOME)/include

# Detect LIBTORCH


# TORCH FLAGS
TORCHDIR=../external/libtorch
TORCH_INC_FLAGS=-I$(TORCHDIR)/include -I$(TORCHDIR)/include/torch/csrc/api/include  -I$(TORCHDIR)/include/TH  -I$(TORCHDIR)/include/THC
TORCH_LIBPATH=$(TORCHDIR)/lib
TKDIR=../thirdparty/ThunderKittens
TK_INC_FLAGS=-I$(TKDIR)/include -I$(TKDIR)/prototype
SOXR_DIR=../external/soxr
SOXR_BUILD_DIR=build/soxr
SOXR_INC_FLAGS=-I$(SOXR_DIR)/src -I$(SOXR_BUILD_DIR)
SOXR_LIBPATH=$(SOXR_BUILD_DIR)/src
SOXR_LIB=$(SOXR_LIBPATH)/libsoxr.a
SOXR_OBJ_DIR=$(SOXR_BUILD_DIR)/obj
SOXR_CONFIG=$(SOXR_BUILD_DIR)/soxr-config.h
SOXR_SRCS := soxr data-io dbesi0 filter fft4g64 cr cr32 fft4g32 cr64 vr32
SOXR_OBJS := $(addprefix $(SOXR_OBJ_DIR)/,$(addsuffix .o,$(SOXR_SRCS)))
SOXR_CFLAGS := -std=gnu89 -O3 -fPIC -I$(SOXR_DIR)/src -I$(SOXR_BUILD_DIR) -DSOXR_LIB

# ARROW/PARQUET FLAGS
# Apache Arrow installed via: brew install apache-arrow (macOS) or apt install libarrow-dev libparquet-dev (Linux)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS - check for Apple Silicon first, then Intel
    ifneq ($(wildcard /opt/homebrew/include/arrow),)
        ARROW_PREFIX := /opt/homebrew
    else ifneq ($(wildcard /usr/local/include/arrow),)
        ARROW_PREFIX := /usr/local
    else
        ARROW_PREFIX := /opt/homebrew
    endif
else
    # Linux
    ARROW_PREFIX := /usr
endif
ARROW_LIBPATH := $(ARROW_PREFIX)/lib
HAS_ARROW := 0
ifneq ($(wildcard $(ARROW_PREFIX)/include/arrow/api.h),)
ifneq ($(wildcard $(ARROW_LIBPATH)/libarrow.*),)
ifneq ($(wildcard $(ARROW_LIBPATH)/libparquet.*),)
HAS_ARROW := 1
endif
endif
endif
ifeq ($(HAS_ARROW),1)
ARROW_INC_FLAGS := -I$(ARROW_PREFIX)/include
ARROW_LINK_FLAGS := -L$(ARROW_LIBPATH) -larrow -lparquet
else
ARROW_INC_FLAGS :=
ARROW_LINK_FLAGS :=
endif



# Config

MKPATH := mkdir -p
RMPATH := rm -rf

AR := ar
CC := gcc
CXX := c++
NVCC ?= nvcc
HAS_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)
CUDA_MAJOR_VERSION := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n1)
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC) 2>/dev/null))
ifeq ($(strip $(CUDA_HOME)),)
CUDA_HOME := /usr/local/cuda
endif
CUDA_INC_FLAGS :=
ifneq ($(wildcard $(CUDA_HOME)/include/cuda_runtime.h),)
CUDA_INC_FLAGS += -I$(CUDA_HOME)/include
endif
CUDA_LIB_FLAGS :=
ifneq ($(wildcard $(CUDA_HOME)/lib64),)
CUDA_LIB_FLAGS += -L$(CUDA_HOME)/lib64
endif
ifneq ($(wildcard $(CUDA_HOME)/targets/aarch64-linux/lib),)
CUDA_LIB_FLAGS += -L$(CUDA_HOME)/targets/aarch64-linux/lib
endif
NCCL_ROOT ?= $(firstword $(wildcard ../.venv-gpu/lib/python*/site-packages/nvidia/nccl))
NCCL_INC_FLAGS :=
ifneq ($(wildcard $(NCCL_ROOT)/include/nccl.h),)
NCCL_INC_FLAGS += -I$(NCCL_ROOT)/include
endif
NCCL_LIB_FLAGS :=
ifneq ($(wildcard $(NCCL_ROOT)/lib),)
NCCL_LIB_FLAGS += -L$(NCCL_ROOT)/lib
endif
CXX_STD ?= c++20
CXX_STD_FLAG := --std=$(CXX_STD)
EXTRA_CXX_FLAGS := -fPIC -O3 -Wall -Wno-sign-compare \
		 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment
NVCCFLAGS := -std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -lineinfo -Xcompiler -fPIC
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# CUDA 12.1 headers do not expose cudaLaunchAttributePreferredClusterDimension.
ifneq ($(strip $(CUDA_MAJOR_VERSION)),)
ifeq ($(shell [ "$(CUDA_MAJOR_VERSION)" -lt 13 ] && echo 1 || echo 0),1)
NVCCFLAGS += -DcudaLaunchAttributePreferredClusterDimension=cudaLaunchAttributeClusterDimension
endif
endif

# Physical GPU target (controls generated SASS) and logical kernel family
# (controls which ThunderKittens/Tyr arch guards are enabled) are split so
# newer Blackwell-family devices such as GB10 can still build Hopper-authored
# kernels while also supporting true Blackwell surfaces when requested.
GPU ?= H100
GPU_FAMILY ?=
GPU_COMPUTE ?=
GPU_CODE ?=

ifeq ($(GPU),GB10)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := HOPPER
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_121
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_121
endif
else ifeq ($(GPU),B300)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := BLACKWELL
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_103a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_103a
endif
else ifeq ($(GPU),B200)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := BLACKWELL
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_100a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_100a
endif
else ifeq ($(GPU),A100)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := AMPERE
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_80a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_80a
endif
else ifeq ($(GPU),H100)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := HOPPER
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_90a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_90a
endif
else
$(error Unsupported GPU: $(GPU). Please set GPU to A100, H100, B200, B300, or GB10)
endif

ifeq ($(GPU_FAMILY),BLACKWELL)
NVCCFLAGS += -DKITTENS_BLACKWELL
else ifeq ($(GPU_FAMILY),AMPERE)
NVCCFLAGS += -DKITTENS_AMPERE
else ifeq ($(GPU_FAMILY),HOPPER)
NVCCFLAGS += -DKITTENS_HOPPER
else
$(error Unsupported GPU_FAMILY: $(GPU_FAMILY). Please set GPU_FAMILY to AMPERE, HOPPER, or BLACKWELL)
endif

NVCCFLAGS += -gencode arch=$(GPU_COMPUTE),code=$(GPU_CODE)

# Optional distributed backend toggles based on available LibTorch artifacts.
ifneq ($(wildcard $(TORCHDIR)/include/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp),)
ifneq ($(wildcard $(TORCH_LIBPATH)/libtorch_cuda.*),)
ifneq ($(wildcard $(TORCH_LIBPATH)/libc10_cuda.*),)
ifneq ($(strip $(NCCL_INC_FLAGS)),)
EXTRA_CXX_FLAGS += -DUSE_C10D_NCCL
TORCH_CUDA_LINK_FLAGS := -ltorch_cuda -lc10_cuda
endif
endif
endif
endif

# Clang-only warning flag (unsupported by GCC).
ifneq ($(findstring clang,$(shell $(CXX) --version 2>/dev/null)),)
EXTRA_CXX_FLAGS += -Winconsistent-missing-override
endif

SRC_DIR := src
HDR_DIR := include
OUT_DIR := build
TOOLS_DIR := tools
BENCH_FLASH_ATTN := $(OUT_DIR)/tools/bench_flash_attn
GPU_STUB_SRC := $(SRC_DIR)/generated/tyr_gpu_kernel_stubs.cpp
GPU_STUB_SCRIPT := $(TOOLS_DIR)/generate_gpu_kernel_stubs.py
GPU_STUB_IR_ROOT := ../.lake/build/ir/Tyr/GPU

HDRS :=

SRCS := \
	tyr.cpp\
	tyr_qwen35.cpp\
	tyr_ops.cpp\
	tyr_distributed.cpp\
	tyr_polar.cpp\
	tyr_media_stub.cpp\
	tyr_parquet.cpp\
	tyr_execution.cpp\
	float_buffer.cpp
MM_SRCS :=
ifeq ($(UNAME_S),Darwin)
MM_SRCS += apple_mps.mm
MM_SRCS += apple_vl_processor.mm
MM_SRCS += apple_audio_input.mm
MM_SRCS += apple_audio_output.mm
APPLE_FRAMEWORK_FLAGS := \
	-framework Foundation \
	-framework CoreFoundation \
	-framework Metal \
	-framework CoreGraphics \
	-framework ImageIO \
	-framework AVFoundation \
	-framework CoreMedia \
	-framework CoreVideo \
	-framework VideoToolbox \
	-framework Accelerate \
	-framework AudioToolbox
else
APPLE_FRAMEWORK_FLAGS :=
endif
STATIC_CU_SRCS := $(SRC_DIR)/tk_vendor_mha_h100.cu
GPU_CODEGEN_MODULES := $(strip $(TYR_GPU_CODEGEN_MODULE))
ifeq ($(GPU_CODEGEN_MODULES),)
CU_SRCS := $(STATIC_CU_SRCS) $(wildcard $(SRC_DIR)/generated/*.cu)
else ifeq ($(GPU_CODEGEN_MODULES),all)
CU_SRCS := $(STATIC_CU_SRCS) $(wildcard $(SRC_DIR)/generated/*.cu)
else
GPU_CODEGEN_CU_NAMES := $(addsuffix .cu,$(subst .,_,$(GPU_CODEGEN_MODULES)))
CU_SRCS := $(STATIC_CU_SRCS) $(addprefix $(SRC_DIR)/generated/,$(GPU_CODEGEN_CU_NAMES))
endif
ifneq ($(HAS_NVCC),1)
ifneq ($(strip $(CU_SRCS)),)
$(warning NVCC not found; disabling CUDA kernel compilation and using CPU stubs)
endif
CU_SRCS :=
SRCS += generated/tyr_gpu_kernel_stubs.cpp
endif
REQUIRES_TK := $(if $(strip $(CU_SRCS)),1,0)

LIB_NAME := TyrC
LIB := lib${LIB_NAME}.a
ifeq ($(UNAME_S),Darwin)
DYLIB := lib${LIB_NAME}.dylib
SHARED_FLAG := -dynamiclib
else
DYLIB := lib${LIB_NAME}.so
SHARED_FLAG := -shared
endif

CPP_OBJ_FILES := $(addprefix $(OUT_DIR)/,$(SRCS:.cpp=.o))
MM_OBJ_FILES := $(addprefix $(OUT_DIR)/,$(MM_SRCS:.mm=.o))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OUT_DIR)/%.o,$(CU_SRCS))
OBJ_FILES := $(CPP_OBJ_FILES) $(MM_OBJ_FILES) $(CU_OBJ_FILES)
HDR_FILES := $(addprefix $(HDR_DIR)/,$(HDRS))

# Lean runtime library
LEAN_LIB := $(LEAN_HOME)/lib/lean

# Shared library link flags
DYLIB_LINK_FLAGS := -L$(LEAN_LIB) -lleanshared \
	-L$(TORCH_LIBPATH) -ltorch -ltorch_cpu -lc10 $(TORCH_CUDA_LINK_FLAGS) \
	$(ARROW_LINK_FLAGS) \
	-L$(SOXR_LIBPATH) -lsoxr \
	$(CUDA_LIB_FLAGS) $(NCCL_LIB_FLAGS) \
	-Wl,-rpath,$(LEAN_LIB) \
	-Wl,-rpath,$(abspath $(TORCH_LIBPATH)) \
	$(APPLE_FRAMEWORK_FLAGS)

# Build Rules

.PHONY: check-submodules

check-submodules:
	@if [ ! -f "$(SOXR_DIR)/CMakeLists.txt" ]; then \
		echo "Missing populated submodule at $(SOXR_DIR). Run: git submodule update --init --recursive"; \
		exit 1; \
	fi
	@if [ "$(REQUIRES_TK)" = "1" ] && [ ! -d "$(TKDIR)/include" ]; then \
		echo "Missing populated submodule at $(TKDIR). Run: git submodule update --init --recursive"; \
		exit 1; \
	fi
	@if [ ! -f "$(TORCHDIR)/include/torch/csrc/api/include/torch/torch.h" ]; then \
		echo "Missing LibTorch headers at $(TORCHDIR). Install external/libtorch (see dependencies_macos.sh or Readme.md)."; \
		exit 1; \
	fi
	@if [ ! -d "$(TORCH_LIBPATH)" ]; then \
		echo "Missing LibTorch libraries at $(TORCH_LIBPATH). Install external/libtorch (see dependencies_macos.sh or Readme.md)."; \
		exit 1; \
	fi

all: lib dylib

lib: check-submodules soxr $(OUT_DIR)/$(LIB)

dylib: check-submodules soxr $(OUT_DIR)/$(DYLIB)

bench-flash-attn: check-submodules soxr $(OUT_DIR)/$(DYLIB) $(BENCH_FLASH_ATTN)

soxr: check-submodules $(SOXR_LIB)

$(SOXR_LIB): $(SOXR_OBJS) | check-submodules
	@if [ ! -f "$(SOXR_DIR)/CMakeLists.txt" ]; then \
		echo "Missing $(SOXR_DIR). Initialize submodules: git submodule update --init --recursive"; \
		exit 1; \
	fi
	$(MKPATH) $(SOXR_LIBPATH)
	$(RMPATH) $@
	$(AR) rcs $@ $(SOXR_OBJS)

$(SOXR_CONFIG): | check-submodules
	$(MKPATH) $(SOXR_BUILD_DIR)
	@printf '%s\n' \
		'#if !defined soxr_config_included' \
		'#define soxr_config_included' \
		'#define AVCODEC_FOUND 0' \
		'#define AVUTIL_FOUND 0' \
		'#define WITH_PFFFT 0' \
		'#define HAVE_FENV_H 1' \
		'#define HAVE_STDBOOL_H 1' \
		'#define HAVE_STDINT_H 1' \
		'#define HAVE_LRINT 1' \
		'#define HAVE_BIGENDIAN 0' \
		'#define WITH_CR32 1' \
		'#define WITH_CR32S 0' \
		'#define WITH_CR64 1' \
		'#define WITH_CR64S 0' \
		'#define WITH_VR32 1' \
		'#define WITH_HI_PREC_CLOCK 1' \
		'#define WITH_FLOAT_STD_PREC_CLOCK 0' \
		'#define WITH_DEV_TRACE 0' \
		'#endif' > $@

$(SOXR_OBJ_DIR)/%.o: $(SOXR_DIR)/src/%.c $(SOXR_CONFIG) | check-submodules
	$(MKPATH) $(SOXR_OBJ_DIR)
	$(CC) -o $@ -c $< $(SOXR_CFLAGS)

$(OUT_DIR):
	$(MKPATH) $@

$(OUT_DIR)/$(LIB) : $(OBJ_FILES) | check-submodules $(OUT_DIR)
	$(RMPATH) $@
	${AR} rcs $@ $^

$(OUT_DIR)/$(DYLIB) : $(OBJ_FILES) | check-submodules $(OUT_DIR)
	$(CXX) $(SHARED_FLAG) -o $@ $^ $(DYLIB_LINK_FLAGS)

$(BENCH_FLASH_ATTN): $(TOOLS_DIR)/bench_flash_attn.cpp $(OUT_DIR)/$(DYLIB) | check-submodules $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_STD_FLAG) -o $@ $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -L$(OUT_DIR) -l$(LIB_NAME) $(DYLIB_LINK_FLAGS) -Wl,-rpath,$(abspath $(OUT_DIR))

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDR_FILES) | check-submodules $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_STD_FLAG) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC

$(OUT_DIR)/%.o : $(SRC_DIR)/%.mm $(HDR_FILES) | check-submodules $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) -x objective-c++ $(CXX_STD_FLAG) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -fobjc-arc

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cu $(HDR_FILES) | check-submodules $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(TK_INC_FLAGS) $(NVCCFLAGS)

$(GPU_STUB_SRC): $(GPU_STUB_SCRIPT)
	$(MKPATH) $(dir $@)
	python3 $(GPU_STUB_SCRIPT) --ir-root $(GPU_STUB_IR_ROOT) --output $@

clean:
	$(RMPATH) $(OUT_DIR)
	$(RMPATH) $(SOXR_BUILD_DIR)
