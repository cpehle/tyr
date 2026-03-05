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
ARROW_INC_FLAGS := -I$(ARROW_PREFIX)/include
ARROW_LIBPATH := $(ARROW_PREFIX)/lib



# Config

MKPATH := mkdir -p
RMPATH := rm -rf

AR := ar
CC := gcc
CXX := c++
NVCC ?= nvcc
HAS_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)
CXX_STD ?= c++20
CXX_STD_FLAG := --std=$(CXX_STD)
EXTRA_CXX_FLAGS := -fPIC -O3 -Wall -Wno-sign-compare \
		 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment
NVCCFLAGS := -std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -lineinfo -Xcompiler -fPIC
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# CUDA 12.1 headers do not expose cudaLaunchAttributePreferredClusterDimension.
NVCCFLAGS += -DcudaLaunchAttributePreferredClusterDimension=cudaLaunchAttributeClusterDimension
GPU ?= H100
ifeq ($(GPU),B300)
NVCCFLAGS += -DKITTENS_BLACKWELL -gencode arch=compute_103a,code=sm_103a
else ifeq ($(GPU),B200)
NVCCFLAGS += -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else ifeq ($(GPU),A100)
NVCCFLAGS += -DKITTENS_AMPERE -gencode arch=compute_80a,code=sm_80a
else
NVCCFLAGS += -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a
endif

# Optional distributed backend toggles based on available LibTorch artifacts.
ifneq ($(wildcard $(TORCHDIR)/include/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp),)
ifneq ($(wildcard $(TORCH_LIBPATH)/libtorch_cuda.*),)
ifneq ($(wildcard $(TORCH_LIBPATH)/libc10_cuda.*),)
EXTRA_CXX_FLAGS += -DUSE_C10D_NCCL
TORCH_CUDA_LINK_FLAGS := -ltorch_cuda -lc10_cuda
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

HDRS :=

SRCS := \
	tyr.cpp\
	tyr_distributed.cpp\
	tyr_polar.cpp\
	tyr_media_stub.cpp\
	tyr_parquet.cpp\
	tyr_execution.cpp
MM_SRCS :=
ifeq ($(UNAME_S),Darwin)
MM_SRCS += apple_vl_processor.mm
MM_SRCS += apple_audio_input.mm
APPLE_FRAMEWORK_FLAGS := \
	-framework Foundation \
	-framework CoreFoundation \
	-framework CoreGraphics \
	-framework ImageIO \
	-framework AVFoundation \
	-framework CoreMedia \
	-framework CoreVideo \
	-framework VideoToolbox \
	-framework Accelerate
else
APPLE_FRAMEWORK_FLAGS :=
endif
CU_SRCS := $(wildcard $(SRC_DIR)/generated/*.cu)
ifneq ($(HAS_NVCC),1)
ifneq ($(strip $(CU_SRCS)),)
$(warning NVCC not found; disabling CUDA kernel compilation and using CPU stubs)
endif
CU_SRCS :=
SRCS += tyr_gpu_kernel_stubs.cpp
endif

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
	-L$(ARROW_LIBPATH) -larrow -lparquet \
	-L$(SOXR_LIBPATH) -lsoxr \
	-Wl,-rpath,$(LEAN_LIB) \
	-Wl,-rpath,$(abspath $(TORCH_LIBPATH)) \
	-Wl,-rpath,$(ARROW_LIBPATH) \
	$(APPLE_FRAMEWORK_FLAGS)

# Build Rules

all: lib dylib

lib: soxr $(OUT_DIR)/$(LIB)

dylib: soxr $(OUT_DIR)/$(DYLIB)

soxr: $(SOXR_LIB)

$(SOXR_LIB): $(SOXR_OBJS)
	@if [ ! -f "$(SOXR_DIR)/CMakeLists.txt" ]; then \
		echo "Missing $(SOXR_DIR). Initialize submodules: git submodule update --init --recursive"; \
		exit 1; \
	fi
	$(MKPATH) $(SOXR_LIBPATH)
	$(RMPATH) $@
	$(AR) rcs $@ $(SOXR_OBJS)

$(SOXR_CONFIG):
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

$(SOXR_OBJ_DIR)/%.o: $(SOXR_DIR)/src/%.c $(SOXR_CONFIG)
	$(MKPATH) $(SOXR_OBJ_DIR)
	$(CC) -o $@ -c $< $(SOXR_CFLAGS)

$(OUT_DIR):
	$(MKPATH) $@

$(OUT_DIR)/$(LIB) : $(OBJ_FILES) | $(OUT_DIR)
	$(RMPATH) $@
	${AR} rcs $@ $^

$(OUT_DIR)/$(DYLIB) : $(OBJ_FILES) | $(OUT_DIR)
	$(CXX) $(SHARED_FLAG) -o $@ $^ $(DYLIB_LINK_FLAGS)

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDR_FILES) | $(OUT_DIR)
	$(CXX) $(CXX_STD_FLAG) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -rdynamic

$(OUT_DIR)/%.o : $(SRC_DIR)/%.mm $(HDR_FILES) | $(OUT_DIR)
	$(CXX) -x objective-c++ $(CXX_STD_FLAG) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -fobjc-arc

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cu $(HDR_FILES) | $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(TK_INC_FLAGS) $(NVCCFLAGS)

clean:
	$(RMPATH) $(OUT_DIR)
	$(RMPATH) $(SOXR_BUILD_DIR)
