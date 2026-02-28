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
CXX := c++
NVCC ?= nvcc
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

LIB_NAME := TyrC
LIB := lib${LIB_NAME}.a
ifeq ($(UNAME_S),Darwin)
DYLIB := lib${LIB_NAME}.dylib
else
DYLIB := lib${LIB_NAME}.so
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
	-L$(ARROW_LIBPATH) -larrow -lparquet -lsoxr \
	-Wl,-rpath,$(LEAN_LIB) \
	-Wl,-rpath,$(abspath $(TORCH_LIBPATH)) \
	-Wl,-rpath,$(ARROW_LIBPATH) \
	$(APPLE_FRAMEWORK_FLAGS)

# Build Rules

all: lib dylib

lib: $(OUT_DIR)/$(LIB)

dylib: $(OUT_DIR)/$(DYLIB)

$(OUT_DIR):
	$(MKPATH) $@

$(OUT_DIR)/$(LIB) : $(OBJ_FILES) | $(OUT_DIR)
	$(RMPATH) $@
	${AR} rcs $@ $^

$(OUT_DIR)/$(DYLIB) : $(OBJ_FILES) | $(OUT_DIR)
	$(CXX) -shared -o $@ $^ $(DYLIB_LINK_FLAGS)

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDR_FILES) | $(OUT_DIR)
	$(CXX) --std=c++17 -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -rdynamic

$(OUT_DIR)/%.o : $(SRC_DIR)/%.mm $(HDR_FILES) | $(OUT_DIR)
	$(CXX) -x objective-c++ --std=c++17 -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(XLA_INC_FLAGS) $(ARROW_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -fobjc-arc

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cu $(HDR_FILES) | $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(TK_INC_FLAGS) $(NVCCFLAGS)

clean:
	$(RMPATH) $(OUT_DIR)
