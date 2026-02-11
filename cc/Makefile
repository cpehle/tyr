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
TORCH_LIBPATH=$(TORCHDIR)/lib/

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
EXTRA_CXX_FLAGS := -fPIC -O3 -Wall -Wno-sign-compare \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment

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
	tyr_execution.cpp\

LIB_NAME := TyrC
LIB := lib${LIB_NAME}.a
ifeq ($(UNAME_S),Darwin)
DYLIB := lib${LIB_NAME}.dylib
else
DYLIB := lib${LIB_NAME}.so
endif

OBJ_FILES := $(addprefix $(OUT_DIR)/,$(SRCS:.cpp=.o))
HDR_FILES := $(addprefix $(HDR_DIR)/,$(HDRS))

# Lean runtime library
LEAN_LIB := $(LEAN_HOME)/lib/lean

# Shared library link flags
DYLIB_LINK_FLAGS := -L$(LEAN_LIB) -lleanshared \
	-L$(TORCH_LIBPATH) -ltorch -ltorch_cpu -lc10 \
	-L$(ARROW_LIBPATH) -larrow -lparquet \
	-Wl,-rpath,$(LEAN_LIB) \
	-Wl,-rpath,$(abspath $(TORCH_LIBPATH)) \
	-Wl,-rpath,$(ARROW_LIBPATH)

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

clean:
	$(RMPATH) $(OUT_DIR)
