# Detect Lean

ifndef LEAN_HOME
LEAN ?= lean
LEAN_HOME := $(shell $(LEAN) --print-prefix)
endif

LEAN_INCLUDE := $(LEAN_HOME)/include

# Detect LIBTORCH

LLVM_CONFIG ?= llvm-config
LLVM_CXX_FLAGS := $(shell $(LLVM_CONFIG) --cxxflags)

# TORCH FLAGS
TORCHDIR=../thirdparty/libtorch
TORCH_INC_FLAGS=-I$(TORCHDIR)/include -I$(TORCHDIR)/include/torch/csrc/api/include  -I$(TORCHDIR)/include/TH  -I$(TORCHDIR)/include/THC
TORCH_LIBPATH=$(TORCHDIR)/lib/


# Config

MKPATH := mkdir -p
RMPATH := rm -rf

AR := ar
CXX := c++
EXTRA_CXX_FLAGS := -O3

SRC_DIR := src
HDR_DIR := include
OUT_DIR := build

HDRS := \
	tyr.h\

SRCS := \
	tyr.cpp\

LIB_NAME := TyrC
LIB := lib${LIB_NAME}.a

OBJ_FILES := $(addprefix $(OUT_DIR)/,$(SRCS:.cpp=.o))
HDR_FILES := $(addprefix $(HDR_DIR)/,$(HDRS))

# Build Rules

all: lib

lib: $(OUT_DIR)/$(LIB)

$(OUT_DIR):
	$(MKPATH) $@

$(OUT_DIR)/$(LIB) : $(OBJ_FILES) | $(OUT_DIR)
	${AR} rcs $@ $^

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDR_FILES) | $(OUT_DIR)
	$(CXX) -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -rdynamic

clean:
	$(RMPATH) $(OUT_DIR)

