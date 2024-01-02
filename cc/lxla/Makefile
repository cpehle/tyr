# Detect Lean

ifndef LEAN_HOME
LEAN ?= lean
LEAN_HOME := $(shell $(LEAN) --print-prefix)
endif

LEAN_INCLUDE := $(LEAN_HOME)/include

# Detect LIBTORCH

# XLA FLAGS
XLADIR=../../external/xla_extension
XLA_INC_FLAGS=-I$(XLADIR)/include
XLA_LIBPATH=$(XLADIR)/lib/

# Config

MKPATH := mkdir -p
RMPATH := rm -rf

AR := ar
CXX := c++
EXTRA_CXX_FLAGS := -O3 -Wall -Wno-sign-compare \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
	 -w -DLLVM_ON_UNIX=1

SRC_DIR := src
HDR_DIR := include
OUT_DIR := build

HDRS := \
	lxla_client.h\

SRCS := \
	lxla.cpp \
	lxla_client.cpp\

LIB_NAME := LXLA
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
	$(CXX) --std=c++17 -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(XLA_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -rdynamic

clean:
	$(RMPATH) $(OUT_DIR)

