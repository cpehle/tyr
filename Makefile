ifndef LEAN_HOME
LEAN ?= lean
LEAN_HOME := $(shell $(LEAN) --print-prefix)
endif

RMPATH := rm -rf
LEANMAKEFILE := ${LEAN_HOME}/share/lean/lean.mk
LEANMAKE := $(MAKE) -f $(LEANMAKEFILE)


LD_PRELOAD="$(PWD)/plugin/build/TyrPlugin.dll $(shell $(LEAN) --print-libdir)/libleanshared.so"

all: plugin

clean: clean-cc clean-lib clean-plugin clean-test

.PHONY: cc lib plugin test clean

cc:
	$(MAKE) -C cc

clean-cc:
	$(MAKE) -C cc clean

lib:
	+$(LEANMAKE) lib PKG=Tyr MORE_DEPS=leanpkg.toml

clean-lib:
	$(RMPATH) build

plugin: lib cc
	$(MAKE) -C plugin

clean-plugin:
	$(MAKE) -C plugin clean

test: plugin
	LD_PRELOAD=$(LD_PRELOAD) $(MAKE) -C test

clean-test:
	$(MAKE) -C test clean

