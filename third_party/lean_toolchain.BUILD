"""BUILD file for local Lean toolchain (elan-managed)."""

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(default_visibility = ["//visibility:public"])

exports_files(["bin/lean", "bin/leanc", "bin/lake"])

sh_binary(
    name = "lean",
    srcs = ["bin/lean"],
)

sh_binary(
    name = "leanc",
    srcs = ["bin/leanc"],
)

sh_binary(
    name = "lake",
    srcs = ["bin/lake"],
)

# Standard library olean files
filegroup(
    name = "stdlib_oleans",
    srcs = glob(["lib/lean/**/*.olean"]),
)

# Full lib/lean directory for LEAN_PATH
filegroup(
    name = "lib_lean",
    srcs = glob(["lib/lean/**"]),
)

# Lean headers for C compilation
filegroup(
    name = "lean_headers",
    srcs = glob(["include/lean/*.h"]),
)

# Lean shared runtime library
cc_import(
    name = "leanshared",
    shared_library = select({
        "@platforms//os:macos": "lib/lean/libleanshared.dylib",
        "@platforms//os:linux": "lib/lean/libleanshared.so",
        "//conditions:default": "lib/lean/libleanshared.so",
    }),
)

# Combined runtime with headers
cc_library(
    name = "leanrt",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    deps = [":leanshared"],
)
