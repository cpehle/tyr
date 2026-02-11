"""BUILD file for pre-downloaded LibTorch."""

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

# Core PyTorch libraries
cc_import(
    name = "torch_import",
    shared_library = select({
        "@platforms//os:macos": "lib/libtorch.dylib",
        "@platforms//os:linux": "lib/libtorch.so",
        "//conditions:default": "lib/libtorch.so",
    }),
)

cc_import(
    name = "torch_cpu_import",
    shared_library = select({
        "@platforms//os:macos": "lib/libtorch_cpu.dylib",
        "@platforms//os:linux": "lib/libtorch_cpu.so",
        "//conditions:default": "lib/libtorch_cpu.so",
    }),
)

cc_import(
    name = "c10_import",
    shared_library = select({
        "@platforms//os:macos": "lib/libc10.dylib",
        "@platforms//os:linux": "lib/libc10.so",
        "//conditions:default": "lib/libc10.so",
    }),
)

cc_import(
    name = "torch_global_deps_import",
    shared_library = select({
        "@platforms//os:macos": "lib/libtorch_global_deps.dylib",
        "@platforms//os:linux": "lib/libtorch_global_deps.so",
        "//conditions:default": "lib/libtorch_global_deps.so",
    }),
)

# Combined LibTorch library with all dependencies and headers
cc_library(
    name = "libtorch",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    linkopts = select({
        "@platforms//os:macos": [
            "-Wl,-rpath,@loader_path/../../../external/libtorch/lib",
        ],
        "@platforms//os:linux": [
            "-Wl,-rpath,$ORIGIN/../../../external/libtorch/lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":c10_import",
        ":torch_cpu_import",
        ":torch_global_deps_import",
        ":torch_import",
    ],
)
