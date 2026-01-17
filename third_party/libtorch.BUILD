"""BUILD file for pre-downloaded LibTorch."""

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

# Core PyTorch libraries
cc_import(
    name = "torch_import",
    shared_library = "lib/libtorch.dylib",
)

cc_import(
    name = "torch_cpu_import",
    shared_library = "lib/libtorch_cpu.dylib",
)

cc_import(
    name = "c10_import",
    shared_library = "lib/libc10.dylib",
)

cc_import(
    name = "torch_global_deps_import",
    shared_library = "lib/libtorch_global_deps.dylib",
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
    linkopts = [
        "-Wl,-rpath,@loader_path/../../../external/libtorch/lib",
    ],
    deps = [
        ":c10_import",
        ":torch_cpu_import",
        ":torch_global_deps_import",
        ":torch_import",
    ],
)
