"""BUILD file for OpenMP (Homebrew libomp)."""

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "openmp",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    linkopts = [
        "-L/opt/homebrew/opt/libomp/lib",
        "-lomp",
        "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
    ],
)
