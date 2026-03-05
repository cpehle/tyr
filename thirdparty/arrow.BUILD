"""BUILD file for Apache Arrow (Homebrew installation)."""

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "arrow",
    hdrs = glob([
        "include/arrow/**/*.h",
        "include/parquet/**/*.h",
    ]),
    includes = ["include"],
    linkopts = select({
        "@platforms//os:macos": [
            "-L/opt/homebrew/lib",
            "-larrow",
            "-lparquet",
            "-Wl,-rpath,/opt/homebrew/lib",
        ],
        "@platforms//os:linux": [
            "-Llib",
            "-larrow",
            "-lparquet",
        ],
        "//conditions:default": [
            "-larrow",
            "-lparquet",
        ],
    }),
)
