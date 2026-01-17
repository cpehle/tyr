"""BUILD file for LeanTest dependency."""

load("@rules_lean//:defs.bzl", "lean_library")

package(default_visibility = ["//visibility:public"])

# LeanTest library
lean_library(
    name = "LeanTest",
    srcs = glob([
        "LeanTest.lean",
        "LeanTest/**/*.lean",
    ]),
    module_name = "LeanTest",
)
