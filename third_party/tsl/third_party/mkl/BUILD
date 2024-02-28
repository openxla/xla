load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])  # 3-Clause BSD

package(default_visibility = ["//visibility:public"])

alias(
    name = "build_with_mkl",
    actual = "@xla//xla/tsl/mkl:build_with_mkl",
)

alias(
    name = "build_with_mkl_lnx_x64",
    actual = "@xla//xla/tsl/mkl:build_with_mkl_lnx_x64",
)

alias(
    name = "build_with_mkl_lnx_openmp",
    actual = "@xla//xla/tsl/mkl:build_with_mkl_lnx_openmp",
)

alias(
    name = "build_with_mkl_windows_openmp",
    actual = "@xla//xla/tsl/mkl:build_with_mkl_windows_openmp",
)

alias(
    name = "build_with_mkl_aarch64",
    actual = "@xla//xla/tsl/mkl:build_with_mkl_aarch64",
)

alias(
    name = "enable_mkl",
    actual = "@xla//xla/tsl/mkl:enable_mkl",
)

alias(
    name = "intel_binary_blob",
    actual = "@xla//xla/tsl/mkl:intel_binary_blob",
)

alias(
    name = "LICENSE",
    actual = "@xla//xla/tsl/mkl:LICENSE",
)

alias(
    name = "mkl_libs_linux",
    actual = "@xla//xla/tsl/mkl:mkl_libs_linux",
)

alias(
    name = "mkl_libs_darwin",
    actual = "@xla//xla/tsl/mkl:mkl_libs_darwin",
)

alias(
    name = "mkl_libs_windows",
    actual = "@xla//xla/tsl/mkl:mkl_libs_windows",
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
