load("//xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

# Add `--//third_party/tensorflow/compiler/xla/mlir/backends/openxla:enable` to build command to
# enable experimental backend compiler and runtime.
bool_flag(
    name = "enable",
    build_setting_default = False,
)

config_setting(
    name = "enabled",
    flag_values = {":enable": "True"},
)

build_test(
    name = "xla-openxla-opt_build_test",
    targets = [":xla-openxla-opt"],
)

xla_cc_binary(
    name = "xla-openxla-opt",
    srcs = ["xla-openxla-opt.cc"],
    deps = [
        "//xla/mlir/backends/openxla/transforms:passes",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Transforms",
        "@tsl//tsl/platform:platform_port",
    ],
)
