load("//xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

# Add `--define=xla_gpu_with_openxla_compiler=1` to build command to enable experimental
# OpenXLA/IREE backend compiler.
config_setting(
    name = "with_openxla_compiler",
    values = {
        "define": "xla_gpu_with_openxla_compiler=1",
    },
)

# copybara:uncomment_begin(not supported in OSS build)
#
# build_test(
#     name = "xla-openxla-opt_build_test",
#     targets = [":xla-openxla-opt"],
# )
#
# xla_cc_binary(
#     name = "xla-openxla-opt",
#     srcs = ["xla-openxla-opt.cc"],
#     deps = [
#         "//xla/mlir/backends/openxla/transforms:passes",
#         "//xla/mlir_hlo:lhlo",
#         "//xla/mlir_hlo:lhlo_gpu",
#         "@tsl//tsl/platform:platform_port",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:MlirOptLib",
#         "@llvm-project//mlir:Transforms",
#     ],
# )
#
# copybara:uncomment_end
