load("//xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
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
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:MlirOptLib",
#     ],
# )
#
# copybara:uncomment_end
