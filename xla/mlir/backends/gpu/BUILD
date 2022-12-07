load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "xla-gpu-opt",
    srcs = ["xla-gpu-opt.cc"],
    deps = [
        "//xla/mlir/backends/gpu/transforms:passes",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/stream_executor:stream_executor_impl",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@tsl//protobuf:dnn_proto_cc_impl",
    ],
)
