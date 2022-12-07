load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "xla-cpu-opt",
    srcs = ["xla-cpu-opt.cc"],
    deps = [
        "//xla/mlir/backends/cpu/transforms:passes",
        "//xla/mlir/xla_cpu/ir:xla_cpu",
        "//xla/mlir_hlo:all_passes",
        "//xla/mlir_hlo:gml_st",
        "//xla/mlir_hlo:gml_st_passes",
        "//xla/mlir_hlo:gml_st_test_passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:thlo",
        "//xla/service/cpu:cpu_compiler",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:register",
        "@tsl//protobuf:dnn_proto_cc_impl",
    ],
)
