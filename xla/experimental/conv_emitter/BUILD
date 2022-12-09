# Description:
#   MLIR-GPU-specific convolution in XLA service implementation.

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "conv_emitter",
    srcs = ["conv_emitter.cc"],
    hdrs = ["conv_emitter.h"],
    deps = [
        ":conv_emitter_transforms",
        "//xla:permutation_util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "conv_emitter_transforms",
    srcs = ["conv_emitter_transforms.cc"],
    hdrs = ["conv_emitter_transforms.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TransformUtils",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:types",
    ],
)

tf_cc_test(
    name = "conv_emitter_test",
    srcs = ["conv_emitter_test.cc"],
    deps = [
        ":conv_emitter",
        "//xla/service:hlo_parser",
        "//xla/tests:filecheck",
        "//xla/tests:verified_hlo_module",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Transforms",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)
