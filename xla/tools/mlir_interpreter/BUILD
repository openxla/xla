load("@tsl/platform:rules_cc.bzl", "cc_binary", "cc_library", "cc_test")

package(
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "interpreter",
    srcs = [
        "interpreter.cc",
        "interpreter_value.cc",
        "interpreter_value_util.cc",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_value.h",
        "interpreter_value_util.h",
    ],
    deps = [
        ":tensor_or_memref",
        "//base:googleinit",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl/platform:logging",
    ],
)

cc_library(
    name = "interpreter_instrumentation",
    srcs = ["interpreter_instrumentation.cc"],
    hdrs = ["interpreter_instrumentation.h"],
    deps = [
        ":interpreter",
        "//xla/tools/mlir_interpreter/dialects",
        "@com_google_absl//absl/random",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tsl/platform:logging",
    ],
)

cc_library(
    name = "tensor_or_memref",
    srcs = ["tensor_or_memref.cc"],
    hdrs = ["tensor_or_memref.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_binary(
    name = "mlir-interpreter-runner",
    srcs = ["interpreter_main.cc"],
    deps = [
        ":interpreter",
        "//xla/mlir_hlo:gml_st",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/mlir_hlo:thlo",
        "//xla/tools/mlir_interpreter/dialects",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Support",
        "@tsl/platform:platform_port",
    ],
)

cc_test(
    name = "interpreter_value_test",
    srcs = ["interpreter_value_test.cc"],
    deps = [
        ":interpreter",
        ":tensor_or_memref",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
