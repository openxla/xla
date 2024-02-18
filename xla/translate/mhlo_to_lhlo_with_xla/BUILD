load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//xla:xla.bzl", "xla_cc_binary")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@tsl//tsl:tsl.bzl", "internal_visibility")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

cc_library(
    name = "mhlo_to_lhlo_with_xla",
    srcs = ["mhlo_to_lhlo_with_xla.cc"],
    hdrs = ["mhlo_to_lhlo_with_xla.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/service:backend",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_install_hdrs_gather",
        "//xla/translate/hlo_to_mhlo:attribute_importer",
        "//xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BuiltinLocationAttributesIncGen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/protobuf:dnn_proto_cc",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)

build_test(
    name = "xla-translate-opt_build_test",
    targets = [
        ":xla-translate-opt",
    ],
)

xla_cc_binary(
    name = "xla-translate-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/framework/transforms:passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:hlo_xla_runtime_pipeline",  # buildcleaner: keep
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@stablehlo//:register",
        "@tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "xla-translate-gpu-opt_build_test",
    targets = [
        ":xla-translate-gpu-opt",
    ],
)

xla_cc_binary(
    name = "xla-translate-gpu-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/framework/transforms:passes",
        "//xla/mlir_hlo:all_passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/service:gpu_plugin",
        "//xla/stream_executor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@stablehlo//:register",
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda(["//xla/stream_executor/cuda:cublas_plugin"]) + if_rocm([
        "//xla/stream_executor/rocm:rocblas_plugin",
    ]),
)
