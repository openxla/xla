# Description:
#    Tests for LLVM-based CPU backend for XLA.

load("@tsl//:tsl.default.bzl", "filegroup")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
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
    name = "cpu_codegen_test",
    testonly = True,
    hdrs = ["cpu_codegen_test.h"],
    deps = [
        "//xla/service:cpu_plugin",
        "//xla/tests:llvm_irgen_test_base",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_dyn_shape_test",
    srcs = ["cpu_dyn_shape_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_fusion_test",
    srcs = ["cpu_fusion_test.cc"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:cpu_instruction_fusion",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/memory",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_bytesizeof_test",
    srcs = ["cpu_bytesizeof_test.cc"],
    deps = [
        "//xla:shape_util",
        "//xla/service/llvm_ir:llvm_util",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_external_constants_test",
    srcs = ["cpu_external_constants_test.cc"],
    deps = [
        "//xla:array2d",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "//xla/tests:filecheck",
        "@tsl//platform:test",
    ],
)

tf_cc_test(
    name = "cpu_noalias_test",
    srcs = ["cpu_noalias_test.cc"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "//xla/service/llvm_ir:alias_analysis",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:filecheck",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Core",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_intrinsic_test",
    srcs = ["cpu_intrinsic_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_eigen_dot_operation_test",
    srcs = ["cpu_eigen_dot_operation_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/strings",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_profiling_test",
    srcs = ["cpu_profiling_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = ["tree_reduction_rewriter_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/cpu:cpu_compiler",
        "//xla/stream_executor/lib",
        "//xla/tests:codegen_test_base",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/strings",
        "@tsl//lib/core:status_test_util",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_infeed_test",
    srcs = ["cpu_infeed_test.cc"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:arithmetic",
        "//xla/service:cpu_plugin",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "@tsl//platform:env",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_literal_caching_test",
    srcs = ["cpu_literal_caching_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_outfeed_test",
    srcs = ["cpu_outfeed_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_key_value_sort_test",
    srcs = ["cpu_key_value_sort_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_spmd_compile_test",
    srcs = ["cpu_spmd_compile_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_query",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/tests:hlo_test_base",
        "@tsl//lib/core:status_test_util",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_topk_test",
    srcs = ["cpu_topk_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/client:xla_builder",
        "//xla/client/lib:sorting",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_vectorization_test",
    srcs = ["cpu_vectorization_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)

tf_cc_test(
    name = "cpu_while_test",
    srcs = ["cpu_while_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@tsl//platform:logging",
        "@tsl//platform:test",
        "@tsl//platform:test_main",
    ],
)
