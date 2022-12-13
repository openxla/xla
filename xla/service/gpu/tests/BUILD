# Description: GPU-specific XLA tests. For example, codegen tests that
# verify the IR emitted.
#
# TODO(jlebar): None of these tests actually use the GPU, so they should not
# need to run on machines with GPUs present.

load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//xla:glob_lit_test.bzl", "glob_lit_tests")
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)

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
    name = "gpu_codegen_test",
    testonly = True,
    srcs = ["gpu_codegen_test.cc"],
    hdrs = ["gpu_codegen_test.h"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:types",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "element_wise_row_vectorization_test",
    srcs = ["element_wise_row_vectorization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "mlir_gpu_test_base",
    testonly = True,
    srcs = ["mlir_gpu_test_base.cc"],
    hdrs = ["mlir_gpu_test_base.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:ir_emitter",
        "//xla/service/gpu:target_constants",
        "//xla/stream_executor/gpu:gpu_init",
        "//xla/tests:hlo_test_base",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

xla_cc_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_fft_test",
    srcs = ["mlir_fft_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_gemm_test",
    srcs = ["mlir_gemm_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_sorting_test",
    srcs = ["mlir_sorting_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mlir_gpu_compile_test",
    srcs = ["mlir_gpu_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":mlir_gpu_test_base",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_reduce_scatter_creator_test",
    srcs = ["gpu_reduce_scatter_creator_test.cc"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service/gpu:gpu_reduce_scatter_creator",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_query",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gemm_rewrite_test",
    srcs = [
        "gemm_rewrite_test.cc",
    ],
    tags = tf_cuda_tests_tags() + [
        "no_rocm",
    ],
    deps = [
        ":gpu_codegen_test",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gemm_broadcast_folding_rewrite_test",
    srcs = [
        "gemm_broadcast_folding_rewrite_test.cc",
    ],
    tags = tf_cuda_tests_tags() + [
        "no_rocm",
    ],
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gemm_broadcast_folding_rewriter",
        "//xla/service/gpu:gemm_rewriter",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_too_many_blocks_test",
    srcs = [
        "gpu_too_many_blocks_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:statusor",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_degenerate_dim_remover_test",
    srcs = [
        "reduction_degenerate_dim_remover_test.cc",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_degenerate_dim_remover",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_layout_normalizer_test",
    srcs = [
        "reduction_layout_normalizer_test.cc",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_layout_normalizer",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = [
        "tree_reduction_rewriter_test.cc",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:tree_reduction_rewriter",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "swap_conv_operands_test",
    srcs = [
        "swap_conv_operands_test.cc",
    ],
    tags = ["no_rocm"] + tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_vectorization_test",
    srcs = [
        "reduction_vectorization_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gemm_rewriter",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_dimension_grouper_test",
    srcs = [
        "reduction_dimension_grouper_test.cc",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:reduction_dimension_grouper",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "parallel_reduction_test",
    srcs = [
        "parallel_reduction_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gpu_executable",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_compilation_parallelism_test",
    srcs = [
        "gpu_compilation_parallelism_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_test",
    srcs = ["gpu_copy_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_copy_alone_test",
    srcs = [
        "gpu_copy_alone_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:verified_hlo_module",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_index_test",
    srcs = ["gpu_index_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_infeed_test",
    srcs = ["infeed_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_kernel_tiling_test",
    srcs = ["gpu_kernel_tiling_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "transpose_emitter_test",
    srcs = ["transpose_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reduction_emitter_test",
    srcs = ["reduction_emitter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fusion_test",
    srcs = ["gpu_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:instruction_fusion",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_fusion_pipeline_test",
    srcs = ["gpu_fusion_pipeline_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service/gpu:fusion_merger",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:instruction_fusion",
        "//xla/service/gpu:multi_output_fusion",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_unrolling_test",
    srcs = ["gpu_unrolling_test.cc"],
    tags = tf_cuda_tests_tags() + ["no_rocm"],
    deps = [
        ":gpu_codegen_test",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_alignment_test",
    testonly = True,
    srcs = ["gpu_alignment_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/service:custom_call_target_registry",
        "//xla/service:gpu_plugin",
        "//xla/service/llvm_ir:alias_analysis",
        "//xla/tests:filecheck",
        "//xla/tests:llvm_irgen_test_base",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_atomic_test",
    srcs = ["gpu_atomic_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:filecheck",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "elemental_ir_emitter_test",
    srcs = ["elemental_ir_emitter_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-nvidia",
        "requires-gpu-sm70-only",
        "requires-gpu-sm80-only",
    ]},
    backends = [
        "gpu",
    ],
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:filecheck",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "gpu_input_fusible_slice_test",
    srcs = ["gpu_input_fusible_slice_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_convolution_regression_test",
    srcs = ["gpu_convolution_regression_test.cc"],
    backend_args = {"gpu": [
        "--xla_enable_hlo_passes_only=layout-assignment,gpu-conv-algorithm-picker",
        "--xla_gpu_crash_on_verification_failures",
    ]},
    backends = ["gpu"],
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "select_and_scatter_test",
    srcs = ["select_and_scatter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "sorting_test",
    srcs = ["sorting_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_binary(
    name = "hlo_to_llvm_ir",
    srcs = ["hlo_to_llvm_ir.cc"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        "@llvm-project//llvm:Target",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_device_info",
        "//xla/service/gpu:target_constants",
        "//xla/service/gpu/llvm_gpu_backend",
        "//xla/tools:hlo_module_loader",
        "@tsl//tsl/platform:logging",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor:device_description_proto_cc_impl",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_impl",
        "//xla/stream_executor:dnn",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/util:command_line_flags",
        "@tsl//tsl/util:determinism",
    ] + if_cuda_is_configured([
        "//xla/service/gpu:nvptx_compiler_impl",
        "//xla/service/gpu:nvptx_helper",
        "//xla/stream_executor/cuda:cublas_plugin",
    ]) + if_rocm_is_configured([
        "//xla/service/gpu:amdgpu_compiler_impl",
    ]),
)

glob_lit_tests(
    data = [":test_utilities"],
    default_tags = tf_cuda_tests_tags() + [
        "no_pip",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = ["execute_memzero_thunk.mlir"],
    tags_override = {
        "reduction_vectorization_sm_all.hlo": ["no_rocm"],
        "element_wise_row_vectorization.hlo": ["no_rocm"],
    },
    test_file_exts = ["hlo"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":hlo_to_llvm_ir",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_codegen_test",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "in_place_op_test",
    srcs = ["in_place_op_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla/service:gpu_plugin",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test_main",
    ],
)
