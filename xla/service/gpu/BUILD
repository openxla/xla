# Description:
#   GPU-specific components in XLA service implementation.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//third_party/tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow/tsl:tsl.bzl", "if_google", "if_nccl", "tsl_copts", "tsl_gpu_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//tensorflow/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_cloud")

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

tf_proto_library(
    name = "backend_configs",
    srcs = ["backend_configs.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//xla:xla_data_proto",
        "//xla/stream_executor:dnn_proto",
    ],
)

cc_library(
    name = "gpu_executable_run_options",
    srcs = ["gpu_executable_run_options.cc"],
    hdrs = ["gpu_executable_run_options.h"],
    compatible_with = get_compatible_with_cloud(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/service:executable",
        "//xla/service:global_device_id",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "gpu_constants",
    hdrs = ["gpu_constants.h"],
    deps = [
        "//xla:types",
    ],
)

cc_library(
    name = "gpu_types",
    hdrs = ["gpu_types.h"],
    deps = [
        "//xla:types",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "launch_dimensions",
    srcs = [
        "launch_dimensions.cc",
    ],
    hdrs = [
        "launch_dimensions.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":gpu_device_info",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
    ],
)

cc_library(
    name = "custom_call_thunk",
    srcs = ["custom_call_thunk.cc"],
    hdrs = ["custom_call_thunk.h"],
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]),
    deps = [
        ":buffer_allocations",
        ":thunk",
        "//tensorflow/tsl/platform:errors",
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/service:custom_call_status_internal",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "custom_call_test",
    srcs = if_gpu_is_configured(["custom_call_test.cc"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla:debug_options_flags",
        "//xla/client:xla_builder",
        "//xla/client/lib:constants",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:gpu_plugin",
        "//xla/runtime:module",
        "//xla/runtime/ffi:ffi_api_hdrs",
        "//xla/runtime:module_registry",
        "//xla/tests:client_library_test_base",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//xla/stream_executor/gpu:gpu_types_header",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "hlo_to_ir_bindings",
    srcs = ["hlo_to_ir_bindings.cc"],
    hdrs = ["hlo_to_ir_bindings.h"],
    deps = [
        ":buffer_allocations",
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:logging",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:tuple_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "target_util",
    srcs = ["target_util.cc"],
    hdrs = ["target_util.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "target_util_test",
    srcs = ["target_util_test.cc"],
    deps = [
        ":target_util",
        "//tensorflow/tsl/platform:test",
        "//xla/tests:xla_internal_test_main",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "gpu_device_info",
    srcs = ["gpu_device_info.cc"],
    hdrs = ["gpu_device_info.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "gpu_device_info_for_tests",
    testonly = 1,
    srcs = ["gpu_device_info_for_tests.cc"],
    hdrs = ["gpu_device_info_for_tests.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":gpu_device_info",
    ],
)

tf_cc_test(
    name = "gpu_device_info_test",
    srcs = ["gpu_device_info_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_device_info_for_tests",
        "//tensorflow/tsl/platform:test_main",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_driver",
        "//xla/stream_executor/cuda:cuda_gpu_executor",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "ir_emitter_context",
    srcs = [
        "ir_emitter_context.cc",
    ],
    hdrs = [
        "ir_emitter_context.h",
    ],
    deps = [
        ":gpu_constants",
        ":gpu_device_info",
        ":gpu_executable",
        "//xla/service:buffer_assignment",
        "//xla/service:name_uniquer",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "ir_emitter",
    srcs = [
        "ir_emitter.cc",
        "ir_emitter_nested.cc",
        "ir_emitter_unnested.cc",
    ],
    hdrs = [
        "ir_emitter.h",
        "ir_emitter_nested.h",
        "ir_emitter_unnested.h",
        "kernel_mapping_scheme.h",
    ],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":buffer_allocations",
        ":elemental_ir_emitter",
        ":ir_emitter_context",
        ":fft_thunk",
        ":gpu_asm_opts_util",
        ":gpu_constants",
        ":gpu_conv_runner",
        ":gpu_executable",
        ":gpu_device_info",
        ":gpu_fusible",
        ":hlo_to_ir_bindings",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":matmul_utils",
        ":nccl_collective_thunks",
        ":parallel_loop_emitter",
        ":target_util",
        ":thunk",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "//xla/translate/mhlo_to_lhlo_with_xla",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/mlir_hlo:transforms_gpu_passes",
        "//xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:union_find",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:elemental_ir_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_execution_profile",
        "//xla/service:name_uniquer",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/service:while_loop_analysis",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:dynamic_update_slice_util",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:ir_builder_mixin",
        "//xla/service/llvm_ir:kernel_support_library",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "//xla/service/llvm_ir:sort_util",
        "//xla/service/llvm_ir:tuple_ops",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
        "//xla/translate/mhlo_to_hlo:location_exporter",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:human_readable_json",
        "//tensorflow/tsl/platform:status",
    ] + if_gpu_is_configured([
        ":triangular_solve_thunk",
        ":cholesky_thunk",
    ]) + if_cuda_is_configured([
        ":cublas_lt_matmul_thunk",
    ]),
)

cc_library(
    name = "parallel_loop_emitter",
    srcs = ["parallel_loop_emitter.cc"],
    hdrs = ["parallel_loop_emitter.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":launch_dimensions",
        ":target_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:kernel_support_library",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        ":backend_configs_cc",
        ":ir_emitter_context",
        ":target_util",
        "//tensorflow/tsl/platform:logging",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "//xla/service/llvm_ir:math_ops",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "buffer_allocations",
    srcs = ["buffer_allocations.cc"],
    hdrs = ["buffer_allocations.h"],
    deps = [
        ":gpu_constants",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "thunk",
    srcs = ["thunk.cc"],
    hdrs = ["thunk.h"],
    deps = [
        ":buffer_allocations",
        ":gpu_executable_run_options",
        "//tensorflow/tsl/platform:status",
        "//xla:executable_run_options",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor",
        "@llvm-project//mlir:IR",
    ],
)

tsl_gpu_library(
    name = "nccl_collective_thunks",
    srcs = [
        "nccl_all_gather_thunk.cc",
        "nccl_all_reduce_thunk.cc",
        "nccl_all_to_all_thunk.cc",
        "nccl_collective_permute_thunk.cc",
        "nccl_collective_thunk.cc",
    ],
    hdrs = [
        "nccl_all_gather_thunk.h",
        "nccl_all_reduce_thunk.h",
        "nccl_all_to_all_thunk.h",
        "nccl_collective_permute_thunk.h",
        "nccl_collective_thunk.h",
    ],
    # Override tsl_gpu_library()'s internal default value of ["//buildenv/target:gce"].
    compatible_with = [],
    deps = [
        ":buffer_allocations",
        ":ir_emission_utils",
        ":nccl_utils",
        ":thunk",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/service:buffer_assignment",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
    ],
)

# Empty library to implement nested dependency conditions.
cc_library(name = "empty")

# If NCCL/RCCL is supported, this target '#defines XLA_ENABLE_XCCL' and
# provides a header which #includes NCCL/RCCL.
alias(
    name = "nccl_utils",
    actual = if_nccl(":_nccl_utils", ":empty"),
)

# Do not depend on this target, but rather depend on :nccl_utils.
tsl_gpu_library(
    name = "_nccl_utils",
    srcs = if_gpu_is_configured(["nccl_utils.cc"]),
    hdrs = if_gpu_is_configured(["nccl_utils.h"]),
    # Override tsl_gpu_library()'s internal default value of ["//buildenv/target:gce"].
    compatible_with = [],
    defines = if_gpu_is_configured(["XLA_ENABLE_XCCL"]),
    tags = ["manual"],  # Only builds with if_nccl().
    deps = if_gpu_is_configured([
        ":gpu_executable_run_options",
        ":thunk",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xla:debug_options_flags",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/service:rendezvous",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:env",
    ]) + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rccl",
    ]),
)

# TODO(b/244780257): Remove this config.
bool_flag(
    name = "enable_xlir",
    build_setting_default = if_google(True, False),
)

cc_library(
    name = "gpu_executable",
    srcs = [
        "conditional_thunk.cc",
        "convolution_thunk.cc",
        "copy_thunk.cc",
        "for_thunk.cc",
        "gpu_executable.cc",
        "infeed_thunk.cc",
        "kernel_thunk.cc",
        "memset_thunk.cc",
        "outfeed_thunk.cc",
        "replica_id_thunk.cc",
        "sequential_thunk.cc",
        "while_thunk.cc",
    ],
    hdrs = [
        "conditional_thunk.h",
        "convolution_thunk.h",
        "copy_thunk.h",
        "custom_call_thunk.h",
        "for_thunk.h",
        "gemm_thunk.h",
        "gpu_executable.h",
        "infeed_thunk.h",
        "kernel_thunk.h",
        "memset_thunk.h",
        "outfeed_thunk.h",
        "replica_id_thunk.h",
        "sequential_thunk.h",
        "while_thunk.h",
    ],
    deps = [
        ":backend_configs_cc",
        ":buffer_allocations",
        ":cusolver_context",
        ":custom_call_thunk",
        ":fft_thunk",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_constants",
        ":gpu_conv_runner",
        ":gpu_executable_run_options",
        ":gpu_types",
        ":io_feed_manager",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":matmul_utils",
        ":nccl_collective_thunks",
        ":stream_executor_util",
        ":thunk",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "//xla/service/gpu/runtime:gemm",
        "//xla:array2d",
        "//xla:literal",
        "//xla:refcounting_hash_map",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/runtime:diagnostics",
        "//xla/runtime:executable",
        "//xla/runtime:jit_executable",
        "//xla/service:buffer_assignment",
        "//xla/service:custom_call_status_internal",
        "//xla/service:executable",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_execution_profile",
        "//xla/service:hlo_parser",
        "//xla/service:logical_buffer",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/service:xla_debug_info_manager",
        "//xla/service/gpu/runtime:kernel_launch",
        "//xla/service/gpu/runtime:collectives",
        "//xla/service/gpu/runtime:cublas_lt_matmul",
        "//xla/service/gpu/runtime:support",
        "//xla/service/gpu/runtime:executable",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/gpu:asm_compiler",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/tsl/platform:casts",
        "//tensorflow/tsl/profiler/lib:scoped_annotation",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/lib/gtl:map_util",
    ] + if_gpu_is_configured([
        ":cholesky_thunk",
        ":precompiled_kernels",
        ":triangular_solve_thunk",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_stream",
        "//tensorflow/tsl/platform/default/build_config:cublas_plugin",
        "//tensorflow/tsl/platform/default/build_config:cudnn_plugin",
        "//tensorflow/tsl/platform/default/build_config:cufft_plugin",
        "//xla/stream_executor/cuda:stream_executor_cuda",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:stream_executor_rocm",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":target_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:lhlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_parser",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/stream_executor",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:ArithDialect",
    ],
)

tf_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:test",
        "//xla/mlir_hlo:lhlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "cublas_cudnn",
    srcs = ["cublas_cudnn.cc"],
    hdrs = ["cublas_cudnn.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:statusor",
        "//xla/hlo/ir:hlo",
    ],
)

# TODO(ezhulenev): Extract `RunCholesky` into a separate library.
cc_library(
    name = "cholesky_thunk",
    srcs = if_gpu_is_configured(["cholesky_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cholesky_thunk.h"]),
    deps = if_gpu_is_configured([
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["//tensorflow/tsl/platform:status"],
)

# TODO(ezhulenev): Extract `RunTriangularSolve` into a separate library.
cc_library(
    name = "triangular_solve_thunk",
    srcs = if_gpu_is_configured(["triangular_solve_thunk.cc"]),
    hdrs = if_gpu_is_configured(["triangular_solve_thunk.h"]),
    deps = if_gpu_is_configured([
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["//tensorflow/tsl/platform:status"],
)

cc_library(
    name = "fft_thunk",
    srcs = ["fft_thunk.cc"],
    hdrs = ["fft_thunk.h"],
    deps = [
        ":buffer_allocations",
        ":thunk",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "gemm_rewriter",
    srcs = ["gemm_rewriter.cc"],
    hdrs = ["gemm_rewriter.h"],
    defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":ir_emission_utils",
        ":matmul_utils",
        "//tensorflow/tsl/platform:errors",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "//xla/stream_executor:blas",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gemm_thunk",
    srcs = ["gemm_thunk.cc"],
    hdrs = ["gemm_thunk.h"],
    deps = [
        ":matmul_utils",
        ":thunk",
        "//tensorflow/tsl/platform:logging",
        "//xla:status",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "cublas_lt_matmul_thunk",
    srcs = if_cuda_is_configured(["cublas_lt_matmul_thunk.cc"]),
    hdrs = if_cuda_is_configured(["cublas_lt_matmul_thunk.h"]),
    deps = if_cuda_is_configured([
        ":matmul_utils",
        ":thunk",
        "//xla/service:buffer_assignment",
        "//xla:status",
        "//tensorflow/tsl/platform/default/build_config:cublas_plugin",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/cuda:cublas_lt_header",
    ]) + ["//tensorflow/tsl/platform:logging"],
)

cc_library(
    name = "gemm_algorithm_picker",
    srcs = if_cuda_is_configured(["gemm_algorithm_picker.cc"]),
    hdrs = if_cuda_is_configured(["gemm_algorithm_picker.h"]),
    deps = if_cuda_is_configured([
        ":backend_configs_cc",
        ":buffer_comparator",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_conv_runner",
        ":ir_emission_utils",
        ":matmul_utils",
        ":stream_executor_util",
        "//xla:autotune_results_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla:status_macros",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor/cuda:cublas_lt_header",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor/gpu:redzone_allocator",
        "//xla:util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform/default/build_config:cublas_plugin",
        "//tensorflow/tsl/platform:logger",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:autotuning_proto_cc",
        "//tensorflow/tsl/util/proto:proto_utils",
    ]),
)

tf_cc_test(
    name = "gemm_algorithm_picker_test",
    srcs = ["gemm_algorithm_picker_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":gemm_algorithm_picker",
        ":gemm_rewriter",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "//xla/service:gpu_plugin",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "matmul_utils",
    srcs = ["matmul_utils.cc"],
    hdrs = ["matmul_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//tensorflow/tsl/platform:statusor",
        "//xla/stream_executor:stream_executor_headers",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cublas_lt_header",
        "//tensorflow/tsl/platform/default/build_config:cublas_plugin",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:scratch_allocator",
    ]),
)

tf_cc_test(
    name = "matmul_utils_test",
    srcs = ["matmul_utils_test.cc"],
    deps = [
        ":matmul_utils",
        "//tensorflow/tsl/platform:status_matchers",
        "//xla:test",
        "//xla/service:hlo_parser",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_conv_algorithm_picker",
    srcs = ["gpu_conv_algorithm_picker.cc"],
    hdrs = ["gpu_conv_algorithm_picker.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":gpu_asm_opts_util",
        ":gpu_autotuning_proto_cc",
        ":gpu_conv_runner",
        ":gpu_executable",
        ":hlo_algorithm_denylist",
        ":ir_emission_utils",
        ":stream_executor_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//xla:autotune_results_proto_cc",
        "//xla:literal_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service:compiler",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla:xla_data_proto_cc",
        "//tensorflow/tsl/platform:logger",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/protobuf:autotuning_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:scratch_allocator",
        "//tensorflow/tsl/util/proto:proto_utils",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:dnn_proto_cc",
        "//tensorflow/tsl/util:env_var",
    ] + if_cuda_is_configured([
        ":buffer_comparator",
        "@local_config_cuda//cuda:cudnn_header",
        "//xla/stream_executor/gpu:redzone_allocator",
    ]),
)

tf_cc_test(
    name = "gpu_conv_algorithm_picker_test",
    srcs = ["gpu_conv_algorithm_picker_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":gpu_conv_algorithm_picker",
        ":gpu_conv_rewriter",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//xla/service:gpu_plugin",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service:tuple_simplifier",
        "//xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "gpu_conv_runner",
    srcs = ["gpu_conv_runner.cc"],
    hdrs = ["gpu_conv_runner.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":stream_executor_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:lazy_op_runner",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_conv_rewriter",
    srcs = ["gpu_conv_rewriter.cc"],
    hdrs = ["gpu_conv_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

cc_library(
    name = "move_copy_to_users",
    srcs = ["move_copy_to_users.cc"],
    hdrs = ["move_copy_to_users.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "move_copy_to_users_test",
    srcs = [
        "move_copy_to_users_test.cc",
    ],
    deps = [
        ":move_copy_to_users",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_conv_rewriter_test",
    srcs = ["gpu_conv_rewriter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gpu_conv_rewriter",
        "//tensorflow/tsl/platform:test",
        "//xla:protobuf_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_matchers",
        "//xla/service:shape_inference",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "cusolver_context",
    srcs = if_gpu_is_configured(["cusolver_context.cc"]),
    hdrs = if_gpu_is_configured(["cusolver_context.h"]),
    deps = [
        "//xla:comparison_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla:xla_data_proto_cc",
        "//tensorflow/tsl/platform:logging",
        "//xla/stream_executor:blas",
        "//tensorflow/tsl/platform:status",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor/cuda:cusolver_lib",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/rocm:rocblas_wrapper",
        "//xla/stream_executor/rocm:rocsolver_wrapper",
        "//xla/stream_executor/rocm:hipsolver_wrapper",
    ]),
)

cc_library(
    name = "cusolver_rewriter",
    srcs = if_gpu_is_configured(["cusolver_rewriter.cc"]),
    hdrs = if_gpu_is_configured(["cusolver_rewriter.h"]),
    deps = if_gpu_is_configured([
        ":cusolver_context",
        ":ir_emission_utils",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/stream_executor",
        "//tensorflow/tsl/platform:logging",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/algorithm:container",
    ]) + ["//tensorflow/tsl/platform:status"],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":gpu_fusible",
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:fusion_node_indexing_evaluation",
        "//xla/service:hlo_query",
        "//xla/service:instruction_fusion",
        "//xla/service:pattern_matcher",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_fusible",
        ":instruction_fusion",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        ":instruction_fusion",
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:logging",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_reachability",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "multi_output_fusion_test",
    srcs = ["multi_output_fusion_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_fusible",
        ":multi_output_fusion",
        "//xla/service:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "softmax_fusion",
    srcs = ["softmax_fusion.cc"],
    hdrs = ["softmax_fusion.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "softmax_fusion_test",
    srcs = ["softmax_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":softmax_fusion",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//xla:status_macros",
        "//xla:test",
        "//xla:util",
        "//xla/service:gpu_plugin",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_sanitize_constant_names",
    srcs = ["gpu_sanitize_constant_names.cc"],
    hdrs = ["gpu_sanitize_constant_names.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service/llvm_ir:buffer_assignment_util",
    ],
)

tf_cc_test(
    name = "gpu_sanitize_constant_names_test",
    srcs = ["gpu_sanitize_constant_names_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_sanitize_constant_names",
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:test",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fusion_merger",
    srcs = ["fusion_merger.cc"],
    hdrs = ["fusion_merger.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_fusible",
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        ":instruction_fusion",
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:errors",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_pass",
        "//xla/service/llvm_ir:fused_ir_emitter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "fusion_merger_test",
    srcs = ["fusion_merger_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":fusion_merger",
        ":gpu_device_info",
        ":gpu_device_info_for_tests",
        ":gpu_fusible",
        ":instruction_fusion",
        "//xla:test_helpers",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_conv_padding_legalization",
    srcs = ["gpu_conv_padding_legalization.cc"],
    hdrs = ["gpu_conv_padding_legalization.h"],
    deps = [
        ":cublas_cudnn",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_conv_padding_legalization_test",
    srcs = ["gpu_conv_padding_legalization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gpu_conv_padding_legalization",
        "//tensorflow/tsl/platform:test",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "cudnn_support_utils",
    srcs = ["cudnn_support_utils.cc"],
    hdrs = ["cudnn_support_utils.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/tsl/platform:status",
        "//xla:comparison_util",
        "//xla:util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:stream_executor_headers",
    ],
)

tf_cc_test(
    name = "cudnn_support_utils_test",
    srcs = ["cudnn_support_utils_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cudnn_support_utils",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//xla:status_macros",
        "//xla:test",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cudnn_pad_for_convolutions",
    srcs = ["cudnn_pad_for_convolutions.cc"],
    hdrs = ["cudnn_pad_for_convolutions.h"],
    deps = [
        ":cudnn_support_utils",
        ":ir_emission_utils",
        ":stream_executor_util",
        "//tensorflow/tsl/platform:status",
        "//xla:literal_util",
        "//xla:util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

tf_cc_test(
    name = "cudnn_pad_for_convolutions_test",
    srcs = ["cudnn_pad_for_convolutions_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":cudnn_pad_for_convolutions",
        "//tensorflow/tsl/platform:test",
        "//xla:status_macros",
        "//xla:test",
        "//xla:util",
        "//xla/service:hlo_parser",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "cudnn_vectorize_convolutions",
    srcs = ["cudnn_vectorize_convolutions.cc"],
    hdrs = ["cudnn_vectorize_convolutions.h"],
    deps = [
        ":cudnn_support_utils",
        ":stream_executor_util",
        "//xla:statusor",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service:call_inliner",
        "//xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "cudnn_vectorize_convolutions_test",
    srcs = ["cudnn_vectorize_convolutions_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":cudnn_vectorize_convolutions",
        "//tensorflow/tsl/platform:statusor",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service:call_inliner",
        "//xla/service:hlo_parser",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "cudnn_simplify_padding",
    srcs = ["cudnn_simplify_padding.cc"],
    hdrs = ["cudnn_simplify_padding.h"],
    deps = [
        ":cublas_cudnn",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
    ],
)

tf_cc_test(
    name = "cudnn_simplify_padding_test",
    srcs = ["cudnn_simplify_padding_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":cudnn_pad_for_convolutions",
        ":cudnn_simplify_padding",
        ":cudnn_vectorize_convolutions",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service:algebraic_simplifier",
        "//xla/service:call_inliner",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service:tuple_simplifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "cublas_pad_for_gemms",
    srcs = ["cublas_pad_for_gemms.cc"],
    hdrs = ["cublas_pad_for_gemms.h"],
    deps = [
        ":ir_emission_utils",
        "//xla:literal_util",
        "//xla:util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "cublas_pad_for_gemms_test",
    srcs = ["cublas_pad_for_gemms_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":cublas_pad_for_gemms",
        ":ir_emission_utils",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

tf_proto_library(
    name = "executable_proto",
    srcs = ["executable.proto"],
    cc_api_version = 2,
    protodeps = [
        "//xla/service:hlo_proto",
    ],
)

cc_library(
    name = "target_constants",
    hdrs = ["target_constants.h"],
)

cc_library(
    name = "gpu_transfer_manager",
    srcs = ["gpu_transfer_manager.cc"],
    hdrs = ["gpu_transfer_manager.h"],
    deps = [
        ":io_feed_manager",
        ":target_constants",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:compiler",
        "//xla/service:generic_transfer_manager",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "@com_google_absl//absl/cleanup",
        "@llvm-project//llvm:Core",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "gpu_reduce_scatter_creator",
    srcs = ["gpu_reduce_scatter_creator.cc"],
    hdrs = ["gpu_reduce_scatter_creator.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_query",
        "//xla/service:reduce_scatter_utils",
    ],
)

cc_library(
    name = "gpu_compiler",
    srcs = [
        "gpu_compiler.cc",
    ],
    hdrs = [
        "gpu_compiler.h",
    ],
    deps = [
        ":alias_passthrough_params",
        ":all_reduce_blueconnect",
        ":all_reduce_promotion",
        ":conv_layout_normalization",
        ":executable_proto_cc",
        ":fusion_merger",
        ":gemm_broadcast_folding_rewriter",
        ":gemm_rewriter",
        ":gpu_constants",
        ":gpu_conv_algorithm_picker",
        ":gpu_conv_rewriter",
        ":gpu_device_info",
        ":gpu_executable",
        ":gpu_hlo_cost_analysis",
        ":gpu_hlo_schedule",
        ":gpu_layout_assignment",
        ":gpu_reduce_scatter_creator",
        ":gpu_sanitize_constant_names",
        ":gpu_scatter_expander",
        ":gpu_shape_verifier",
        ":hlo_fusion_stats",
        ":horizontal_input_fusion",
        ":horizontal_loop_fusion",
        ":instruction_fusion",
        ":ir_emission_utils",
        ":ir_emitter",
        ":ir_emitter_context",
        ":matmul_utils",
        ":metrics",
        ":move_copy_to_users",
        ":multi_output_fusion",
        ":reduction_degenerate_dim_remover",
        ":reduction_dimension_grouper",
        ":reduction_layout_normalizer",
        ":reduction_splitter",
        ":runtime_intrinsics",
        ":scatter_slice_simplifier",
        ":softmax_fusion",
        ":tree_reduction_rewriter",
        ":variadic_op_splitter",
        "//tensorflow/tsl/platform:blocking_counter",
        "//tensorflow/tsl/platform:casts",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:hlo_constant_splitter",
        "//xla/mlir/backends/gpu/transforms:passes",
        "//xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//xla/mlir_hlo:transforms_gpu_passes",
        "//xla/runtime:jit_executable",
        "//xla/service:algebraic_simplifier",
        "//xla/service:all_gather_broadcast_reorder",
        "//xla/service:all_gather_combiner",
        "//xla/service:all_reduce_combiner",
        "//xla/service:all_reduce_contiguous",
        "//xla/service:all_reduce_folder",
        "//xla/service:all_reduce_reassociate",
        "//xla/service:all_to_all_decomposer",
        "//xla/service:async_collective_creator",
        "//xla/service:batchnorm_expander",
        "//xla/service:bfloat16_normalization",
        "//xla/service:bitcast_dtypes_expander",
        "//xla/service:broadcast_canonicalizer",
        "//xla/service:buffer_assignment",
        "//xla/service:call_inliner",
        "//xla/service:collectives_schedule_linearizer",
        "//xla/service:comparison_expander",
        "//xla/service:conditional_canonicalizer",
        "//xla/service:conditional_simplifier",
        "//xla/service:convert_mover",
        "//xla/service:convolution_4d_expander",
        "//xla/service:convolution_pred_expander",
        "//xla/service:copy_insertion",
        "//xla/service:dot_decomposer",
        "//xla/service:dot_merger",
        "//xla/service:dump",
        "//xla/service:dynamic_dimension_simplifier",
        "//xla/service:dynamic_index_splitter",
        "//xla/service:dynamic_padder",
        "//xla/service:eigh_expander",
        "//xla/service:executable",
        "//xla/service:flatten_call_graph",
        "//xla/service:gather_expander",
        "//xla/service:gather_simplifier",
        "//xla/service:hlo_computation_deduplicator",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_verifier",
        "//xla/service:layout_normalization",
        "//xla/service:llvm_compiler",
        "//xla/service:logistic_expander",
        "//xla/service:loop_schedule_linearizer",
        "//xla/service:operand_upcaster",
        "//xla/service:optimization_barrier_expander",
        "//xla/service:qr_expander",
        "//xla/service:real_imag_expander",
        "//xla/service:reduce_decomposer",
        "//xla/service:reduce_scatter_combiner",
        "//xla/service:reduce_scatter_reassociate",
        "//xla/service:reshape_decomposer",
        "//xla/service:reshape_mover",
        "//xla/service:result_caster",
        "//xla/service:rng_bit_generator_expander",
        "//xla/service:rng_expander",
        "//xla/service:scatter_simplifier",
        "//xla/service:sharding_propagation",
        "//xla/service:sharding_remover",
        "//xla/service:simplify_fp_conversions",
        "//xla/service:slow_operation_alarm",
        "//xla/service:sort_simplifier",
        "//xla/service:stable_sort_expander",
        "//xla/service:stochastic_convert_decomposer",
        "//xla/service:transpose_folding",
        "//xla/service:tuple_simplifier",
        "//xla/service:while_loop_all_reduce_code_motion",
        "//xla/service:while_loop_constant_sinking",
        "//xla/service:while_loop_simplifier",
        "//xla/service:while_loop_trip_count_annotator",
        "//xla/service:zero_sized_hlo_elimination",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/spmd:collective_permute_motion",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "//xla/stream_executor",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "//xla/translate/mhlo_to_hlo:location_exporter",
        "//xla/translate/mhlo_to_lhlo_with_xla",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "gpu_compiler_test",
    srcs = ["gpu_compiler_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":horizontal_loop_fusion",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "nvptx_compiler",
    srcs = if_cuda_is_configured([
        "nvptx_compiler_registration.cc",
    ]),
    deps = if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_platform_id",
        ":nvptx_compiler_impl",
        "//tensorflow/tsl/platform:path",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "nvptx_helper",
    srcs = ["nvptx_helper.cc"],
    hdrs = ["nvptx_helper.h"],
    deps = [
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "nvptx_compiler_impl",
    srcs = if_cuda_is_configured([
        "nvptx_compiler.cc",
    ]),
    hdrs = if_cuda_is_configured([
        "nvptx_compiler.h",
    ]),
    deps = if_cuda_is_configured([
        ":cublas_cudnn",
        ":cublas_pad_for_gemms",
        ":cudnn_fused_conv_rewriter",
        ":cudnn_pad_for_convolutions",
        ":cudnn_simplify_padding",
        ":cudnn_vectorize_convolutions",
        ":cusolver_rewriter",
        ":gemm_algorithm_picker",
        ":gpu_asm_opts_util",
        ":gpu_compiler",
        ":gpu_conv_padding_legalization",
        ":gpu_conv_rewriter",
        ":gpu_executable",
        ":gpu_layout_assignment",
        ":ir_emission_utils",
        ":metrics",
        ":nvptx_helper",
        ":target_constants",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/container:node_hash_map",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "//xla/service:algebraic_simplifier",
        "//xla/service:call_inliner",
        "//xla/service:dump",
        "//xla/service/gpu/llvm_gpu_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_verifier",
        "//xla/service:llvm_compiler",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service:tuple_simplifier",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//xla/stream_executor/cuda:cuda_diagnostics",
        "//xla/stream_executor/gpu:asm_compiler",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/util:env_var",
        ":triangular_solve_rewriter",
    ]),
)

tf_cc_test(
    name = "nvptx_compiler_test",
    srcs = if_gpu_is_configured([
        "nvptx_compiler_test.cc",
    ]),
    tags = [
        "gpu",
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false
        # positives in msan.
    ],
    deps = [
        ":nvptx_compiler_impl",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

tf_cc_test(
    name = "gpu_aot_compilation_test",
    srcs = if_cuda_is_configured([
        "gpu_aot_compilation_test.cc",
    ]),
    env = {
        "XLA_FLAGS": "--xla_gpu_enable_xla_runtime_executable",
    },
    tags = [
        "gpu",
        "no_oss",
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false positives in msan.
        "requires-gpu-nvidia",
    ],
    deps = if_cuda_is_configured([
        ":nvptx_compiler_impl",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ]),
)

cc_library(
    name = "amdgpu_compiler",
    srcs = if_rocm_is_configured([
        "amdgpu_compiler_registration.cc",
    ]),
    deps = if_rocm_is_configured([
        ":amdgpu_compiler_impl",
        "//xla/stream_executor/host:host_platform_id",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "amdgpu_compiler_impl",
    srcs = if_rocm_is_configured([
        "amdgpu_compiler.cc",
    ]),
    hdrs = if_rocm_is_configured([
        "amdgpu_compiler.h",
    ]),
    deps = if_rocm_is_configured([
        ":cusolver_rewriter",
        ":gemm_rewriter",
        ":gpu_compiler",
        ":gpu_conv_algorithm_picker",
        ":gpu_conv_padding_legalization",
        ":gpu_conv_rewriter",
        ":gpu_layout_assignment",
        ":reduction_degenerate_dim_remover",
        ":reduction_dimension_grouper",
        ":reduction_layout_normalizer",
        ":target_constants",
        ":tree_reduction_rewriter",
        ":triangular_solve_rewriter",
        "//xla:statusor",
        "//xla/service:algebraic_simplifier",
        "//xla/service:call_inliner",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service:tuple_simplifier",
        "//xla/service/gpu/llvm_gpu_backend",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/tsl/platform:rocm_rocdl_path",
    ]),
)

cc_library(
    name = "all_reduce_blueconnect",
    srcs = ["all_reduce_blueconnect.cc"],
    hdrs = ["all_reduce_blueconnect.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_query",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "all_reduce_blueconnect_test",
    srcs = ["all_reduce_blueconnect_test.cc"],
    deps = [
        ":all_reduce_blueconnect",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test_main",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
    ],
)

cc_library(
    name = "all_reduce_promotion",
    srcs = ["all_reduce_promotion.cc"],
    hdrs = ["all_reduce_promotion.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:change_op_data_type",
        "//xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "all_reduce_promotion_test",
    srcs = ["all_reduce_promotion_test.cc"],
    deps = [
        ":all_reduce_promotion",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test_main",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
    ],
)

cc_library(
    name = "xfeed_queue",
    hdrs = ["xfeed_queue.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "io_feed_manager",
    srcs = [
        "infeed_manager.cc",
        "outfeed_manager.cc",
        "xla_executor_state.h",
    ],
    hdrs = [
        "infeed_manager.h",
        "outfeed_manager.h",
    ],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":xfeed_queue",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:notification",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "gpu_shape_verifier",
    srcs = ["gpu_shape_verifier.cc"],
    hdrs = ["gpu_shape_verifier.h"],
    deps = [
        "//xla/service:hlo_verifier",
    ],
)

cc_library(
    name = "gpu_layout_assignment",
    srcs = ["gpu_layout_assignment.cc"],
    hdrs = ["gpu_layout_assignment.h"],
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        ":matmul_utils",
        ":stream_executor_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:layout_assignment",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "gpu_layout_assignment_test",
    srcs = ["gpu_layout_assignment_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gemm_rewriter",
        ":gpu_layout_assignment",
        "//tensorflow/tsl/platform:status_matchers",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/stream_executor/lib",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_schedule",
    srcs = ["gpu_hlo_schedule.cc"],
    hdrs = ["gpu_hlo_schedule.h"],
    deps = [
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_value",
        "//xla/service:hlo_memory_scheduler",
        "//xla/service:hlo_ordering",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_hlo_schedule_test",
    srcs = [
        "gpu_hlo_schedule_test.cc",
    ],
    tags = ["no_pip"],
    deps = [
        ":gpu_hlo_schedule",
        "//xla:test_helpers",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "while_transformer_test",
    srcs = ["while_transformer_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":instruction_fusion",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/service:copy_insertion",
        "//xla/service:hlo_verifier",
        "//xla/service:while_loop_analysis",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "stream_executor_util",
    srcs = ["stream_executor_util.cc"],
    hdrs = ["stream_executor_util.h"],
    copts = tsl_copts(),
    deps = [
        ":cublas_cudnn",
        ":launch_dimensions",
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/protobuf:autotuning_proto_cc",
        "//tensorflow/tsl/util:determinism_for_kernels",
        "//tensorflow/tsl/util:env_var",
        "//tensorflow/tsl/util/proto:proto_utils",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor",
        "//xla/stream_executor:kernel_spec",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_asm_opts_util",
    srcs = ["gpu_asm_opts_util.cc"],
    hdrs = ["gpu_asm_opts_util.h"],
    copts = tsl_copts(),
    deps = [
        "//xla:xla_proto_cc",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_cost_analysis",
    srcs = ["gpu_hlo_cost_analysis.cc"],
    hdrs = ["gpu_hlo_cost_analysis.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//xla/hlo/ir:hlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service:hlo_cost_analysis",
    ],
)

tf_cc_test(
    name = "gpu_hlo_cost_analysis_test",
    srcs = ["gpu_hlo_cost_analysis_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "gpu_performance_model",
    srcs = ["gpu_performance_model.cc"],
    hdrs = ["gpu_performance_model.h"],
    deps = [
        ":gpu_device_info",
        ":gpu_hlo_cost_analysis",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "gpu_performance_model_test",
    srcs = ["gpu_performance_model_test.cc"],
    deps = [
        ":gpu_device_info_for_tests",
        ":gpu_performance_model",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "buffer_comparator",
    srcs = if_cuda_is_configured(["buffer_comparator.cc"]),
    hdrs = if_cuda_is_configured(["buffer_comparator.h"]),
    deps = if_cuda_is_configured([
        ":launch_dimensions",
        ":gpu_asm_opts_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:asm_compiler",
    ]),
)

tf_cc_test(
    name = "buffer_comparator_test",
    srcs = if_cuda_is_configured(["buffer_comparator_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/tsl/platform:test_main",
        "//xla:shape_util",
        "//xla:types",
        "//tensorflow/tsl/platform:test",
    ] + if_cuda_is_configured([
        ":buffer_comparator",
        "//xla/stream_executor/cuda:stream_executor_cuda",
        "//xla/stream_executor:device_memory",
    ]),
)

cc_library(
    name = "gpu_fusible",
    srcs = ["gpu_fusible.cc"],
    hdrs = ["gpu_fusible.h"],
    deps = [
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:instruction_fusion",
    ],
)

tf_cc_test(
    name = "gpu_fusible_test",
    srcs = ["gpu_fusible_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_fusible",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cudnn_fused_conv_rewriter",
    srcs = ["cudnn_fused_conv_rewriter.cc"],
    hdrs = ["cudnn_fused_conv_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn_proto_cc",
    ],
)

tf_cc_test(
    name = "cudnn_fused_conv_rewriter_test",
    srcs = ["cudnn_fused_conv_rewriter_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":cudnn_fused_conv_rewriter",
        ":gpu_conv_rewriter",
        ":ir_emission_utils",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//xla:test_helpers",
        "//xla/service:algebraic_simplifier",
        "//xla/service:convert_mover",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service:reshape_mover",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "conv_layout_normalization_test",
    srcs = ["conv_layout_normalization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//xla:test_helpers",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service:reshape_mover",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "variadic_op_splitter",
    srcs = ["variadic_op_splitter.cc"],
    hdrs = ["variadic_op_splitter.h"],
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_scatter_expander",
    srcs = ["gpu_scatter_expander.cc"],
    hdrs = ["gpu_scatter_expander.h"],
    deps = [
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:scatter_expander",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "variadic_op_splitter_test",
    srcs = ["variadic_op_splitter_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":ir_emission_utils",
        ":variadic_op_splitter",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:pattern_matcher",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

tf_proto_library(
    name = "gpu_autotuning_proto",
    srcs = ["gpu_autotuning.proto"],
    cc_api_version = 2,
    protodeps = [
        "//xla:xla_data_proto",
        "//xla/service:hlo_proto",
        "//tensorflow/tsl/protobuf:autotuning_proto",
    ],
)

cc_library(
    name = "hlo_algorithm_denylist",
    srcs = ["hlo_algorithm_denylist.cc"],
    hdrs = ["hlo_algorithm_denylist.h"],
    deps = [
        ":gpu_autotuning_proto_cc",
        "//tensorflow/tsl/protobuf:autotuning_proto_cc",
        "//xla:debug_options_flags",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "hlo_algorithm_denylist_test",
    srcs = ["hlo_algorithm_denylist_test.cc"],
    data = ["data/hlo_algorithm_denylist.pbtxt"],
    tags = ["no_pip"],
    deps = [
        ":hlo_algorithm_denylist",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:resource_loader",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//xla/stream_executor:dnn",
    ],
)

cc_library(
    name = "alias_passthrough_params",
    srcs = ["alias_passthrough_params.cc"],
    hdrs = ["alias_passthrough_params.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "alias_passthrough_params_test",
    srcs = ["alias_passthrough_params_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":alias_passthrough_params",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "horizontal_loop_fusion",
    srcs = ["horizontal_loop_fusion.cc"],
    hdrs = ["horizontal_loop_fusion.h"],
    deps = [
        ":gpu_fusible",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "horizontal_loop_fusion_test",
    srcs = ["horizontal_loop_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":fusion_merger",
        ":horizontal_loop_fusion",
        ":instruction_fusion",
        ":multi_output_fusion",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:tuple_simplifier",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "horizontal_input_fusion",
    srcs = ["horizontal_input_fusion.cc"],
    hdrs = ["horizontal_input_fusion.h"],
    deps = [
        ":gpu_fusible",
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "horizontal_input_fusion_test",
    srcs = ["horizontal_input_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":horizontal_input_fusion",
        ":multi_output_fusion",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/tests:filecheck",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "reduction_degenerate_dim_remover",
    srcs = ["reduction_degenerate_dim_remover.cc"],
    hdrs = ["reduction_degenerate_dim_remover.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduction_dimension_grouper",
    srcs = ["reduction_dimension_grouper.cc"],
    hdrs = ["reduction_dimension_grouper.h"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "reduction_splitter",
    srcs = ["reduction_splitter.cc"],
    hdrs = ["reduction_splitter.h"],
    deps = [
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "reduction_splitter_test",
    srcs = ["reduction_splitter_test.cc"],
    deps = [
        ":reduction_splitter",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "reduction_layout_normalizer",
    srcs = ["reduction_layout_normalizer.cc"],
    hdrs = ["reduction_layout_normalizer.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tree_reduction_rewriter",
    srcs = ["tree_reduction_rewriter.cc"],
    hdrs = ["tree_reduction_rewriter.h"],
    deps = [
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gemm_broadcast_folding_rewriter",
    srcs = ["gemm_broadcast_folding_rewriter.cc"],
    hdrs = ["gemm_broadcast_folding_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/tsl/platform:errors",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
    ],
)

# These tests are intended to be run with --test_env=XLA_FLAGS=--xla_gpu_enable_xla_runtime_executable
# See tap/tensorflow.xla_gpu_jitrt.
test_suite(
    name = "jitrt_executable_tests",
    tests = [
        # copybara:uncomment "//platforms/xla/tests/internal:xfeed_test_gpu",
        # copybara:uncomment "//third_party/py/jax/experimental/jax2tf/tests:primitives_test_gpu",
        # copybara:uncomment "//third_party/py/jax/tests:pmap_test_gpu",
        "//third_party/tensorflow/compiler/tests:fft_test_gpu",
        "//xla/python:xla_client_test_gpu",
        "//xla/service/gpu:cudnn_fused_conv_rewriter_test",
        "//xla/service/gpu:custom_call_test",
        # copybara:uncomment "//xla/service/gpu:gpu_aot_compilation_test",
        "//xla/service/gpu/tests:add_preds.hlo.test",
        "//xla/service/gpu/tests:all_reduce.hlo.test",
        "//xla/service/gpu/tests:concat.hlo.test",
        "//xla/service/gpu/tests:constant.hlo.test",
        "//xla/service/gpu/tests:copy.hlo.test",
        "//xla/service/gpu/tests:copy_nested.hlo.test",
        "//xla/service/gpu/tests:dynamic_update_slice_inplace.hlo.test",
        "//xla/service/gpu/tests:element_wise_row_vectorization.hlo.test",
        "//xla/service/gpu/tests:element_wise_row_vectorization_test",
        "//xla/service/gpu/tests:fused_scatter.hlo.test",
        "//xla/service/gpu/tests:fused_slice.hlo.test",
        "//xla/service/gpu/tests:fused_slice_different_operands.hlo.test",
        "//xla/service/gpu/tests:fusion.hlo.test",
        "//xla/service/gpu/tests:gemm_broadcast_folding_rewrite_test",
        "//xla/service/gpu/tests:gemm_rewrite_test",
        "//xla/service/gpu/tests:gpu_alignment_test",
        "//xla/service/gpu/tests:gpu_atomic_test",
        "//xla/service/gpu/tests:gpu_compilation_parallelism_test",
        "//xla/service/gpu/tests:gpu_convolution_regression_test",
        "//xla/service/gpu/tests:gpu_copy_alone_test",
        "//xla/service/gpu/tests:gpu_copy_test",
        "//xla/service/gpu/tests:gpu_dyn_shape_test",
        "//xla/service/gpu/tests:gpu_ftz_test",
        "//xla/service/gpu/tests:gpu_fusion_test",
        "//xla/service/gpu/tests:gpu_index_test",
        "//xla/service/gpu/tests:gpu_infeed_test",
        "//xla/service/gpu/tests:gpu_input_fusible_slice_test",
        "//xla/service/gpu/tests:gpu_kernel_tiling_test",
        "//xla/service/gpu/tests:gpu_ldg_test",
        "//xla/service/gpu/tests:gpu_noalias_test",
        "//xla/service/gpu/tests:gpu_reduce_scatter_creator_test",
        "//xla/service/gpu/tests:gpu_spmd_e2e_compile_test",
        "//xla/service/gpu/tests:gpu_too_many_blocks_test",
        "//xla/service/gpu/tests:gpu_unrolling_test",
        "//xla/service/gpu/tests:in_place_op_test",
        "//xla/service/gpu/tests:kernel_launch_test",
        "//xla/service/gpu/tests:launch_dimensions.hlo.test",
        "//xla/service/gpu/tests:mlir_fft_test",
        "//xla/service/gpu/tests:mlir_gemm_test",
        "//xla/service/gpu/tests:mlir_gpu_compile_test",
        "//xla/service/gpu/tests:mlir_sorting_test",
        "//xla/service/gpu/tests:pad_to_static.hlo.test",
        "//xla/service/gpu/tests:parallel_reduction_test",
        "//xla/service/gpu/tests:pred_arithmetic_test",
        "//xla/service/gpu/tests:reduce_unnested.hlo.test",
        "//xla/service/gpu/tests:reduction_degenerate_dim_remover_test",
        "//xla/service/gpu/tests:reduction_dimension_grouper_test",
        "//xla/service/gpu/tests:reduction_layout_normalizer_test",
        "//xla/service/gpu/tests:reduction_vectorization_sm_all.hlo.test",
        "//xla/service/gpu/tests:reduction_vectorization_test",
        "//xla/service/gpu/tests:rng_get_and_update_state.hlo.test",
        "//xla/service/gpu/tests:scatter.hlo.test",
        "//xla/service/gpu/tests:select_and_scatter.hlo.test",
        "//xla/service/gpu/tests:select_and_scatter_test",
        "//xla/service/gpu/tests:single_instruction.hlo.test",
        "//xla/service/gpu/tests:slice_to_dynamic.hlo.test",
        "//xla/service/gpu/tests:sorting.hlo.test",
        "//xla/service/gpu/tests:sorting_test",
        "//xla/service/gpu/tests:swap_conv_operands_test",
        "//xla/service/gpu/tests:tree_reduction_rewriter_test",
        "//xla/tests:all_reduce_test_gpu",
        "//xla/tests:array_elementwise_ops_test_gpu",
        "//xla/tests:axpy_simple_test_gpu",
        "//xla/tests:bad_rng_shape_validation_test_gpu",
        "//xla/tests:batch_normalization_test_gpu",
        "//xla/tests:bfloat16_test_gpu",
        "//xla/tests:binop_scaling_test_gpu",
        "//xla/tests:bitcast_convert_test_gpu",
        "//xla/tests:broadcast_simple_test_gpu",
        "//xla/tests:broadcast_test_gpu",
        "//xla/tests:buffer_donation_test_gpu",
        "//xla/tests:call_test_gpu",
        "//xla/tests:check_execution_arity_test_gpu",
        "//xla/tests:cholesky_test_gpu",
        "//xla/tests:client_test_gpu",
        "//xla/tests:compilation_cache_test_gpu",
        "//xla/tests:compute_constant_test_gpu",
        "//xla/tests:concat_test_gpu",
        "//xla/tests:conditional_test_gpu",
        "//xla/tests:constant_reduction_function_test_gpu",
        "//xla/tests:constants_test_gpu",
        "//xla/tests:conv_depthwise_backprop_filter_test_gpu",
        "//xla/tests:conv_depthwise_test_gpu",
        "//xla/tests:convert_test_gpu",
        "//xla/tests:convolution_dimension_numbers_test_gpu",
        "//xla/tests:convolution_test_1d_autotune_disabled_gpu",
        "//xla/tests:convolution_test_1d_gpu_alternative_layout_gpu",
        "//xla/tests:convolution_test_1d_no_vmodule_gpu",
        "//xla/tests:convolution_test_autotune_disabled_gpu",
        "//xla/tests:convolution_test_cudnn_frontend_disabled_gpu",
        "//xla/tests:convolution_test_gpu",
        "//xla/tests:convolution_test_gpu_alternative_layout_gpu",
        "//xla/tests:convolution_variants_test_gpu",
        "//xla/tests:copy_test_gpu",
        "//xla/tests:cpu_gpu_fusion_test_gpu",
        "//xla/tests:deallocation_test_gpu",
        "//xla/tests:deconstruct_tuple_test_gpu",
        "//xla/tests:deep_graph_test_gpu",
        "//xla/tests:dot_operation_single_threaded_runtime_test_gpu",
        "//xla/tests:dot_operation_test_autotune_disabled_gpu",
        "//xla/tests:dot_operation_test_gpu",
        "//xla/tests:dynamic_ops_test_gpu",
        "//xla/tests:execution_profile_test_gpu",
        "//xla/tests:execution_profile_test_with_xla_hlo_profile_gpu",
        "//xla/tests:floor_ceil_test_gpu",
        "//xla/tests:fmax_fmin_test_gpu",
        "//xla/tests:gather_operation_test_gpu",
        "//xla/tests:get_dimension_size_test_gpu",
        "//xla/tests:grouped_convolution_test_gpu",
        "//xla/tests:half_test_gpu",
        "//xla/tests:iota_test_gpu",
        "//xla/tests:local_client_allocation_test_gpu",
        "//xla/tests:local_client_execute_test_gpu",
        "//xla/tests:log_test_gpu",
        "//xla/tests:map_test_gpu",
        "//xla/tests:matmul_test_gpu",
        "//xla/tests:matrix_ops_simple_test_gpu",
        "//xla/tests:multidimensional_slice_test_gpu",
        "//xla/tests:multioutput_fusion_test_gpu",
        "//xla/tests:outfeed_in_nested_computation_test_gpu",
        "//xla/tests:pad_test_gpu",
        "//xla/tests:params_test_gpu",
        "//xla/tests:pred_test_gpu",
        "//xla/tests:prng_test_gpu",
        "//xla/tests:ptxas_bug_120501638_gpu",
        "//xla/tests:query_inferred_shape_test_gpu",
        "//xla/tests:reduce_hlo_test_gpu",
        "//xla/tests:reduce_precision_test_gpu",
        "//xla/tests:reduce_test_gpu",
        "//xla/tests:reduce_window_test_gpu",
        "//xla/tests:replay_test_gpu",
        "//xla/tests:reshape_motion_test_gpu",
        "//xla/tests:reshape_test_gpu",
        "//xla/tests:reverse_test_gpu",
        "//xla/tests:round_trip_packed_literal_test_gpu",
        "//xla/tests:round_trip_transfer_test_gpu",
        "//xla/tests:sample_text_test_gpu",
        "//xla/tests:scalar_computations_test_gpu",
        "//xla/tests:scatter_test_gpu",
        "//xla/tests:select_and_scatter_test_gpu",
        "//xla/tests:select_test_gpu",
        "//xla/tests:slice_test_gpu",
        "//xla/tests:token_hlo_test_gpu",
        "//xla/tests:transfer_manager_test_gpu",
        "//xla/tests:transpose_test_gpu",
        "//xla/tests:triangular_solve_test_gpu",
        "//xla/tests:tuple_test_gpu",
        "//xla/tests:unary_op_test_gpu",
        "//xla/tests:value_inference_test_gpu",
        "//xla/tests:vector_ops_reduce_test_gpu",
        "//xla/tests:vector_ops_simple_test_gpu",
        "//xla/tests:while_test_gpu",
        "//xla/tests:xla_hlo_profile_test_gpu",
    ] + if_google([
        # Currently fails in OSS.
        "//third_party/tensorflow/python/kernel_tests/signal:fft_ops_test_xla_gpu",
    ]),
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/tsl/lib/monitoring:sampler",
    ],
)

cc_library(
    name = "precompiled_kernels",
    srcs = if_gpu_is_configured(["precompiled_kernels.cc"]),
    hdrs = if_gpu_is_configured(["precompiled_kernels.h"]),
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:asm_compiler",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/gpu:gpu_stream_header",
    ]),
)

cc_library(
    name = "triangular_solve_rewriter",
    srcs = ["triangular_solve_rewriter.cc"],
    hdrs = ["triangular_solve_rewriter.h"],
    deps = [
        ":cublas_cudnn",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/strings",
    ],
)

tsl_gpu_library(
    name = "runtime_intrinsics",
    srcs = ["runtime_intrinsics.cc"],
    hdrs = ["runtime_intrinsics.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "@com_google_absl//absl/cleanup",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_fusion_stats",
    srcs = ["hlo_fusion_stats.cc"],
    hdrs = ["hlo_fusion_stats.h"],
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "//xla:status",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_fusion_stats_test",
    srcs = ["hlo_fusion_stats_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":hlo_fusion_stats",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test",
        "//xla:status_macros",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "scatter_slice_simplifier",
    srcs = ["scatter_slice_simplifier.cc"],
    hdrs = ["scatter_slice_simplifier.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "scatter_slice_simplifier_test",
    srcs = ["scatter_slice_simplifier_test.cc"],
    deps = [
        ":scatter_slice_simplifier",
        "//xla/service:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "conv_layout_normalization",
    srcs = ["conv_layout_normalization.cc"],
    hdrs = ["conv_layout_normalization.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:padding",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)
