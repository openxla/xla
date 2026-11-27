# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter_cuda",
    srcs = ["ptx_custom_kernel_emitter_cuda.cc"],
    hdrs = ["ptx_custom_kernel_emitter.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":kernel_call",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/codegen/kernels:ptx_custom_kernel",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:gpu_constants",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:launch_dimensions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter_rocm",
    srcs = ["ptx_custom_kernel_emitter_rocm.cc"],
    hdrs = ["ptx_custom_kernel_emitter.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emitter_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter_sycl",
    srcs = ["ptx_custom_kernel_emitter_sycl_stub.cc"],
    hdrs = ["ptx_custom_kernel_emitter.h"],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emitter_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ptx_custom_kernel_emitter",
    hdrs = ["ptx_custom_kernel_emitter.h"],
    tags = ["gpu"],
    deps = if_cuda_is_configured([
        ":ptx_custom_kernel_emitter_cuda",
    ]) + if_rocm_is_configured([
        ":ptx_custom_kernel_emitter_rocm",
    ]) + if_sycl_is_configured([
        ":ptx_custom_kernel_emitter_sycl",
    ]) + ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "thunk_emitter",
    srcs = ["thunk_emitter.cc"],
    hdrs = ["thunk_emitter.h"],
    tags = ["gpu"],
    deps = [
        ":execution_stream_assignment",
        ":gpu_hlo_ordering",
        ":ptx_custom_kernel_emitter",
        ":triton_call",
        "//xla:comparison_util",
        "//xla:future",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen:kernel_compiler",
        "//xla/backends/gpu/codegen:pad_to_static",
        "//xla/backends/gpu/codegen:rng_get_and_update_state",
        "//xla/backends/gpu/codegen:slice_to_dynamic",
        "//xla/backends/gpu/codegen:sort",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/codegen/kernels:ptx_custom_kernel",
        "//xla/backends/gpu/codegen/triton:triton_kernel_source",
        "//xla/backends/gpu/codegen/triton:triton_wrapper_result",
        "//xla/backends/gpu/codegen/triton:xtile_compiler",
        "//xla/backends/gpu/ffi:ffi_attributes_from_backend_config",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_copy",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_fusion",
        "//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_emitter_context",
        "//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_handler_registry",
        "//xla/backends/gpu/runtime:all_gather_thunk",
        "//xla/backends/gpu/runtime:all_reduce_thunk",
        "//xla/backends/gpu/runtime:all_to_all_thunk",
        "//xla/backends/gpu/runtime:async_execution",
        "//xla/backends/gpu/runtime:async_thunk",
        "//xla/backends/gpu/runtime:collective_broadcast_thunk",
        "//xla/backends/gpu/runtime:collective_group_thunk",
        "//xla/backends/gpu/runtime:collective_permute_thunk",
        "//xla/backends/gpu/runtime:collective_reduce_thunk",
        "//xla/backends/gpu/runtime:collective_thunk",
        "//xla/backends/gpu/runtime:conditional_thunk",
        "//xla/backends/gpu/runtime:convolution_reorder_thunk",
        "//xla/backends/gpu/runtime:convolution_thunk",
        "//xla/backends/gpu/runtime:cudnn_thunk",
        "//xla/backends/gpu/runtime:custom_call_thunk",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:device_to_host_copy_thunk",
        "//xla/backends/gpu/runtime:dynamic_slice_fusion_v2_thunk",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/backends/gpu/runtime:fft_thunk",
        "//xla/backends/gpu/runtime:gpu_constants",
        "//xla/backends/gpu/runtime:gpu_executable",
        "//xla/backends/gpu/runtime:gpublas_lt_matmul_thunk",
        "//xla/backends/gpu/runtime:host_execute_thunk",
        "//xla/backends/gpu/runtime:host_send_recv_thunk",
        "//xla/backends/gpu/runtime:host_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:infeed_thunk",
        "//xla/backends/gpu/runtime:legacy_custom_call_thunk",
        "//xla/backends/gpu/runtime:memset_thunk",
        "//xla/backends/gpu/runtime:norm_thunk",
        "//xla/backends/gpu/runtime:outfeed_thunk",
        "//xla/backends/gpu/runtime:ragged_all_to_all_thunk",
        "//xla/backends/gpu/runtime:recv_thunk",
        "//xla/backends/gpu/runtime:replica_id_thunk",
        "//xla/backends/gpu/runtime:rng_seed_thunk",
        "//xla/backends/gpu/runtime:select_k_exec_ffi",
        "//xla/backends/gpu/runtime:send_thunk",
        "//xla/backends/gpu/runtime:sequential_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:topk",
        "//xla/backends/gpu/runtime:triangular_solve_thunk",
        "//xla/backends/gpu/runtime:while_thunk",
        "//xla/backends/gpu/target_config:cudnn_device_props_json",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_kernel_source",
        "//xla/codegen/emitters:computation_fingerprint",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/codegen/xtile:block_level_parameters",
        "//xla/core/host_offloading:host_offloading_executable_proto_cc",
        "//xla/ffi:attribute_map",
        "//xla/ffi:attributes",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_instruction_utils",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:transforms_gpu_passes",
        "//xla/runtime:device_id",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:call_graph",
        "//xla/service:collective_opt_utils",
        "//xla/service:dump",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_proto_cc",
        "//xla/service:logical_buffer",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:dense_data_intermediate",
        "//xla/service/gpu:gpu_conv_runner",
        "//xla/service/gpu:gpu_memory_space_assignment",
        "//xla/service/gpu:gpu_norm_runner",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/gpu:kernel_reuse_cache_proto_cc",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:metrics",
        "//xla/service/gpu:mlir_context_pool",
        "//xla/service/gpu:stream_executor_util",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/gpu:blas_lt_matmul_algorithm",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
        "@tsl//tsl/platform:human_readable_json",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:scoped_annotation",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "thunk_emitter_test",
    srcs = ["thunk_emitter_test.cc"],
    tags = ["gpu"],
    deps = [
        ":thunk_emitter",
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/gpu/codegen:cubin_custom_kernel_compiler",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen/emitters:mlir_kernel_emitter",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:buffer_value",
        "//xla/service:gpu_topology",
        "//xla/service/gpu:mlir_context_pool",
        "//xla/service/gpu:target_constants",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional_ref",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "kernel_call",
    srcs = ["kernel_call.cc"],
    hdrs = ["kernel_call.h"],
    deps = [
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "kernel_call_test",
    srcs = ["kernel_call_test.cc"],
    deps = [
        ":kernel_call",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "triton_call",
    srcs = ["triton_call.cc"],
    hdrs = ["triton_call.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "triton_call_test",
    srcs = ["triton_call_test.cc"],
    deps = [
        ":triton_call",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "execution_stream_assignment",
    srcs = ["execution_stream_assignment.cc"],
    hdrs = ["execution_stream_assignment.h"],
    deps = [
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_copy",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/backends/gpu/hlo/transforms/hlo_scheduling:gpu_latency_hiding_scheduler",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:collective_opt_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "execution_stream_assignment_test",
    srcs = ["execution_stream_assignment_test.cc"],
    deps = [
        ":execution_stream_assignment",
        "//xla:side_effect_util",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/backends/gpu/runtime:execution_stream_id",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "gpu_hlo_ordering",
    srcs = ["gpu_hlo_ordering.cc"],
    hdrs = ["gpu_hlo_ordering.h"],
    deps = [
        "//xla:shape_util",
        "//xla/ffi:ffi_registry",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_hlo_ordering_test",
    size = "small",
    srcs = ["gpu_hlo_ordering_test.cc"],
    deps = [
        ":gpu_hlo_ordering",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",  # buildcleaner: keep
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
