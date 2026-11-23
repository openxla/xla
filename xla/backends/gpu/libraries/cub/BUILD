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

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/service/gpu:build_defs.bzl", "get_cub_sort_kernel_types")
load("//xla/stream_executor/cuda:build_defs.bzl", "embeddable_cuda_library")
load("//xla/tests:build_defs.bzl", "NVIDIA_GPU_DEFAULT_BACKENDS", "xla_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "cub_sort_utils",
    srcs = ["cub_sort_utils.cc"],
    hdrs = ["cub_sort_utils.h"],
    visibility = ["//xla:internal"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "cub_sort_utils_test",
    srcs = ["cub_sort_utils_test.cc"],
    deps = [
        ":cub_sort_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cub_sort_policy",
    hdrs = ["cub_sort_policy.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//xla:internal"],
)

cc_library(
    name = "cub_sort_planner",
    srcs = ["cub_sort_planner.cc"],
    hdrs = ["cub_sort_planner.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//xla:internal"],
    deps = [
        ":cub_sort_policy",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "cub_sort_planner_test",
    srcs = ["cub_sort_planner_test.cc"],
    deps = [
        ":cub_sort_planner",
        ":cub_sort_policy",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cub_sort_kernels_impl_hdrs",
    hdrs = ["cub_sort_kernels_impl.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "//xla:xla_data_proto_cc",
    ],
)

embeddable_cuda_library(
    name = "cub_sort_kernels_exclusive_sum",
    srcs = ["cub_sort_kernels_exclusive_sum.cu.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    register_with_cudart = False,
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cub_sort_kernels_impl_hdrs",
        ":cub_sort_policy",
        "@local_config_cuda//cuda:cub_headers",
    ],
)

[embeddable_cuda_library(
    name = "cub_sort_kernels_type_{}".format(typename),
    srcs = ["cub_sort_kernels_impl.cu.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    local_defines = ["CUB_TYPE_" + typename.upper()],
    register_with_cudart = False,
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cub_sort_kernels_impl_hdrs",
        ":cub_sort_policy",
        "//xla:xla_data_proto_cc",
        "@local_config_cuda//cuda:cub_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
) for typename in get_cub_sort_kernel_types()]

cc_library(
    name = "cub_sort_kernels",
    srcs = ["cub_sort_kernels.cc"],
    hdrs = ["cub_sort_kernels.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//xla:internal"],
    deps = [
        ":cub_sort_kernels_exclusive_sum",  # buildcleaner: keep
        ":cub_sort_kernels_impl_hdrs",
        ":cub_sort_policy",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_elf_utils",
        "//xla/stream_executor/cuda:cuda_fatbin",
        "//xla/stream_executor/cuda:cudart_kernel_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + [
        ":cub_sort_kernels_type_{}".format(typename)
        for typename in get_cub_sort_kernel_types()
    ],
)

xla_test(
    name = "cub_sort_kernels_test",
    srcs = ["cub_sort_kernels_test.cc"],
    backends = NVIDIA_GPU_DEFAULT_BACKENDS,
    tags = ["cuda-only"],
    deps = [
        ":cub_sort_kernels",
        ":cub_sort_policy",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_elf_utils",
        "//xla/stream_executor/cuda:cuda_platform",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cub_sort_thunk_emitter_cuda",
    srcs = [
        "cub_sort_thunk_emitter.h",
        "cub_sort_thunk_emitter_cuda.cc",
    ],
    # copybara:uncomment compatible_with = [],
    tags = [
        "cuda-only",
        "gpu",
        "manual",
    ],
    deps = [
        ":cub_sort_kernels",
        ":cub_sort_planner",
        ":cub_sort_policy",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/service:buffer_assignment",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:kernel_args_packing_spec",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_elf_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cub_sort_thunk_emitter_stub",
    srcs = [
        "cub_sort_thunk_emitter.h",
        "cub_sort_thunk_emitter_stub.cc",
    ],
    # copybara:uncomment compatible_with = [],
    tags = ["manual"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/service:shaped_slice",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cub_sort_thunk_emitter",
    hdrs = ["cub_sort_thunk_emitter.h"],
    # copybara:uncomment compatible_with = [],
    visibility = ["//xla:internal"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/service:shaped_slice",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status:statusor",
    ] + if_cuda_is_configured(
        [":cub_sort_thunk_emitter_cuda"],
        [":cub_sort_thunk_emitter_stub"],
    ),
)

xla_test(
    name = "cub_sort_thunk_emitter_test",
    srcs = ["cub_sort_thunk_emitter_test.cc"],
    backends = NVIDIA_GPU_DEFAULT_BACKENDS,
    tags = ["cuda-only"],
    deps = [
        ":cub_sort_planner",
        ":cub_sort_policy",
        ":cub_sort_thunk_emitter",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/runtime:command",
        "//xla/backends/gpu/runtime:command_buffer_thunk",
        "//xla/backends/gpu/runtime:command_executor",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:thunk_id",
        "//xla/backends/gpu/runtime:thunk_proto_cc",
        "//xla/backends/gpu/runtime:thunk_test_utils",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/service:shaped_slice",
        "//xla/service/gpu:buffer_allocations",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
