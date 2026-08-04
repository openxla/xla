# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

_RUNTIME_TEST_LINKOPTS = if_google([
    "-Wl,--wrap=CuteDSLRT_Function_Run",
    "-Wl,--wrap=CuteDSLRT_GetErrorName",
    "-Wl,--wrap=CuteDSLRT_GetErrorString",
    "-Wl,--wrap=CuteDSLRT_Module_Create_From_Bytes",
    "-Wl,--wrap=CuteDSLRT_Module_Destroy",
    "-Wl,--wrap=CuteDSLRT_Module_Get_Function",
], [])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

tf_proto_library(
    name = "collective_config_proto",
    srcs = ["config.proto"],
    deps = ["//xla:xla_data_proto"],
)

cc_library(
    name = "runtime_static_unavailable",
    visibility = ["//visibility:public"],
)

label_flag(
    name = "cutedsl_runtime_static",
    build_setting_default = if_google(
        "//third_party/py/cutlass/runtime",
        ":runtime_static_unavailable",
    ),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "runtime_api",
    hdrs = ["runtime_api.h"],
)

cc_library(
    name = "runtime",
    srcs = ["runtime_api.cc"],
    hdrs = ["runtime.h"],
    linkopts = if_google([], select({
        "@platforms//os:linux": ["-ldl"],
        "//conditions:default": [],
    })),
    local_defines = if_google(["XLA_CUTEDSL_RUNTIME_STATIC"]),
    deps = if_google([
        ":cutedsl_runtime_static",
    ], []) + [
        ":runtime_api",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "module",
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        ":runtime",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ffi_abi",
    hdrs = ["ffi_abi.h"],
)

cc_library(
    name = "kernel_call",
    srcs = ["kernel_call.cc"],
    hdrs = ["kernel_call.h"],
    deps = [
        ":ffi_abi",
        ":module",
        "//xla/ffi",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_binary(
    name = "libcute_dsl_runtime.so",
    testonly = True,
    srcs = ["runtime_test_library.cc"],
    linkshared = True,
    target_compatible_with = if_google(["@platforms//:incompatible"], []),
    deps = [":runtime_api"],
)

xla_cc_test(
    name = "runtime_test",
    srcs = ["runtime_test.cc"],
    data = [":libcute_dsl_runtime.so"],
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    target_compatible_with = if_google(["@platforms//:incompatible"], []) + select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":runtime",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    deps = [
        ":collective_config_proto_cc",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "ffi",
    srcs = [
        "ffi.cc",
        "//xla/ffi/api:all_headers",
    ],
    hdrs = ["ffi.h"],
    deps = [
        ":collective_config",
        ":collective_config_proto_cc",
        ":ffi_abi",
        ":kernel_call",
        ":module",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu:ffi",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_communicator",
        "//xla/backends/gpu/runtime:collective_clique_requests",
        "//xla/backends/gpu/runtime:collective_cliques",
        "//xla/backends/gpu/runtime:collective_execution",
        "//xla/backends/gpu/runtime:collective_kernel_api",
        "//xla/backends/gpu/runtime:collective_memory",
        "//xla/backends/gpu/runtime:collective_memory_requests",
        "//xla/backends/gpu/runtime:collective_params",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:symmetric_memory",
        "//xla/ffi",
        "//xla/ffi/api:c_api",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:multi_gpu_barrier_kernel",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util:tied_ref",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

cc_library(
    name = "handlers",
    visibility = ["//xla/service:__pkg__"],
    deps = [":ffi"],
)

xla_cc_test(
    name = "module_test",
    srcs = ["module_test.cc"],
    linkopts = _RUNTIME_TEST_LINKOPTS,
    deps = [
        ":module",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "collective_config_test",
    srcs = ["config_test.cc"],
    deps = [
        ":collective_config",
        ":collective_config_proto_cc",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "collective_ffi_test",
    srcs = ["collective_ffi_test.cc"],
    linkopts = _RUNTIME_TEST_LINKOPTS,
    deps = [
        ":collective_config_proto_cc",
        ":ffi",
        ":ffi_abi",
        ":module",
        "//xla:types",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_collectives_stub",
        "//xla/backends/gpu/runtime:collective_clique_requests",
        "//xla/backends/gpu/runtime:collective_cliques",
        "//xla/backends/gpu/runtime:collective_memory",
        "//xla/backends/gpu/runtime:collective_memory_requests",
        "//xla/backends/gpu/runtime:collective_params",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:symmetric_memory",
        "//xla/ffi",
        "//xla/ffi:attribute_map",
        "//xla/ffi:call_frame",
        "//xla/ffi:execution_state",
        "//xla/ffi:ffi_api",
        "//xla/ffi:ffi_registry",
        "//xla/ffi:invoke",
        "//xla/ffi/api:c_api",
        "//xla/service:computation_placer",
        "//xla/service:service_executable_run_options",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:mock_platform",
        "//xla/stream_executor:mock_stream",
        "//xla/stream_executor:mock_stream_executor",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "ffi_unit_test",
    srcs = ["ffi_unit_test.cc"],
    linkopts = _RUNTIME_TEST_LINKOPTS,
    deps = [
        ":ffi",
        "//xla:types",
        "//xla/ffi",
        "//xla/ffi:call_frame",
        "//xla/ffi:execution_state",
        "//xla/ffi:ffi_api",
        "//xla/ffi:ffi_registry",
        "//xla/ffi:invoke",
        "//xla/ffi/api:c_api",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:mock_stream",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    backends = ["b200"],
    data = ["vector_add.hlo"],
    target_compatible_with = if_google([], ["@platforms//:incompatible"]),
    deps = [
        ":ffi",
        "//xla:error_spec",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)
