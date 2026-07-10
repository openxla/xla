load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

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

cc_library(
    name = "runtime_unavailable",
)

# Bazel providers can select either a cc_import for libcute_dsl_runtime.so or
# the standalone combined libcute_dsl_runtime.a archive. Google builds retain
# their existing in-tree provider; OSS builds must opt in explicitly.
label_flag(
    name = "cutedsl_runtime",
    build_setting_default = if_google(
        "//third_party/py/cutlass/runtime",
        ":runtime_unavailable",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cutedsl_runtime_unavailable_selected",
    flag_values = {
        ":cutedsl_runtime": "//xla/backends/gpu/libraries/cutedsl:runtime_unavailable",
    },
)

cc_library(
    name = "runtime_api",
    srcs = ["runtime_api.cc"],
    hdrs = ["runtime_api.h"],
    local_defines = select({
        ":cutedsl_runtime_unavailable_selected": [
            "XLA_CUTEDSL_RUNTIME_UNAVAILABLE",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":cutedsl_runtime",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "module",
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        ":runtime_api",
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
    name = "collective_config",
    srcs = ["collective_config.cc"],
    hdrs = ["collective_config.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "collective_ffi",
    srcs = [
        "collective_ffi.cc",
        "//xla/ffi/api:all_headers",
    ],
    hdrs = ["collective_ffi.h"],
    deps = [
        ":collective_config",
        ":module",
        ":runtime_api",
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
        "//xla/runtime:device_id",
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
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ffi",
    srcs = [
        "ffi.cc",
        "//xla/ffi/api:all_headers",
    ],
    deps = [
        ":module",
        ":runtime_api",
        "//xla/backends/gpu:ffi",
        "//xla/ffi",
        "//xla/ffi/api:c_api",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "runtime_api_test",
    srcs = ["runtime_api_test.cc"],
    deps = [
        ":runtime_api",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "module_test",
    srcs = ["module_test.cc"],
    deps = [
        ":module",
        ":runtime_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "collective_config_test",
    srcs = ["collective_config_test.cc"],
    deps = [
        ":collective_config",
        "//xla/ffi",
        "//xla/ffi:attribute_map",
        "//xla/ffi:call_frame",
        "//xla/ffi:ffi_api",
        "//xla/ffi:invoke",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "collective_ffi_test",
    srcs = ["collective_ffi_test.cc"],
    deps = [
        ":collective_config",
        ":collective_ffi",
        ":runtime_api",
        "//xla:types",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_collectives_stub",
        "//xla/backends/gpu/runtime:collective_clique_requests",
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
        "//xla/runtime:device_id",
        "//xla/service:computation_placer",
        "//xla/service:service_executable_run_options",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_address",
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
    ],
)

xla_cc_test(
    name = "ffi_unit_test",
    srcs = ["ffi_unit_test.cc"],
    deps = [
        ":ffi",
        ":runtime_api",
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
    # Requires a B200 and a matching CuTeDSL runtime supplied out of tree.
    tags = [
        "manual",
        "no_oss",
    ],
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
