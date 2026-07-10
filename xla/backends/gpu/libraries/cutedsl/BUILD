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
    name = "ffi",
    srcs = [
        "ffi.cc",
        "//xla/ffi/api:all_headers",
    ],
    deps = [
        ":runtime_api",
        "//xla/backends/gpu:ffi",
        "//xla/ffi",
        "//xla/ffi/api:c_api",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
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
