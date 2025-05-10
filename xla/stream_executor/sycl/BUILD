# Description:
#   SYCL-platform specific StreamExecutor support code.

load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")
load("//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "sycl_platform_id",
    srcs = ["sycl_platform_id.cc"],
    hdrs = ["sycl_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "sycl_platform",
    srcs = if_sycl_is_configured(["sycl_platform.cc"]),
    hdrs = if_sycl_is_configured(["sycl_platform.h"]),
    visibility = ["//visibility:public"],
    deps = if_sycl_is_configured([
        ":sycl_platform_id",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor/platform:initialize",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@tsl//tsl/platform:errors",
    ]),
    alwayslink = True,  # Registers itself with the PlatformManager.
)

xla_cc_test(
    name = "sycl_platform_test",
    srcs = if_sycl_is_configured(["sycl_platform_test.cc"]),
    fail_if_no_test_linked = False,  # NOLINT=If not building with SYCL, we don't have any tests linked.
    deps = if_sycl_is_configured([
        ":sycl_platform",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ]) + [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = if_sycl_is_configured([":sycl_platform"]),
    alwayslink = 1,
)

cc_library(
    name = "sycl_rpath",
    linkopts = ["-Wl,-rpath,../local_config_sycl/sycl/sycl/lib"],
)

cc_library(
    name = "stream_executor_sycl",
    deps = [
        ":sycl_platform_id",
        ":sycl_rpath",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:scratch_allocator",
    ] + if_static([":all_runtime"]),
)

cc_library(
    name = "sycl_status",
    srcs = ["sycl_status.cc",],
    hdrs = ["sycl_status.h",],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_sycl//sycl:sycl_headers",
    ],
)

xla_cc_test(
    name = "sycl_status_test",
    srcs = ["sycl_status_test.cc",],
    tags = [
        "gpu",
        "oneapi-only",
    ],
    deps = [
        ":sycl_status",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_config_sycl//sycl:sycl_headers",
    ],
)
