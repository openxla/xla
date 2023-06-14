# Description:
#   Host-platform specific StreamExecutor support code.

load("//xla:xla.bzl", "xla_cc_test")
load("@tsl//tsl:tsl.bzl", "set_external_visibility")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "host_platform_id",
    srcs = [
        "host_platform_id.cc",
    ],
    hdrs = [
        "host_platform_id.h",
    ],
    deps = [
        "//xla/stream_executor:platform",
    ],
)

cc_library(
    name = "host_platform",
    srcs = [
        "host_platform.cc",
    ],
    hdrs = [
        "host_platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":host_gpu_executor",
        ":host_platform_id",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:errors",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "host_stream",
    srcs = [
        "host_stream.cc",
    ],
    hdrs = [
        "host_stream.h",
    ],
    deps = [
        "//xla/stream_executor:kernel",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:denormal",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:setround",
    ],
)

cc_library(
    name = "host_timer",
    srcs = [
        "host_timer.cc",
    ],
    hdrs = [
        "host_timer.h",
    ],
    deps = [
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor:timer",
        "//xla/stream_executor/platform",
    ],
)

# TODO(22689637): Rename this target.
cc_library(
    name = "host_gpu_executor",
    srcs = [
        "host_gpu_executor.cc",
    ],
    hdrs = [
        "host_gpu_executor.h",
    ],
    deps = [
        ":host_platform_id",
        ":host_stream",
        ":host_timer",
        "//xla/stream_executor",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:rng",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor:stream_executor_pimpl",  # fixdeps: keep
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform/profile_utils:profile_utils_cpu_utils",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "host_stream_test",
    srcs = ["host_stream_test.cc"],
    deps = [
        ":host_platform",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)
