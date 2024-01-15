# Description:
#   SYCL-platform specific StreamExecutor support code.

load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("@tsl//tsl:tsl.bzl", "internal_visibility", "tsl_copts")
load("@tsl//tsl/platform:build_config_root.bzl", "if_static")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "onednn_plugin",
    srcs = ["sycl_dnn.cc"],
    hdrs = ["sycl_dnn.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_driver",
        ":sycl_executor",
        ":sycl_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_timer_header",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

cc_library(
    name = "sycl_gpu_runtime",
    srcs = ["sycl_gpu_runtime.cc"],
    hdrs = ["sycl_gpu_runtime.h"],
    deps = [
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/strings",
        "@local_config_sycl//sycl",
        "@local_config_sycl//sycl:sycl_headers",
        "@tsl//tsl/util:env_var",
    ],
)

cc_library(
    name = "sycl_driver",
    srcs = ["sycl_driver.cc"],
    deps = [
        ":sycl_gpu_runtime",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:stacktrace",
        "@tsl//tsl/platform:status",
        "//xla/stream_executor:device_options",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
    ] + if_sycl_is_configured([
        "@local_config_sycl//sycl:mkl",
    ]),
)

cc_library(
    name = "sycl_collectives",
    srcs = ["sycl_collectives.cc"],
    deps = [
        "//xla/stream_executor/gpu:gpu_collectives_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sycl_platform_id",
    srcs = ["sycl_platform_id.cc"],
    hdrs = ["sycl_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "sycl_platform",
    srcs = ["sycl_platform.cc"],
    hdrs = ["sycl_platform.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_executor",
        ":sycl_platform_id",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//xla/stream_executor",  # buildcleaner: keep
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/platform",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "sycl_kernel",
    srcs = ["sycl_kernel.cc"],
    deps = [
        ":sycl_gpu_runtime",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_event",
    srcs = ["sycl_event.cc"],
    deps = [
        ":sycl_driver",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_stream_header",
    ],
)

cc_library(
    name = "sycl_executor",
    srcs = ["sycl_executor.cc"],
    deps = [
        ":sycl_collectives",
        ":sycl_driver",
        ":sycl_event",
        ":sycl_kernel",
        ":sycl_platform_id",
        "//xla/stream_executor:plugin_registry",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = [
        ":onednn_plugin",
        ":sycl_platform",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sycl_rpath",
    data = [],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-rpath,../local_config_sycl/sycl/sycl/lib",
        ],
    }),
    deps = [],
)

cc_library(
    name = "stream_executor_sycl",
    deps = [
        ":sycl_rpath",
        "//xla/stream_executor:stream_executor_bundle",
    ] + if_static(
        [":all_runtime"],
    ),
)
