load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load(
    "@tsl//tsl:tsl.bzl",
    "tsl_copts",
    "tsl_gpu_library",
)
load("@tsl//tsl:tsl.default.bzl", "tsl_gpu_cc_test")
load(
    "@tsl//tsl/platform:build_config.bzl",
    "tf_additional_device_tracer_srcs",
)
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "tf_additional_cupti_deps",
)
load("@tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    features = [
        "-layering_check",
    ],
)

tsl_gpu_library(
    name = "device_tracer",
    srcs = tf_additional_device_tracer_srcs(),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_tracer",
        ":cupti_wrapper",
        ":rocm_tracer",
    ],
    deps = [
        ":cupti_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/platform:env_time",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:time_utils",
        "@tsl//tsl/util:env_var",
    ],
    alwayslink = 1,
)

tsl_gpu_library(
    name = "cupti_interface",
    hdrs = if_cuda(["cupti_interface.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:types",
    ] + tf_additional_cupti_deps(),
)

tsl_gpu_library(
    name = "mock_cupti",
    testonly = 1,
    hdrs = if_cuda(["mock_cupti.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_interface",
    ],
    deps = [
        "@tsl//tsl/platform:test",
    ],
)

tsl_gpu_library(
    name = "cupti_error_manager",
    srcs = if_cuda(["cupti_error_manager.cc"]),
    hdrs = if_cuda(["cupti_error_manager.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_interface",
        ":cupti_wrapper",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

tsl_gpu_cc_test(
    name = "cupti_error_manager_test",
    size = "small",
    srcs = ["cupti_error_manager_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "gpu_cupti",
        "nomac",
    ],
    deps = [
        "@tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        ":cuda_test",
        ":cupti_error_manager",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":mock_cupti",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/profiler/utils:time_utils",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
    ]),
)

cuda_library(
    name = "cuda_test",
    testonly = 1,
    srcs = ["cuda_test.cu.cc"],
    hdrs = ["cuda_test.h"],
    copts = select({
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options",
            "ptxas-options=-v",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@tsl//tsl/platform:test",
    ],
)

# Rationale for linkstatic: The symbols in libcupti_static.a have hidden
# visibility. The wrapper will fail to find them if it's ever built as a
# shared library. This is the same issue as b/11094727. Always linking
# the wrapper statically works around the issue. An alternative would be
# to patch libcupti_static, but it's not worth the trouble considering
# that the wrapper is about the only direct user.
tsl_gpu_library(
    name = "cupti_wrapper",
    srcs = if_cuda(["cupti_wrapper.cc"]),
    hdrs = if_cuda(["cupti_wrapper.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
    ] + tf_additional_cupti_deps(),
)

tsl_gpu_library(
    name = "cupti_tracer",
    srcs = if_cuda(["cupti_tracer.cc"]),
    hdrs = if_cuda(["cupti_tracer.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_utils",
        ":nvtx_utils",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack",
        "@tsl//tsl/profiler/lib:scoped_annotation",
        "@tsl//tsl/profiler/utils:buffer_pool",
    ],
)

tsl_gpu_library(
    name = "rocm_tracer",
    srcs = if_rocm(["rocm_tracer.cc"]),
    hdrs = if_rocm(["rocm_tracer.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack",
        "@tsl//tsl/profiler/utils:time_utils",
    ],
)

tsl_gpu_library(
    name = "nvtx_utils",
    srcs = if_cuda(["nvtx_utils.cc"]),
    hdrs = if_cuda(["nvtx_utils.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    deps = [
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
    ],
)

tsl_gpu_library(
    name = "cupti_collector",
    srcs = if_cuda(["cupti_collector.cc"]),
    hdrs = if_cuda(["cupti_collector.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:parse_annotation",
        "@tsl//tsl/profiler/utils:trace_utils",
        "@tsl//tsl/profiler/utils:xplane_builder",
        "@tsl//tsl/profiler/utils:xplane_schema",
        "@tsl//tsl/profiler/utils:xplane_utils",
    ] + tf_additional_cupti_deps(),
)

cc_library(
    name = "cupti_collector_header",
    hdrs = ["cupti_collector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_gpu_library(
    name = "cupti_utils",
    srcs = if_cuda(["cupti_utils.cc"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_wrapper",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
