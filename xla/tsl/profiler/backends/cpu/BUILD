load("@tsl//tsl/platform:build_config.bzl", "tsl_cc_test")
load("@tsl//tsl/platform:build_config_root.bzl", "if_static")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//xla/tsl:tsl.bzl", "internal_visibility")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "traceme_recorder",
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_profiler_backends",
        "//tensorflow/lite:__pkg__",
    ]),
    deps = [
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
    ] + if_static([
        ":traceme_recorder_impl",
    ]),
)

cc_library(
    name = "traceme_recorder_impl",
    srcs = [
        "traceme_recorder.cc",
    ],
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tensorflow/python:__pkg__",
        "@tsl//tsl/platform/cloud:__pkg__",
        "//xla/tsl/profiler:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/utils:lock_free_queue",
        "@tsl//tsl/profiler/utils:per_thread",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "traceme_recorder_test",
    srcs = ["traceme_recorder_test.cc"],
    deps = [
        ":traceme_recorder",
        ":traceme_recorder_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:notification",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/utils:math_utils",
        "@tsl//tsl/profiler/utils:time_utils",
        "@tsl//tsl/profiler/utils:time_utils_impl",
    ],
)

cc_library(
    name = "annotation_stack",
    hdrs = ["annotation_stack.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:types",
    ] + if_static([
        ":annotation_stack_impl",
    ]),
)

cc_library(
    name = "annotation_stack_impl",
    srcs = [
        "annotation_stack.cc",
        "annotation_stack.h",
    ],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla:__subpackages__",
        "//xla/tsl/profiler:internal",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_utils",
    srcs = ["host_tracer_utils.cc"],
    hdrs = ["host_tracer_utils.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        ":traceme_recorder",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:parse_annotation",
        "@tsl//tsl/profiler/utils:tf_op_utils",
        "@tsl//tsl/profiler/utils:xplane_builder",
        "@tsl//tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "threadpool_listener",
    srcs = ["threadpool_listener.cc"],
    hdrs = ["threadpool_listener.h"],
    visibility = internal_visibility([
        "//tensorflow/python:__pkg__",
        "@tsl//tsl/platform/cloud:__pkg__",
        "//xla/tsl/profiler:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        ":threadpool_listener_state",
        ":traceme_recorder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:tracing",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:time_utils",
        "@tsl//tsl/profiler/utils:xplane_schema",
    ],
)

cc_library(
    name = "threadpool_listener_state",
    srcs = ["threadpool_listener_state.cc"],
    hdrs = ["threadpool_listener_state.h"],
    visibility = internal_visibility([
        "@tsl//tsl/platform:__subpackages__",
    ]),
)
