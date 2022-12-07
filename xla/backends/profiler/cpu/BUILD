load("@tsl//platform:rules_cc.bzl", "cc_library")
load("@tsl//profiler/builds:build_config.bzl", "tf_profiler_copts")

# copybara:uncomment package(default_applicable_licenses = ["//third_party/tensorflow:license"])

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer_factory.cc"],
    visibility = [
        "//third_party/car/onboard/gpu:__subpackages__",
        "//third_party/tensorflow/core/profiler:internal",
        "//xla/backends/profiler:__pkg__",
    ],
    deps = [
        ":host_tracer_impl",
        "@tsl//profiler/lib:profiler_factory",
        "@tsl//profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_impl",
    srcs = ["host_tracer.cc"],
    hdrs = ["host_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//third_party/tensorflow/core/profiler:internal",
    ],
    deps = [
        "@tsl//platform:errors",
        "@tsl//platform:status",
        "@tsl//platform:types",
        "@tsl//profiler/backends/cpu:host_tracer_utils",
        "@tsl//profiler/backends/cpu:traceme_recorder",
        "@tsl//profiler/lib:profiler_interface",
        "@tsl//profiler/protobuf:xplane_proto_cc",
        "@tsl//profiler/utils:time_utils",
        "@tsl//profiler/utils:xplane_schema",
        "@tsl//profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "python_tracer",
    srcs = ["python_tracer_factory.cc"],
    visibility = [
        "//third_party/tensorflow/core/profiler:internal",
        "//xla/python:__pkg__",
    ],
    deps = [
        ":python_tracer_impl",
        "@tsl//profiler/lib:profiler_factory",
        "@tsl//profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer_impl",
    srcs = ["python_tracer.cc"],
    hdrs = ["python_tracer.h"],
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = [
        "//third_party/tensorflow/core/profiler:internal",
    ],
    deps = [
        "//xla/python/profiler/internal:python_hooks",
        "@tsl//platform:errors",
        "@tsl//platform:logging",
        "@tsl//platform:macros",
        "@tsl//platform:status",
        "@tsl//profiler/lib:profiler_interface",
        "@tsl//profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "metadata_collector",
    srcs = ["metadata_collector.cc"],
    copts = tf_profiler_copts(),
    visibility = [
        "//third_party/tensorflow/core/profiler:internal",
        "//xla/backends/profiler:__pkg__",
    ],
    deps = [
        ":metadata_utils",
        "//xla/service:hlo_proto_cc",
        "//xla/service:xla_debug_info_manager",
        "@tsl//platform:macros",
        "@tsl//platform:status",
        "@tsl//profiler/lib:profiler_factory",
        "@tsl//profiler/lib:profiler_interface",
        "@tsl//profiler/protobuf:profiler_options_proto_cc",
        "@tsl//profiler/protobuf:xplane_proto_cc",
        "@tsl//profiler/utils:xplane_schema",
        "@tsl//profiler/utils:xplane_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_utils",
    hdrs = ["metadata_utils.h"],
    visibility = ["//third_party/tensorflow/core/profiler:internal"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@tsl//profiler/convert:xla_op_utils",
        "@tsl//profiler/protobuf:xplane_proto_cc",
        "@tsl//profiler/utils:xplane_builder",
        "@tsl//profiler/utils:xplane_schema",
    ],
)
