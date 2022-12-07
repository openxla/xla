load("@tsl//platform:rules_cc.bzl", "cc_library")
load(
    "@tsl//profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)
load("@tsl//:tsl.default.bzl", "tsl_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["@tsl//profiler:internal"],
    licenses = ["notice"],
)

# Linked to pywrap_tensorflow.
cc_library(
    name = "profiler_service_impl",
    srcs = ["profiler_service_impl.cc"],
    hdrs = ["profiler_service_impl.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//third_party/tensorflow/core/data/service:__pkg__",
        "//third_party/tensorflow/core/distributed_runtime/rpc:__pkg__",
        "//third_party/tensorflow/core/profiler/rpc:__pkg__",
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow_serving/model_servers:__pkg__",
        "@tsl//profiler/rpc/client:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//platform:env",
        "@tsl//platform:env_time",
        "@tsl//platform:errors",
        "@tsl//platform:logging",
        "@tsl//platform:macros",
        "@tsl//platform:mutex",
        "@tsl//platform:status",
        "@tsl//profiler/lib:profiler_session",
        "@tsl//profiler/protobuf:profiler_service_proto_cc",
        "@tsl//profiler/protobuf:profiler_service_cc_grpc_proto",
        "@tsl//profiler/protobuf:xplane_proto_cc",
        "@tsl//profiler/rpc/client:save_profile",
        "@tsl//profiler/utils:file_system_utils",
        "@tsl//profiler/utils:math_utils",
        "@tsl//profiler/utils:time_utils",
        "@tsl//profiler/utils:xplane_utils",
    ] + tsl_grpc_cc_dependencies(),
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_server_for_pybind",
    actual = ":profiler_server_impl",
    visibility = ["//third_party/tensorflow/python/profiler/internal:__pkg__"],
)

cc_library(
    name = "profiler_server_impl",
    srcs = ["profiler_server.cc"],
    hdrs = ["profiler_server.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//third_party/tensorflow/core/profiler/rpc:__pkg__",
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow/python/profiler/internal:__pkg__",
        "//xla:__subpackages__",
        "@tsl//profiler:internal",
        "@tsl//profiler/rpc/client:__pkg__",
    ],
    deps = [
        ":profiler_service_impl",
        "@com_google_absl//absl/strings",
        "@tsl//platform:logging",
        "@tsl//platform:types",
        "@tsl//profiler/protobuf:profiler_service_cc_grpc_proto",
    ] + tsl_grpc_cc_dependencies(),
    alwayslink = True,
)
