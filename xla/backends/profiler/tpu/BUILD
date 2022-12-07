load("@tsl//platform:rules_cc.bzl", "cc_library")
load("@tsl//profiler/builds:build_config.bzl", "tf_profiler_copts")
load("@tsl//:tsl.bzl", "if_with_tpu_support")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//third_party/tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_tracer",
    srcs = if_with_tpu_support(["tpu_tracer.cc"]),
    copts = tf_profiler_copts(),
    deps = [
        "//xla/stream_executor/tpu:status_helper",
        "//xla/stream_executor/tpu:tpu_api",
        "//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@com_google_absl//absl/strings",
        "@tsl//platform:errors",
        "@tsl//platform:status",
        "@tsl//platform:types",
        "@tsl//profiler/lib:profiler_factory",
        "@tsl//profiler/lib:profiler_interface",
        "@tsl//profiler/protobuf:profiler_options_proto_cc",
        "@tsl//profiler/protobuf:xplane_proto_cc",
        "@tsl//profiler/utils:xplane_schema",
    ],
    alwayslink = True,
)
