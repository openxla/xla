load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "plugin_tracer",
    srcs = ["plugin_tracer.cc"],
    hdrs = ["plugin_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//third_party/xprof/plugin/tensorboard_plugin_profile/integration_tests/tpu/tensorflow:__pkg__",
        "//xla:internal",
    ],
    deps = [
        "//xla:status",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)
