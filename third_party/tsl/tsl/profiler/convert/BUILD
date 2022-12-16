load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//tsl/profiler:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_op_utils",
    hdrs = ["xla_op_utils.h"],
    visibility = [
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_profiler_backends",
    ],
    deps = ["@com_google_absl//absl/strings"],
)

tsl_cc_test(
    name = "xla_op_utils_test",
    size = "small",
    srcs = ["xla_op_utils_test.cc"],
    deps = [
        ":xla_op_utils",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "post_process_single_host_xplane",
    srcs = ["post_process_single_host_xplane.cc"],
    hdrs = ["post_process_single_host_xplane.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tsl/profiler:internal"],
    deps = [
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:xplane_schema",
        "//tsl/profiler/utils:xplane_utils",
    ],
)
