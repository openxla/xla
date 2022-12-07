# Description:
#   Implementation of XLA’s HLO transformations.

load("@tsl//platform:rules_cc.bzl", "cc_library")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_constant_splitter",
    srcs = ["hlo_constant_splitter.cc"],
    hdrs = ["hlo_constant_splitter.h"],
    deps = ["//xla/service:hlo_pass"],
)

tf_cc_test(
    name = "hlo_constant_splitter_test",
    srcs = ["hlo_constant_splitter_test.cc"],
    deps = [
        ":hlo_constant_splitter",
        "//xla:shape_util",
        "//xla:test",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//lib/core:status_test_util",
    ],
)
