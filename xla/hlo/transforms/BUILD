# Description:
#   Implementation of XLA’s HLO transformations.

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    name = "hlo_computation_deduplicator",
    srcs = ["hlo_computation_deduplicator.cc"],
    hdrs = ["hlo_computation_deduplicator.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "hlo_computation_deduplicator_test",
    size = "small",
    srcs = ["hlo_computation_deduplicator_test.cc"],
    deps = [
        ":hlo_computation_deduplicator",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_pass",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "hlo_constant_splitter",
    srcs = ["hlo_constant_splitter.cc"],
    hdrs = ["hlo_constant_splitter.h"],
    deps = ["//xla/service:hlo_pass"],
)

xla_cc_test(
    name = "hlo_constant_splitter_test",
    srcs = ["hlo_constant_splitter_test.cc"],
    deps = [
        ":hlo_constant_splitter",
        "//xla:test",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "hlo_dce",
    srcs = ["hlo_dce.cc"],
    hdrs = ["hlo_dce.h"],
    deps = [
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_dce_test",
    srcs = ["hlo_dce_test.cc"],
    deps = [
        ":hlo_dce",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)
