load(
    "@tsl//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "@tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "histogram",
    srcs = ["histogram.cc"],
    hdrs = ["histogram.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/protobuf:histogram_proto_cc",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "histogram.cc",
        "histogram.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/histogram:__pkg__"]),
)

filegroup(
    name = "legacy_lib_histogram_all_headers",
    srcs = [
        "histogram.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/histogram:__pkg__"]),
)

tsl_cc_test(
    name = "histogram_test",
    srcs = [
        "histogram_test.cc",
    ],
    deps = [
        ":histogram",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/protobuf:histogram_proto_cc",
    ],
)
