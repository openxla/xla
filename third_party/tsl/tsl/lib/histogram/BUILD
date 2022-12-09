load("//tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        "//third_party/tensorflow/core/lib/histogram:__pkg__",
        "//tsl/lib/monitoring:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "histogram",
    srcs = ["histogram.cc"],
    hdrs = ["histogram.h"],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/protobuf:histogram_proto_cc",
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
    visibility = [
        "//third_party/tensorflow/core:__pkg__",
        "//third_party/tensorflow/core/lib/histogram:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_histogram_all_headers",
    srcs = [
        "histogram.h",
    ],
    visibility = [
        "//third_party/tensorflow/core:__pkg__",
        "//third_party/tensorflow/core/lib/histogram:__pkg__",
    ],
)

tsl_cc_test(
    name = "histogram_test",
    srcs = [
        "histogram_test.cc",
    ],
    deps = [
        ":histogram",
        "//tsl/platform:logging",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/protobuf:histogram_proto_cc",
    ],
)
