# Description:
# Distributed runtime modules for machine learning, which allows coordination between multiple
# processes for distributed operations.

load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        "//tsl:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "call_options",
    srcs = ["call_options.cc"],
    hdrs = ["call_options.h"],
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "noncopyable_buffer",
    hdrs = ["noncopyable_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:platform_port",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "call_options.h",
    ],
)
