load(
    "@tsl//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")

# Snappy targets.

load(
    "@tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/lib/io:__pkg__",
        "//xla/tsl/lib/io:__pkg__",
    ]),
    licenses = ["notice"],
)

exports_files([
    "snappy_compression_options.h",
    "snappy_inputbuffer.h",
    "snappy_inputstream.h",
    "snappy_outputbuffer.h",
    "snappy_inputstream.cc",
    "snappy_test.cc",
])

cc_library(
    name = "snappy_inputbuffer",
    srcs = ["snappy_inputbuffer.cc"],
    hdrs = ["snappy_inputbuffer.h"],
    deps = [
        "//xla/tsl/lib/io:inputstream_interface",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_outputbuffer",
    srcs = ["snappy_outputbuffer.cc"],
    hdrs = ["snappy_outputbuffer.h"],
    deps = [
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_inputstream",
    srcs = ["snappy_inputstream.cc"],
    hdrs = ["snappy_inputstream.h"],
    deps = [
        "//xla/tsl/lib/io:inputstream_interface",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:platform_port",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_compression_options",
    hdrs = ["snappy_compression_options.h"],
    deps = [
        "@tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "snappy_test",
    size = "small",
    srcs = ["snappy_test.cc"],
    deps = [
        ":snappy_inputbuffer",
        ":snappy_inputstream",
        ":snappy_outputbuffer",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/lib/io:inputbuffer",
        "//xla/tsl/lib/io:random_inputstream",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)
