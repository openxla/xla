load("@xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@xla//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "proto_serialization",
    srcs = ["proto_serialization.cc"],
    hdrs = ["proto_serialization.h"],
    visibility = internal_visibility([
        "@xla//xla/pjrt:__subpackages__",
        "@xla//xla/python:__pkg__",
        "@xla//xla/service:__pkg__",
        "@xla//xla/stream_executor:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
        "//tensorflow/compiler/tf2xla/kernels:__pkg__",
        "//tensorflow/core/util/autotune_maps:__pkg__",
    ]),
    deps = [
        "//third_party/protobuf:protobuf_lite",
        "//third_party/protobuf/io",
        "//tsl/platform:hash",
        "//tsl/platform:logging",
        "//tsl/platform:protobuf",
        "//tsl/platform:types",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "proto_serialization.cc",
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)
