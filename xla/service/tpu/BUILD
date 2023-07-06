# Description:
#    Open-sourced parts of TPU backend for XLA.

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "host_command",
    hdrs = ["host_command.h"],
)

cc_library(
    name = "host_command_util",
    srcs = ["host_command_util.cc"],
    hdrs = ["host_command_util.h"],
    deps = [
        ":host_command",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "host_send_recv",
    hdrs = ["host_send_recv.h"],
)

cc_library(
    name = "x64_literal_linearizer",
    srcs = ["x64_literal_linearizer.cc"],
    hdrs = ["x64_literal_linearizer.h"],
    deps = [
        "//tensorflow/core/profiler/lib:traceme",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_transfer_metadata",
    hdrs = ["tpu_transfer_metadata.h"],
    deps = ["//xla/service:transfer_manager"],
)
