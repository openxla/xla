# Description:
#   RPC communication interfaces and implementations for TensorFlow.

load("@tsl//tsl/platform:build_config.bzl", "tf_proto_library", "tsl_cc_test")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "async_service_interface",
    srcs = [],
    hdrs = ["async_service_interface.h"],
    deps = [],
)

cc_library(
    name = "grpc_call",
    srcs = [],
    hdrs = ["grpc_call.h"],
    deps = [
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:refcount",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = ["grpc_util.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:stringpiece",
        "@tsl//tsl/platform:stringprintf",
        "@tsl//tsl/platform:tstring",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/protobuf:distributed_runtime_payloads_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

tsl_cc_test(
    name = "grpc_util_test",
    size = "small",
    srcs = ["grpc_util_test.cc"],
    tags = [
        "no_mac",
    ],
    deps = [
        ":grpc_util",
        ":test_request_proto_cc_impl",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
        "@tsl//tsl/platform:test_main",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_channel_common",
    hdrs = ["grpc_channel_common.h"],
    deps = [
        ":grpc_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "grpc_channel",
    srcs = ["grpc_channel.cc"],
    hdrs = ["grpc_channel.h"],
    deps = [
        ":grpc_channel_common",
        ":grpc_util",
        "//xla/tsl/util:device_name_utils",
        "@com_github_grpc_grpc//:grpc_security_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/gtl:map_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:str_util",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/protobuf:rpc_options_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

tsl_cc_test(
    name = "grpc_channel_test",
    size = "small",
    srcs = [
        "grpc_channel_test.cc",
    ],
    deps = [
        ":grpc_channel",
        ":grpc_util",
        "//xla/tsl/util:device_name_utils",
        "@com_google_absl//absl/log",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/protobuf:rpc_options_proto_cc_impl",
    ],
)

cc_library(
    name = "grpc_state",
    hdrs = ["grpc_state.h"],
    deps = [
        ":grpc_client_cq_tag",
        ":grpc_util",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:strcat",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_client_cq_tag",
    srcs = [],
    hdrs = ["grpc_client_cq_tag.h"],
    deps = [
        "@tsl//tsl/platform:macros",
    ],
)

tf_proto_library(
    name = "test_request_proto",
    testonly = 1,
    srcs = ["test_request.proto"],
    create_java_proto = False,
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)
