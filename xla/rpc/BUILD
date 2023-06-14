load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_py_grpc_library",
)
load(
    "@tsl//tsl:tsl.bzl",
    "if_libtpu",
)
load(
    "@tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "xla_service_proto",
    srcs = ["xla_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    create_service = True,
    protodeps = [
        "//xla:xla_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc_stub",
    srcs = ["grpc_stub.cc"],
    hdrs = ["grpc_stub.h"],
    deps = [
        ":xla_service_cc_grpc_proto",
        ":xla_service_proto_cc",
        "//xla:service_interface",
        "//xla:xla_data_cc_grpc_proto",
        "//xla:xla_data_proto_cc",
        "@tsl//tsl/distributed_runtime/rpc:grpc_util",
    ],
)

cc_library(
    name = "grpc_service_main_library",
    srcs = ["grpc_service_main.cc"],
    deps = [
        ":grpc_service",
        "//xla/service:platform_util",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/util:command_line_flags",
    ] + tsl_grpc_cc_dependencies() + if_libtpu(
        if_false = ["//xla/service:cpu_plugin"],
        if_true = [],
    ),
)

xla_cc_binary(
    name = "grpc_service_main_cpu",
    deps = [
        ":grpc_service_main_library",
        "//xla/service:cpu_plugin",
    ],
)

xla_cc_test(
    name = "grpc_client_test",
    srcs = ["grpc_client_test.cc"],
    data = [
        ":grpc_service_main_cpu",
    ],
    deps = [
        ":grpc_stub",
        "//xla/client",
        "//xla/client:xla_builder",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:net",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:subprocess",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_service",
    srcs = ["grpc_service.cc"],
    hdrs = ["grpc_service.h"],
    deps = [
        ":xla_service_cc_grpc_proto",
        ":xla_service_proto_cc",
        "//xla/service",
        "//xla/service:platform_util",
        "@tsl//tsl/distributed_runtime/rpc:grpc_util",
    ] + tsl_grpc_cc_dependencies(),
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xla_py_pb2",
#     has_services = True,
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":xla_service_proto"],
# )
#
# xla_py_grpc_library(
#     name = "xla_py_pb2_grpc",
#     srcs = [":xla_service_proto"],
#     deps = [":xla_py_pb2"],
# )
# copybara:uncomment_end
