load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//third_party/tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_py_grpc_library",
)
load(
    "//tensorflow/tsl:tsl.bzl",
    "if_libtpu",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//third_party/tensorflow:internal"],
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
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_util",
        "//xla:service_interface",
        "//xla:xla_data_cc_grpc_proto",
        "//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "grpc_service_main_library",
    srcs = ["grpc_service_main.cc"],
    deps = [
        ":grpc_service",
        "@com_google_absl//absl/strings:str_format",
        "//xla/service:platform_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/util:command_line_flags",
    ] + tsl_grpc_cc_dependencies() + if_libtpu(
        if_false = ["//xla/service:cpu_plugin"],
        if_true = [],
    ),
)

xla_cc_binary(
    name = "grpc_service_main_cpu",
    deps = [
        ":grpc_service_main_library",
        "//tensorflow/tsl/protobuf:dnn_proto_cc_impl",
        "//xla/service:cpu_plugin",
    ],
)

tf_cc_test(
    name = "grpc_client_test",
    srcs = ["grpc_client_test.cc"],
    data = [
        ":grpc_service_main_cpu",
    ],
    deps = [
        ":grpc_stub",
        "@com_google_absl//absl/strings:str_format",
        "//xla/client",
        "//xla/client:xla_builder",
        "//xla/tests:literal_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:net",
        "//tensorflow/tsl/platform:subprocess",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:path",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_service",
    srcs = ["grpc_service.cc"],
    hdrs = ["grpc_service.h"],
    deps = [
        ":xla_service_proto_cc",
        ":xla_service_cc_grpc_proto",
        "//xla/service",
        "//xla/service:platform_util",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_util",
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
