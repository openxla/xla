load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@tsl//tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("//xla:xla.bzl", "xla_cc_test")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
)

tf_proto_library(
    name = "protocol_proto",
    srcs = ["protocol.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    use_grpc_namespace = True,
)

cc_library(
    name = "protocol",
    hdrs = ["protocol.h"],
)

cc_library(
    name = "key_value_store",
    srcs = ["key_value_store.cc"],
    hdrs = ["key_value_store.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":key_value_store",
        ":protocol",
        ":protocol_cc_grpc_proto",
        ":util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@tsl//tsl/distributed_runtime/coordination:coordination_service",
        "@tsl//tsl/distributed_runtime/coordination:coordination_service_impl",
        "@tsl//tsl/distributed_runtime/rpc:async_service_interface",
        "@tsl//tsl/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:random",
        "@tsl//tsl/protobuf:coordination_config_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

xla_cc_test(
    name = "service_test",
    srcs = ["service_test.cc"],
    deps = [
        ":protocol_proto_cc",
        ":service",
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "client",
    srcs = [
        "client.cc",
    ],
    hdrs = [
        "client.h",
    ],
    deps = [
        ":protocol",
        ":protocol_cc_grpc_proto",
        ":util",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@tsl//tsl/distributed_runtime/coordination:coordination_client",
        "@tsl//tsl/distributed_runtime/coordination:coordination_service_agent",
        "@tsl//tsl/distributed_runtime/coordination:coordination_service_error_util",
        "@tsl//tsl/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:random",
        "@tsl//tsl/protobuf:coordination_config_proto_cc",
        "@tsl//tsl/protobuf:coordination_service_proto_cc",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//xla:status",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "distributed",
    srcs = ["distributed.cc"],
    hdrs = ["distributed.h"],
    deps = [
        ":client",
        ":service",
        "//xla:statusor",
    ] + tsl_grpc_cc_dependencies(),
)

xla_cc_test(
    name = "client_server_test",
    size = "small",
    srcs = ["client_server_test.cc"],
    deps = [
        ":client",
        ":distributed",
        ":protocol_proto_cc",
        ":service",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//xla:protobuf_util",
        "//xla:status_macros",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ] + tsl_grpc_cc_dependencies(),
)
