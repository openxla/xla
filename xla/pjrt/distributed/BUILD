load("//xla:xla.bzl", "xla_cc_test")
load("@tsl//tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("@tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
)

tf_proto_library(
    name = "protocol_proto",
    srcs = ["protocol.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":topology_util",
        ":util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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
    name = "topology_util_test",
    srcs = ["topology_util_test.cc"],
    deps = [
        ":in_memory_key_value_store",
        ":protocol_proto_cc",
        ":topology_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
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
        ":key_value_store_interface",
        ":util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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

cc_library(
    name = "topology_util",
    srcs = ["topology_util.cc"],
    hdrs = ["topology_util.h"],
    deps = [
        ":key_value_store_interface",
        ":protocol_proto_cc",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
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
        ":topology_util",
        "//xla:protobuf_util",
        "//xla:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "key_value_store_interface",
    hdrs = ["key_value_store_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "in_memory_key_value_store",
    srcs = ["in_memory_key_value_store.cc"],
    hdrs = ["in_memory_key_value_store.h"],
    deps = [
        ":key_value_store_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
