# Copyright 2023 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//xla:xla.bzl", "xla_cc_test")
load("@tsl//tsl:tsl.bzl", "if_google")
load("@tsl//tsl/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc_credentials",
    srcs = ["grpc_credentials.cc"],
    hdrs = ["grpc_credentials.h"],
    defines = select({
        # copybara:uncomment(google-only) "//tools/cc_target_os:gce": ["CC_TARGET_OS_GCE=1"],
        "//conditions:default": [],
    }),
    deps = ["@com_github_grpc_grpc//:grpc++"],
)

tf_proto_library(
    name = "types_proto",
    srcs = ["types.proto"],
    protodeps = ["//xla/python/ifrt:serdes_proto"],
)

tf_proto_library(
    name = "ifrt_service_proto",
    srcs = ["ifrt_service.proto"],
    protodeps = [
        ":types_proto",
        "//xla:xla_data_proto",
        "//xla/pjrt:execute_options_proto",
        "//xla/python/ifrt:serdes_proto",
        "@tsl//tsl/protobuf:status_proto",
    ],
)

tf_proto_library(
    name = "grpc_ifrt_service_proto",
    srcs = ["grpc_ifrt_service.proto"],
    has_services = True,
    create_go_proto = False,
    create_grpc_library = True,
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = [":ifrt_service_proto"],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    deps = [
        ":ifrt_service_proto_cc",
        ":types_proto_cc",
        "//xla/pjrt:pjrt_common",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:statusor",
    ] + if_google(["@com_google_absl//absl/types:source_location"]),
)

xla_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        ":types_proto_cc",
        "//xla/pjrt:pjrt_common",
        "//xla/python/ifrt",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "array_util",
    srcs = ["array_util.cc"],
    hdrs = ["array_util.h"],
    deps = [
        "//xla/python/ifrt",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "array_util_test",
    srcs = ["array_util_test.cc"],
    deps = [
        ":array_util",
        "//xla/python/ifrt",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
    ],
)

# common_serdes is a collection of all common libraries that register SerDes implementations.
cc_library(
    name = "common_serdes",
    deps = ["//xla/python/pjrt_ifrt:xla_program_serdes"],
    alwayslink = True,
)

cc_library(
    name = "proto_util",
    srcs = ["proto_util.cc"],
    hdrs = ["proto_util.h"],
    deps = [
        ":ifrt_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:status_to_from_proto",
    ],
)
