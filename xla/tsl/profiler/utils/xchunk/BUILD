# Copyright 2026 The OpenXLA Authors.
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
# ==============================================================================

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl/profiler:friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "xchunk",
    hdrs = ["xchunk.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts(),
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "xchunk_visitor",
    hdrs = ["xchunk_visitor.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts(),
    deps = [
        ":xchunk",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "xplane_to_xchunk",
    srcs = ["xplane_to_xchunk.cc"],
    hdrs = ["xplane_to_xchunk.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts(),
    deps = [
        ":xchunk",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "xchunk_wire_scanner",
    srcs = ["xchunk_wire_scanner.cc"],
    hdrs = ["xchunk_wire_scanner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/varint:varint_reading",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "xchunk_test",
    srcs = ["xchunk_test.cc"],
    deps = [
        ":xchunk",
        ":xchunk_visitor",
        ":xplane_to_xchunk",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_cc_test(
    name = "xchunk_wire_scanner_test",
    srcs = ["xchunk_wire_scanner_test.cc"],
    deps = [
        ":xchunk_wire_scanner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
