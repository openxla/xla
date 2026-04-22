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

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "scratch_space_lookup_table_proto",
    srcs = ["scratch_space_lookup_table.proto"],
)

cc_library(
    name = "cub_scratch_size_deviceless_lookup",
    srcs = ["cub_scratch_size_deviceless_lookup.cc"],
    hdrs = ["cub_scratch_size_deviceless_lookup.h"],
    deps = [
        ":cub_sort_utils",
        ":scratch_space_lookup_table_proto_cc",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cub_sort_utils",
    hdrs = ["cub_sort_utils.h"],
    visibility = ["//xla:internal"],
    deps = [],
)

xla_cc_test(
    name = "cub_scratch_size_deviceless_lookup_test",
    srcs = ["cub_scratch_size_deviceless_lookup_test.cc"],
    deps = [
        ":cub_scratch_size_deviceless_lookup",
        ":scratch_space_lookup_table_proto_cc",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/util/proto:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "cub_sort_utils_test",
    srcs = ["cub_sort_utils_test.cc"],
    deps = [
        ":cub_sort_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
