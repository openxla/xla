# Copyright 2025 The OpenXLA Authors.
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
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "golden_baselines",
    hdrs = ["golden_baselines.h"],
)

cc_library(
    name = "accuracy_budget",
    hdrs = ["accuracy_budget.h"],
)

cc_library(
    name = "accuracy_test_framework",
    testonly = True,
    hdrs = ["accuracy_test_framework.h"],
    deps = [
        ":golden_baselines",
        "//xla:fp_util",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "intrinsic_accuracy_test",
    srcs = ["intrinsic_accuracy_test.cc"],
    deps = [
        ":accuracy_budget",
        ":accuracy_test_framework",
        ":golden_baselines",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:exp",
        "//xla/codegen/intrinsic:log1p",
        "//xla/codegen/intrinsic:rsqrt",
        "//xla/codegen/intrinsic:simple_jit_runner",
        "//xla/codegen/intrinsic:tanh",
        "//xla/codegen/intrinsic:type",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Support",
    ],
)
