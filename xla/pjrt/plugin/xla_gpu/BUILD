# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_cuda_or_rocm", "if_google")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_gpu_pjrt_client",
    srcs = [
        "xla_gpu_pjrt_client.cc",
    ],
    hdrs = ["xla_gpu_pjrt_client.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]) + if_sycl([
        "TENSORFLOW_USE_SYCL=1",
    ]),
    deps = [
        ":xla_gpu_client_options",
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + if_cuda_or_rocm([
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
    ]) + if_sycl([
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
    ]),
)

cc_library(
    name = "xla_gpu_client_options",
    srcs = [],
    hdrs = ["xla_gpu_client_options.h"],
    deps = [
        "//xla/pjrt:host_memory_allocator",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/gpu:allocator_config",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "xla_gpu_allocator_config",
    srcs = [],
    hdrs = ["xla_gpu_allocator_config.h"],
    deps = [
        "//xla/pjrt/gpu:allocator_config",
    ],
)

xla_cc_test(
    name = "xla_gpu_allocator_config_test",
    srcs = ["xla_gpu_allocator_config_test.cc"],
    deps = [
        ":xla_gpu_allocator_config",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "xla_gpu_pjrt_client_test",
    srcs = ["xla_gpu_pjrt_client_test.cc"],
    backends = ["nvgpu_any"],
    tags = [
        "gpu",
        "no_oss",
    ] + if_google(["config-cuda-only"]),
    deps = [
        ":xla_gpu_pjrt_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_static_registration",
    srcs = ["gpu_static_registration.cc"],
    deps = [
        "//xla/pjrt/c:pjrt_c_api_gpu_internal",
        "//xla/pjrt/plugin:plugin_names",
        "//xla/pjrt/plugin:static_registration",
        "//xla/service:gpu_plugin",
    ],
    alwayslink = True,
)
