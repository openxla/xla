# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# fmt is an open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams.
licenses(["notice"])  # MIT

exports_files(["LICENSE"])

cc_library(
    name = "fmt",
    hdrs = glob(["include/fmt/*.h"]),
    copts = ["-fexceptions"],
    defines = [
        "FMT_HEADER_ONLY=1",
        "FMT_USE_USER_DEFINED_LITERALS=0",
    ],
    features = ["-use_header_modules"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "fmt_smoke_test",
    srcs = [
        "test/assert-test.cc",
        "test/header-only-test.cc",
        "test/test-main.cc",
    ],
    # fmt 8.1.1 only enables its consteval FMT_STRING checker above C++17, and
    # that path does not compile under clang 22 (see the _CXX17 comment in
    # //third_party/roc_mori:src_application.BUILD). Every consumer in this
    # repo compiles fmt at C++17, so smoke-test it the same way.
    copts = ["-std=c++17"],
    deps = [
        ":fmt",
        "@com_google_googletest//:gtest",
    ],
)
