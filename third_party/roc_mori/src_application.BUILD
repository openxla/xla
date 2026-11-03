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

# Overlay BUILD for @roc_mori//src/application. Symlinked over the extracted
# tarball's src/application/BUILD.bazel by tf_http_archive (see workspace.bzl).
#
# Mirrors src/application/CMakeLists.txt: a single cc_library named
# mori_application. All sources are host C++ (CMake sets LANGUAGE CXX on
# them); they just call into the ROCm runtime APIs (HIP, HSA, rocm-smi,
# hsakmt) plus ibverbs. No device kernels live here.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

_IBV_SHIM_SRC = "transport/rdma/providers/ibverbs/ibv_shim.cpp"

# Every MORI TU here includes mori/utils/mori_log.hpp -> spdlog -> @fmt, and
# @fmt is pinned to 8.1.1. fmt 8 only enables its consteval FMT_STRING checker
# when __cplusplus > 201703L, and that code path does a pointer subtraction
# across two evaluations of the same string literal (core.h:2930,
# `begin - &*context_.begin()`). Clang 22 no longer gives those evaluations a
# common object, so the subtraction stops being a constant expression and the
# consteval constructor fails to compile. Upstream MORI builds as C++17
# (CMAKE_CXX_STANDARD 17), so pinning matches how AMD ships it; this mirrors
# what //third_party/rapids_logger:rapids_logger already does for the CUDA
# side of the same spdlog -> fmt chain. Target copts come after the global
# --cxxopt=-std=c++20, so this wins. Drop it once @fmt moves past 10.x --
# note that also needs format_as() overloads for the HIP enums MORI logs,
# since fmt 10 removed implicit enum-to-integer formatting.
_CXX17 = ["-std=c++17"]

# Mirrors CMake's mori_ibv_shim OBJECT library. The shim defines the core ibv_*
# symbols and dlopen()s libibverbs.so.1 at runtime; it must be compiled with
# hidden visibility so those definitions never interpose a real libibverbs
# loaded by another library in the same process (e.g. RCCL, UCX, MPI).
cc_library(
    name = "mori_ibv_shim",
    srcs = [_IBV_SHIM_SRC],
    copts = [
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
    ] + _CXX17,
    linkstatic = True,
    visibility = ["//visibility:private"],
    deps = [
        "@roc_mori//:ibverbs",
        "@roc_mori//:mori_application_headers",
        "@spdlog",
    ],
)

filegroup(
    name = "mori_ibv_shim_objs",
    srcs = [":mori_ibv_shim"],
    output_group = "compilation_outputs",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "mori_application",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "bootstrap/mpi_bootstrap.cpp",
            "bootstrap/torch_bootstrap.cpp",
            _IBV_SHIM_SRC,
        ],
    ) + [
        ":mori_ibv_shim_objs",
    ],
    # Hide the GCC 8 std::filesystem code (libstdc++fs.a and inline members):
    # it is ABI-incompatible with the libstdc++.so that other libraries use.
    copts = ["-fvisibility-inlines-hidden"] + _CXX17,
    linkopts = [
        "-ldl",
        "-Wl,--exclude-libs,libstdc++fs.a",
    ],
    deps = [
        "@roc_mori//:mori_application_headers",
        # symmetric_memory.cpp includes mori/shmem/internal.hpp.
        "@roc_mori//:mori_shmem_headers",
        # CMake hip::host: libamdhip64.so + HIP host headers.
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:hsa_runtime",
        "@local_config_rocm//rocm:hsakmt",
        # Vendored <infiniband/verbs.h>; libibverbs itself is dlopen()ed by
        # :mori_ibv_shim at runtime.
        "@roc_mori//:ibverbs",
        # System libdrm + libdrm_amdgpu. Required transitively by libhsakmt.a
        # (amdgpu_get_marketing_name, amdgpu_query_gpu_info, amdgpu_*, drmClose).
        "@roc_mori//:libdrm",
        # System libnuma. Required transitively by libhsakmt.a (numa_available,
        # numa_max_node, numa_bitmask_*, mbind, numa_node_size64).
        "@roc_mori//:libnuma",
        # mori_logging interface lib in CMake is spdlog::spdlog_header_only.
        "@spdlog",
    ],
)
