load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cuda_library(
    name = "gpu_bandwidth_kernels_cuda",
    srcs = ["gpu_bandwidth_kernels.cu.cc"],
    hdrs = ["gpu_bandwidth_kernels.h"],
    tags = ["cuda-only"],
    deps = [
        "//xla/stream_executor/cuda:cuda_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "gpu_bandwidth_kernels_stub",
    srcs = ["gpu_bandwidth_kernels_stub.cc"],
    hdrs = ["gpu_bandwidth_kernels.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

# Portable facade: resolves to the CUDA kernel when CUDA is configured and to a
# stub returning `kUnimplemented` otherwise, so dependents need no `#if`.
cc_library(
    name = "gpu_bandwidth_kernels",
    hdrs = ["gpu_bandwidth_kernels.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ] + if_cuda_is_configured(
        [":gpu_bandwidth_kernels_cuda"],
        no_cuda = [":gpu_bandwidth_kernels_stub"],
    ),
)

xla_test(
    name = "gpu_bandwidth_kernels_test",
    srcs = ["gpu_bandwidth_kernels_test.cc"],
    backends = [
        "a100",
        "h100",
    ],
    deps = [
        ":gpu_bandwidth_kernels",
        "//xla/stream_executor:cuda_platform",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_bandwidth_benchmark",
    srcs = ["gpu_bandwidth_benchmark.cc"],
    hdrs = ["gpu_bandwidth_benchmark.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/stream_executor/gpu:gpu_init",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ] + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
)

xla_test(
    name = "gpu_bandwidth_benchmark_test",
    srcs = ["gpu_bandwidth_benchmark_test.cc"],
    backends = [
        "h100",
        "amdgpu_any",
    ],
    deps = [
        ":gpu_bandwidth_benchmark",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)
