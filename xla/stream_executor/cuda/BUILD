# Description:
#   CUDA-platform specific StreamExecutor support code.

load("@tsl//tsl:tsl.bzl", "if_google", "tsl_copts")
load("@tsl//tsl:tsl.default.bzl", "tsl_gpu_cc_test")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
    "tf_additional_cuda_driver_deps",
    "tf_additional_cuda_platform_deps",
    "tf_additional_cudnn_plugin_copts",
    "tf_additional_cudnn_plugin_deps",
    "tf_additional_gpu_compilation_copts",
)
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load(
    "@tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "cuda_platform_id",
    srcs = ["cuda_platform_id.cc"],
    hdrs = ["cuda_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "cuda_platform",
    srcs = if_cuda_is_configured(["cuda_platform.cc"]),
    hdrs = if_cuda_is_configured(["cuda_platform.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured(
        [
            ":cuda_driver",
            ":cuda_gpu_executor",
            ":cuda_platform_id",
            ":cuda_activation",
            "//xla/stream_executor",  # buildcleaner: keep
            "//xla/stream_executor:executor_cache",
            "//xla/stream_executor:multi_platform_manager",
            "//xla/stream_executor:stream_executor_pimpl_header",
            "//xla/stream_executor/lib",
            "//xla/stream_executor/platform",
        ],
    ) + tf_additional_cuda_platform_deps() + [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "cuda_diagnostics",
    srcs = if_cuda_is_configured(["cuda_diagnostics.cc"]),
    hdrs = if_cuda_is_configured(["cuda_diagnostics.h"]),
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
    ]) + ["@com_google_absl//absl/strings:str_format"],
)

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "ptxas_wrapper")

cc_library(name = "nvlink_wrapper")

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "fatbinary_wrapper")

cc_library(
    name = "cuda_driver",
    srcs = if_cuda_is_configured(["cuda_driver.cc"]),
    hdrs = if_cuda_is_configured(["cuda_driver.h"]),
    deps = if_cuda_is_configured([
        ":cuda_diagnostics",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor:device_options",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
    ] + tf_additional_cuda_driver_deps()) + select({
        # include dynamic loading implementation only when if_cuda_is_configured and build dynamically
        "@tsl//tsl:is_cuda_enabled_and_oss": ["cudart_stub"],
        "//conditions:default": ["@tsl//tsl/platform:cuda"],
    }) + [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

tsl_gpu_cc_test(
    name = "stream_search_test",
    size = "small",
    srcs = ["stream_search_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_impl",
        "//xla/stream_executor/host:host_platform",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

tsl_gpu_cc_test(
    name = "cuda_driver_test",
    srcs = ["cuda_driver_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        ":cuda_driver",
        "//xla/stream_executor/lib",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

tsl_gpu_cc_test(
    name = "memcpy_test",
    srcs = ["memcpy_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:stream_executor_impl",
        "//xla/stream_executor/lib",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

alias(
    name = "cudart_stub",
    actual = "@tsl//tsl/cuda:cudart_stub",
    visibility = ["//visibility:public"],
)

# The activation library is tightly coupled to the executor library.
# TODO(leary) split up cuda_gpu_executor.cc so that this can stand alone.
cc_library(
    name = "cuda_activation_header",
    hdrs = ["cuda_activation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "cuda_activation",
    srcs = [],
    hdrs = if_cuda_is_configured(["cuda_activation.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/gpu:gpu_activation",
        "//xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "cuda_gpu_executor_header",
    textual_hdrs = if_cuda_is_configured(["cuda_gpu_executor.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cuda_kernel",
        "//xla/stream_executor:event",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
    ]),
)

alias(
    name = "cublas_stub",
    actual = "@tsl//tsl/cuda:cublas_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cublas_lib",
    actual = "@tsl//tsl/cuda:cublas_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_lt_header",
    hdrs = if_cuda_is_configured([
        "cuda_blas_lt.h",
        "cuda_blas_utils.h",
    ]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:status",
        "//xla/stream_executor/lib",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/platform",
    ]) + ["@tsl//tsl/platform:errors"],
)

alias(
    name = "cublas_lt_stub",
    actual = "@tsl//tsl/cuda:cublas_lt_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cublas_lt_lib",
    actual = "@tsl//tsl/cuda:cublas_lt_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_plugin",
    srcs = if_cuda_is_configured([
        "cuda_blas.cc",
        "cuda_blas_lt.cc",
    ]),
    hdrs = if_cuda_is_configured([
        "cuda_blas.h",
        "cuda_blas_lt.h",
    ]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cublas_lib",
        ":cublas_lt_lib",
        ":cuda_activation",
        ":cuda_blas_utils",
        ":cuda_gpu_executor",
        ":cuda_helpers",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla:status_macros",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_timer_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "@tsl//tsl/platform:tensor_float_32_hdr_lib",
    ]) + if_static([
        "@tsl//tsl/platform:tensor_float_32_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "cuda_blas_utils",
    srcs = if_cuda_is_configured(["cuda_blas_utils.cc"]),
    hdrs = if_cuda_is_configured(["cuda_blas_utils.h"]),
    deps = if_cuda_is_configured([
        ":cublas_lib",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
    ]) + ["@tsl//tsl/platform:errors"],
)

alias(
    name = "cufft_stub",
    actual = "@tsl//tsl/cuda:cufft_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cufft_lib",
    actual = "@tsl//tsl/cuda:cufft_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufft_plugin",
    srcs = if_cuda_is_configured(["cuda_fft.cc"]),
    hdrs = if_cuda_is_configured(["cuda_fft.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cuda_activation_header",
        ":cuda_gpu_executor_header",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_helpers",
        ":cufft_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor:event",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
    ]) + ["@tsl//tsl/platform:errors"],
    alwayslink = True,
)

alias(
    name = "cudnn_stub",
    actual = "@tsl//tsl/cuda:cudnn_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cudnn_lib",
    actual = "@tsl//tsl/cuda:cudnn_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_dnn_headers",
    textual_hdrs = ["cuda_dnn.h"],
    deps = if_cuda_is_configured([
        ":cuda_activation_header",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/lib",
    ]),
)

cc_library(
    name = "cudnn_plugin",
    srcs = if_cuda_is_configured(["cuda_dnn.cc"]),
    hdrs = if_cuda_is_configured(["cuda_dnn.h"]),
    copts = tf_additional_cudnn_plugin_copts(),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_gpu_executor",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        ":cudnn_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@cudnn_frontend_archive//:cudnn_frontend",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
        "@tsl//tsl/cuda:cudnn_version",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@tsl//tsl/util:determinism",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_pimpl_header",
        "//xla/stream_executor:temporary_device_memory",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@tsl//tsl/util:env_var",
    ]) + tf_additional_cudnn_plugin_deps() + [
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)

alias(
    name = "curand_stub",
    actual = "@tsl//tsl/cuda:curand_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "curand_lib",
    actual = "@tsl//tsl/cuda:curand_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand_plugin",
    srcs = if_cuda_is_configured(["cuda_rng.cc"]),
    hdrs = if_cuda_is_configured(["cuda_rng.h"]),
    deps = if_cuda_is_configured([
        ":cuda_activation",
        ":cuda_gpu_executor",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_helpers",
        ":curand_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor:event",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:rng",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_rng_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
    ]),
    alwayslink = True,
)

alias(
    name = "cupti_stub",
    actual = "@tsl//tsl/cuda:cupti_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusolver_stub",
    actual = "@tsl//tsl/cuda:cusolver_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusolver_lib",
    actual = "@tsl//tsl/cuda:cusolver_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusparse_stub",
    actual = "@tsl//tsl/cuda:cusparse_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusparse_lib",
    actual = "@tsl//tsl/cuda:cusparse_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_kernel",
    srcs = if_cuda_is_configured(["cuda_kernel.cc"]),
    hdrs = if_cuda_is_configured(["cuda_kernel.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_pimpl_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
    ]),
)

# TODO(leary) we likely need to canonicalize/eliminate this.
cc_library(
    name = "cuda_helpers",
    textual_hdrs = if_cuda_is_configured(["cuda_helpers.h"]),
    deps = if_cuda_is_configured([
        "//xla/stream_executor/gpu:gpu_helpers_header",
    ]),
)

cc_library(
    name = "cuda_event",
    srcs = if_cuda_is_configured(["cuda_event.cc"]),
    hdrs = if_cuda_is_configured(["cuda_event.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        ":cuda_stream",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/lib",
    ]),
)

cc_library(
    name = "cuda_stream",
    srcs = [],
    hdrs = if_cuda_is_configured(["cuda_stream.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "cuda_timer",
    srcs = [],
    hdrs = if_cuda_is_configured(["cuda_timer.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        ":cuda_stream",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/gpu:gpu_timer",
        "//xla/stream_executor/lib",
    ]),
)

cc_library(
    name = "cuda_asm_compiler",
    srcs = if_cuda_is_configured(["cuda_asm_compiler.cc"]),
    copts = tf_additional_gpu_compilation_copts(),
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:str_format",
        "//xla/stream_executor/gpu:asm_compiler_header",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla:status_macros",
        "@tsl//tsl/platform:errors",
    ]),
)

cc_library(
    name = "cuda_gpu_executor",
    srcs = if_cuda_is_configured(["cuda_gpu_executor.cc"]),
    hdrs = if_cuda_is_configured(["cuda_gpu_executor.h"]),
    deps = if_cuda_is_configured([
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_event",
        ":cuda_kernel",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        ":cuda_asm_compiler",
        "@com_google_absl//absl/strings",
        "//xla/stream_executor:event",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor:stream_executor_pimpl_header",
        "//xla/stream_executor:timer",
        "//xla/stream_executor/gpu:asm_compiler",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
    ]) + [
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_plugin",
        ":cuda_driver",
        ":cuda_platform",
        ":cudnn_plugin",
        ":cufft_plugin",
        ":curand_plugin",
    ],
    alwayslink = 1,
)

# OSX framework for device driver access
cc_library(
    name = "IOKit",
    linkopts = ["-framework IOKit"],
)

cc_library(
    name = "stream_executor_cuda",
    deps = [
        "//xla/stream_executor:stream_executor_bundle",
    ] + if_google(
        select({
            # copybara:uncomment_begin(different config setting in OSS)
            # "//tools/cc_target_os:gce": [],
            # copybara:uncomment_end_and_comment_begin
            "//conditions:default": [
                "@local_config_cuda//cuda:cudart_static",  # buildcleaner: keep
                ":cuda_platform",
            ],
        }),
        [
            ":cudart_stub",
        ] + select({
            "@tsl//tsl:macos": ["IOKit"],
            "//conditions:default": [],
        }),
    ),
)

tsl_gpu_cc_test(
    name = "redzone_allocator_test",
    srcs = ["redzone_allocator_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        ":cuda_activation",
        ":cuda_gpu_executor",
        ":stream_executor_cuda",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:event",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:stream_executor_impl",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "//xla/stream_executor/gpu:redzone_allocator",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//tsl/profiler/utils:time_utils_impl",
    ],
)
