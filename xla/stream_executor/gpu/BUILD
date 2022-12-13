# Description:
#   GPU-platform specific StreamExecutor support code.

load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@tsl//tsl:tsl.bzl",
    "if_libtpu",
    "tsl_copts",
    "tsl_gpu_library",
)
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "@tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        "//third_party/tensorflow/compiler/tf2xla:__subpackages__",
        "//third_party/tensorflow/core/common_runtime/gpu:__subpackages__",
        "//xla:__subpackages__",
        "//xla/pjrt:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_activation_header",
    hdrs = ["gpu_activation.h"],
    deps = ["//xla/stream_executor/platform"],
)

cc_library(
    name = "gpu_activation",
    srcs = if_gpu_is_configured(["gpu_activation.cc"]),
    hdrs = if_gpu_is_configured(["gpu_activation.h"]),
    deps = if_gpu_is_configured([
        ":gpu_executor_header",
        ":gpu_activation_header",
        ":gpu_driver_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "gpu_diagnostics_header",
    hdrs = if_gpu_is_configured(["gpu_diagnostics.h"]),
    deps = [
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_driver_header",
    hdrs = if_gpu_is_configured(["gpu_driver.h"]),
    visibility = [
        "//third_party/tensorflow/core/common_runtime/gpu:__subpackages__",
        "//third_party/tensorflow/core/util/autotune_maps:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ],
    deps = [
        ":gpu_types_header",
        "//xla/stream_executor:device_options",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
    ] + if_libtpu(
        if_false = ["@local_config_cuda//cuda:cuda_headers"],
        if_true = [],
    ),
)

cc_library(
    name = "gpu_event_header",
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = if_gpu_is_configured([
        ":gpu_driver_header",
        ":gpu_stream_header",
        "//xla/stream_executor:event",
        "//xla/stream_executor/lib",
    ]),
)

cc_library(
    name = "gpu_event",
    srcs = if_gpu_is_configured(["gpu_event.cc"]),
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_executor_header",
    hdrs = if_gpu_is_configured(["gpu_executor.h"]),
    deps = [
        ":gpu_kernel_header",
        "//xla/stream_executor:event",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_pimpl_header",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = if_gpu_is_configured(["gpu_helpers.h"]),
    deps = [
        ":gpu_types_header",
        "@tsl//tsl/platform:logging",
    ],
)

tsl_gpu_library(
    name = "gpu_init",
    hdrs = [
        "gpu_init.h",
    ],
    visibility = [
        "@tsl//tsl:internal",
    ],
    deps = [
        "//xla/stream_executor/lib",
    ] + if_static(
        [":gpu_init_impl"],
    ),
)

tsl_gpu_library(
    name = "gpu_init_impl",
    srcs = [
        "gpu_init.cc",
    ],
    hdrs = [
        "gpu_init.h",
    ],
    copts = tsl_copts(),
    linkstatic = True,
    visibility = [
        "//third_party/tensorflow/compiler/tf2xla:__subpackages__",
        "//third_party/tensorflow/core/common_runtime/gpu:__subpackages__",
        "//third_party/tensorflow/stream_executor:__subpackages__",
        "//xla:__subpackages__",
    ],
    deps = [
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/lib",
        "@tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_kernel_header",
    hdrs = if_gpu_is_configured(["gpu_kernel.h"]),
    deps = [
        ":gpu_driver_header",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_pimpl_header",
        "//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_rng_header",
    hdrs = if_gpu_is_configured(["gpu_rng.h"]),
    deps = [
        ":gpu_types_header",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:rng",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "gpu_stream",
    srcs = if_gpu_is_configured(["gpu_stream.cc"]),
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "gpu_timer_header",
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//xla/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "gpu_timer",
    srcs = if_gpu_is_configured(["gpu_timer.cc"]),
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_types_header",
    hdrs = if_gpu_is_configured(["gpu_types.h"]),
    deps = [
        "//xla/stream_executor/platform",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "gpu_asm_opts",
    hdrs = ["gpu_asm_opts.h"],
    visibility = [
        "//third_party/tensorflow/core/kernels:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "asm_compiler_header",
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tsl_copts(),
    visibility = [
        "//third_party/tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//third_party/tensorflow/core/kernels:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:subprocess",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:cuda_libdevice_path",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]),
)

cc_library(
    name = "asm_compiler",
    srcs = if_gpu_is_configured(["asm_compiler.cc"]),
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tsl_copts(),
    visibility = [
        "//third_party/tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//third_party/tensorflow/core/kernels:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:subprocess",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:cuda_libdevice_path",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/lib",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_asm_compiler",
        "//xla/stream_executor/cuda:cuda_driver",
        "//xla/stream_executor/cuda:ptxas_wrapper",
        "//xla/stream_executor/cuda:nvlink_wrapper",
        "//xla/stream_executor/cuda:fatbinary_wrapper",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]),
)

cc_library(
    name = "redzone_allocator",
    srcs = if_gpu_is_configured(["redzone_allocator.cc"]),
    hdrs = if_gpu_is_configured(["redzone_allocator.h"]),
    copts = tsl_copts(),
    visibility = [
        "//third_party/tensorflow/core/kernels:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":asm_compiler",
        ":gpu_asm_opts",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@tsl//tsl/lib/math:math_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/framework:allocator",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_headers",
        "@tsl//tsl/platform:status",
    ]),
)

# TODO(tlongeri): Remove gpu_cudamallocasync_allocator header/impl split
tsl_gpu_library(
    name = "gpu_cudamallocasync_allocator_header",
    hdrs = ["gpu_cudamallocasync_allocator.h"],
    deps = [
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:device_id",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
    ],
)

tsl_gpu_library(
    name = "gpu_cudamallocasync_allocator",
    srcs = [
        "gpu_cudamallocasync_allocator.cc",
    ],
    hdrs = ["gpu_cudamallocasync_allocator.h"],
    cuda_deps = [
        "//xla/stream_executor/cuda:cuda_activation",
        "//xla/stream_executor/cuda:cuda_platform",
    ],
    deps = [
        ":gpu_init_impl",
        "//xla/stream_executor",
        "//xla/stream_executor:device_id_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:device_id",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/util:env_var",
    ],
)
