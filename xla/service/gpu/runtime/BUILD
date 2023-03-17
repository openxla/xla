load("@tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

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
    name = "cholesky",
    srcs = ["cholesky.cc"],
    hdrs = ["cholesky.h"],
    deps = [
        ":support",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/service/gpu:cholesky_thunk",
        "//xla/service/gpu:gpu_asm_opts_util",
    ],
)

cc_library(
    name = "collectives",
    srcs = ["collectives.cc"],
    hdrs = ["collectives.h"],
    deps = [
        ":support",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:computation_placer_hdr",
        "//xla/service:executable",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/service/gpu:nccl_collective_thunks",
        "//xla/stream_executor:event",
        "//xla/stream_executor:executor_cache",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        ":support",
        "//xla:status",
        "//xla:xla_proto_cc",
        "//xla/mlir/runtime/transforms:custom_call_encoding",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/service/gpu:gpu_asm_opts_util",
        "//xla/service/gpu:gpu_conv_algorithm_picker",
        "//xla/service/gpu:gpu_conv_runner",
        "//xla/service/gpu:gpu_serializable_autotuner",
        "//xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "conv_reorder",
    srcs = ["conv_reorder.cc"],
    hdrs = ["conv_reorder.h"],
    deps = [
        ":support",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    deps = [
        ":support",
        ":triangular_solve",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:custom_call_status_internal",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:custom_call_target_registry",
        "//xla/service:executable",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor/gpu:gpu_stream_header",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":cholesky",
        ":collectives",
        ":conv",
        ":conv_reorder",
        ":cublas_lt_matmul",
        ":custom_call",
        ":fft",
        ":gemm",
        ":graph_launch",
        ":io_feed",
        ":kernel_launch",
        ":memcpy",
        ":memset",
        ":send_recv",
        ":support",
        ":tracing",
        "//xla:xla_proto_cc",
        "//xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//xla/runtime:executable",
        "//xla/runtime:ffi",
        "//xla/runtime:jit_executable",
        "//xla/runtime:module_registry",
        "//xla/service:executable",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//xla/stream_executor/gpu:gpu_stream",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/protobuf:dnn_proto_cc",
    ],
)

cc_library(
    name = "fft",
    srcs = ["fft.cc"],
    hdrs = ["fft.h"],
    deps = [
        ":support",
        "//xla/mlir/runtime/transforms:custom_call_encoding",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:state",
        "//xla/service/gpu:fft_thunk",
        "//xla/stream_executor:fft",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
    ],
)

cc_library(
    name = "gemm",
    srcs = ["gemm.cc"],
    hdrs = ["gemm.h"],
    deps = [
        ":support",
        "@com_google_absl//absl/container:node_hash_map",
        "//xla:status",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:state",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
    ] + if_cuda_is_configured([
        "//xla/service/gpu:gemm_algorithm_picker",
        "//xla/stream_executor/gpu:redzone_allocator",
    ]),
)

cc_library(
    name = "graph_launch",
    srcs = ["graph_launch.cc"],
    hdrs = ["graph_launch.h"],
    deps = [
        ":conv",
        ":gemm",
        ":kernel_launch",
        ":support",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
        "//xla:types",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:state",
        "//xla/service:executable",
        "//xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//xla/stream_executor",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_graph",
    ]),
)

cc_library(
    name = "io_feed",
    srcs = ["io_feed.cc"],
    hdrs = ["io_feed.h"],
    deps = [
        ":support",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/service/gpu:io_feed_manager",
    ],
)

cc_library(
    name = "kernel_launch",
    srcs = ["kernel_launch.cc"],
    hdrs = ["kernel_launch.h"],
    deps = [
        ":support",
        "//xla:types",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:state",
        "//xla/service:executable",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "cublas_lt_matmul",
    srcs = ["cublas_lt_matmul.cc"],
    hdrs = ["cublas_lt_matmul.h"],
    deps = [
        ":support",
        "//xla:xla_proto_cc",
        "//xla/mlir/runtime/transforms:custom_call_encoding",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:logical_result",
        "//xla/runtime:state",
        "//xla/service:executable",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/cuda:cublas_lt_header",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "memcpy",
    srcs = ["memcpy.cc"],
    hdrs = ["memcpy.h"],
    deps = [
        ":support",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/service/gpu:io_feed_manager",
    ],
)

cc_library(
    name = "memset",
    srcs = ["memset.cc"],
    hdrs = ["memset.h"],
    deps = [
        ":support",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/service/gpu:io_feed_manager",
    ],
)

cc_library(
    name = "support",
    hdrs = ["support.h"],
    deps = [
        "//xla:shape_util",
        "//xla/mlir/runtime/transforms:custom_call_encoding",
        "//xla/runtime:custom_call",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "send_recv",
    srcs = ["send_recv.cc"],
    hdrs = ["send_recv.h"],
    deps = [
        ":support",
        "//xla/mlir/runtime/transforms:custom_call_encoding",
        "//xla/mlir_hlo",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/stream_executor:event",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@tf_runtime//:async_value",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
    ],
)

cc_library(
    name = "tracing",
    srcs = ["tracing.cc"],
    hdrs = ["tracing.h"],
    deps = [
        ":support",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/runtime:tracing",
        "//xla/runtime:type_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/profiler/lib:scoped_annotation_stack",
    ],
)

cc_library(
    name = "triangular_solve",
    srcs = ["triangular_solve.cc"],
    hdrs = ["triangular_solve.h"],
    deps = [
        ":support",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:executable",
        "//xla/service:executable",
        "//xla/service/gpu:gpu_asm_opts_util",
        "//xla/service/gpu:triangular_solve_thunk",
        "@tsl//tsl/platform:human_readable_json",
    ],
)
