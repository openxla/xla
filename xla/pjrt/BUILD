load("@tsl//platform:rules_cc.bzl", "cc_library")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "@tsl//platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//third_party/tensorflow:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//third_party/tensorflow:internal",
        "//xla:friends",
    ],
    packages = [
        "//third_party/australis/...",
        "//third_party/py/jax/...",
    ],
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@tsl//platform:env",
        "@tsl//platform:logging",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    deps = [
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@tsl//platform:logging",
    ],
)

tf_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//xla:test",
        "@com_google_absl//absl/synchronization",
        "@tsl//platform:env",
        "@tsl//platform:env_impl",
        "@tsl//platform:logging",
        "@tsl//platform:test_main",
    ],
)

cc_library(
    name = "tracked_device_buffer",
    srcs = ["tracked_device_buffer.cc"],
    hdrs = ["tracked_device_buffer.h"],
    visibility = [
        "//learning/pathways/data_parallel:__pkg__",
        "//third_party/tensorflow:internal",
    ],
    deps = [
        ":event_pool",
        ":local_device_state",
        "//xla:shape_util",
        "//xla:types",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:event",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
        "@tsl//platform:logging",
    ],
)

tf_cc_test(
    name = "tracked_device_buffer_test",
    srcs = ["tracked_device_buffer_test.cc"],
    deps = [
        ":tracked_device_buffer",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/client:client_library",
        "//xla/service:cpu_plugin",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "@tsl//platform:test_main",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    deps = [
        ":event_pool",
        ":semaphore",
        ":worker_thread",
        "//xla:status",
        "//xla:util",
        "//xla/client:local_client",
        "//xla/stream_executor",
        "//xla/stream_executor:event",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@tsl//profiler/lib:traceme",
        "@tsl//protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "pjrt_client",
    srcs = ["pjrt_client.cc"],
    hdrs = ["pjrt_client.h"],
    visibility = ["//xla:friends"],
    deps = [
        ":pjrt_executable",
        ":pjrt_future",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_computation",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tsl//platform:errors",
        "@tsl//platform:fingerprint",
    ],
)

cc_library(
    name = "pjrt_client_test_common",
    testonly = 1,
    srcs = ["pjrt_client_test.cc"],
    hdrs = ["pjrt_client_test.h"],
    deps = [
        ":pjrt_client",
        "//xla:test",
        "//xla/client:xla_builder",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "pjrt_client_test_cpu",
    srcs = ["pjrt_client_test_cpu.cc"],
    deps = [
        ":pjrt_client_test_common",
        ":tfrt_cpu_pjrt_client",
        "@tsl//platform:test_main",
    ],
)

cc_library(
    name = "pjrt_executable",
    srcs = ["pjrt_executable.cc"],
    hdrs = ["pjrt_executable.h"],
    visibility = [":friends"],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/client:executable_build_options",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/strings",
        "@tsl//platform:statusor",
    ],
)

tf_cc_test(
    name = "pjrt_executable_test",
    srcs = ["pjrt_executable_test.cc"],
    deps = [
        ":compile_options_proto_cc",
        ":pjrt_executable",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pjrt_compiler",
    srcs = ["pjrt_compiler.cc"],
    hdrs = ["pjrt_compiler.h"],
    visibility = [":friends"],
    deps = [
        ":pjrt_client",
        ":pjrt_executable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "pjrt_compiler_test",
    srcs = ["pjrt_compiler_test.cc"],
    deps = [
        ":pjrt_client",
        ":pjrt_compiler",
        "//xla/client:xla_computation",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//xla:friends"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_placer",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//lib/monitoring:counter",
    ],
)

cc_library(
    name = "pjrt_stream_executor_client",
    srcs = ["pjrt_stream_executor_client.cc"],
    hdrs = ["pjrt_stream_executor_client.h"],
    visibility = ["//xla:friends"],
    deps = [
        ":event_pool",
        ":local_device_state",
        ":metrics",
        ":mlir_to_hlo",
        ":pjrt_client",
        ":pjrt_future",
        ":tracked_device_buffer",
        ":transpose",
        ":utils",
        "//xla:cpu_function_runtime",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor",
        "//xla/stream_executor:event",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//framework:allocator",
        "@tsl//platform:casts",
        "@tsl//platform:env",
        "@tsl//platform:errors",
        "@tsl//platform:fingerprint",
        "@tsl//platform:platform_port",
        "@tsl//platform:status",
        "@tsl//platform:statusor",
        "@tsl//profiler/lib:connected_traceme",
        "@tsl//profiler/lib:traceme",
        "@tsl//profiler/lib:traceme_encode",
    ],
)

tf_cc_test(
    name = "pjrt_stream_executor_client_test",
    srcs = ["pjrt_stream_executor_client_test.cc"],
    deps = [
        ":pjrt_client",
        ":pjrt_stream_executor_client",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:xla_builder",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tpu_client",
    srcs = ["tpu_client.cc"],
    hdrs = ["tpu_client.h"],
    visibility = ["//xla:friends"],
    deps = [
        ":local_device_state",
        ":pjrt_stream_executor_client",
        ":utils",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/service:computation_placer",
        "//xla/service:tpu_computation_placer",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/tpu:tpu_executable",
        "//xla/stream_executor/tpu:tpu_executable_interface",
        "//xla/stream_executor/tpu:tpu_executor",
        "//xla/stream_executor/tpu:tpu_executor_interface",
        "//xla/stream_executor/tpu:tpu_initializer_helper",
        "//xla/stream_executor/tpu:tpu_on_demand_compiler",
        "//xla/stream_executor/tpu:tpu_platform_interface",
        "//xla/stream_executor/tpu:tpu_topology_external",
        "//xla/stream_executor/tpu:tpu_transfer_manager",
        "@tsl//platform:casts",
        "@tsl//platform:errors",
    ],
)

cc_library(
    name = "interpreter_device",
    srcs = ["interpreter_device.cc"],
    hdrs = ["interpreter_device.h"],
    deps = [
        ":pjrt_stream_executor_client",
        "//xla:statusor",
        "//xla/client:client_library",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pjrt_plugin_device_client",
    deps = [
        ":pjrt_plugin_device_client_headers",
        "//xla/pjrt/plugin",
    ],
)

cc_library(
    name = "pjrt_plugin_device_client_headers",
    hdrs = ["pjrt_plugin_device_client.h"],
    deps = [
        "//xla:statusor",
    ],
)

cc_library(
    name = "mlir_to_hlo",
    srcs = ["mlir_to_hlo.cc"],
    hdrs = ["mlir_to_hlo.h"],
    visibility = [":friends"],
    deps = [
        "//xla:status",
        "//xla/client:xla_computation",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "pjrt_future",
    hdrs = ["pjrt_future.h"],
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "tracked_tfrt_cpu_device_buffer",
    srcs = ["tracked_tfrt_cpu_device_buffer.cc"],
    hdrs = ["tracked_tfrt_cpu_device_buffer.h"],
    deps = [
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
        "@tsl//platform:platform_port",
    ],
)

tf_cc_test(
    name = "tracked_tfrt_cpu_device_buffer_test",
    srcs = ["tracked_tfrt_cpu_device_buffer_test.cc"],
    deps = [
        ":tracked_tfrt_cpu_device_buffer",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "tfrt_cpu_pjrt_client",
    srcs = ["tfrt_cpu_pjrt_client.cc"],
    hdrs = ["tfrt_cpu_pjrt_client.h"],
    visibility = [
        "//xla:friends",
    ],
    deps = [
        ":mlir_to_hlo",
        ":pjrt_client",
        ":pjrt_executable",
        ":pjrt_future",
        ":semaphore",
        ":tracked_tfrt_cpu_device_buffer",
        ":transpose",
        ":utils",
        ":worker_thread",
        "//third_party/eigen3",  # TODO(zhangqiaorjc): Remove if use TFRT threadpool.
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_computation",
        "//xla/service:buffer_assignment",
        "//xla/service:computation_placer_hdr",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_proto_cc",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/cpu:cpu_xfeed",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tsl//platform:denormal",
        "@tsl//platform:errors",
        "@tsl//platform:setround",
        "@tsl//profiler/lib:connected_traceme",
        "@tsl//profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "tfrt_cpu_pjrt_client_test",
    srcs = ["tfrt_cpu_pjrt_client_test.cc"],
    deps = [
        ":tfrt_cpu_pjrt_client",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@tsl//platform:logging",
    ],
)

tf_cc_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.cc"],
    deps = [
        ":lru_cache",
        "//xla:test",
        "@tsl//platform:logging",
        "@tsl//platform:test_main",
    ],
)

cc_library(
    name = "transpose",
    srcs = [
        "transpose.cc",
        "transpose_kernels.h",
    ],
    hdrs = ["transpose.h"],
    visibility = [":friends"],
    deps = [
        ":lru_cache",
        "//third_party/eigen3",
        "//xla:permutation_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@tsl//platform:logging",
        "@tsl//profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":transpose",
        "//third_party/eigen3",
        "//xla:array",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/numeric:int128",
        "@tsl//platform:env",
        "@tsl//platform:test_benchmark",
        "@tsl//platform:test_main",
        "@tsl//protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "pjrt_c_api_client",
    srcs = ["pjrt_c_api_client.cc"],
    hdrs = ["pjrt_c_api_client.h"],
    deps = [
        ":pjrt_client",
        ":pjrt_executable",
        ":pjrt_future",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",  # TODO(skyewm): remove
        "//xla/stream_executor/tpu:c_api_conversions",  # TODO(b/238999986): Remove this.
        "//xla/stream_executor/tpu:pjrt_api",
        "//xla/stream_executor/tpu:tpu_initializer_helper",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:BytecodeWriter",
        "@tsl//platform:status",
    ],
)

cc_library(
    name = "host_callback",
    srcs = ["host_callback.cc"],
    hdrs = ["host_callback.h"],
    visibility = [":friends"],
    deps = [":pjrt_client"],
)

tf_cc_test(
    name = "host_callback_test",
    srcs = ["host_callback_test.cc"],
    deps = [
        ":host_callback",
        ":pjrt_client",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@tsl//lib/core:status_test_util",
    ],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    cc_api_version = 2,
    protodeps = [
        "//xla:xla_data_proto",
        "//xla:xla_proto",
    ],
    visibility = ["//visibility:public"],
)
