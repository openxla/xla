"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "@tsl//tsl:tsl.bzl",
    "clean_dep",
    "if_tsl_link_protobuf",
    "tsl_copts",
)
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "tf_exec_properties",
)

def register_extension_info(**kwargs):
    pass

def xla_py_proto_library(**kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    _ignore = kwargs
    pass

def xla_py_grpc_library(**kwargs):
    # Note: we don't currently define any special targets for Python GRPC in OSS.
    _ignore = kwargs
    pass

ORC_JIT_MEMORY_MAPPER_TARGETS = []

def xla_py_test_deps():
    return []

def xla_cc_binary(deps = None, copts = tsl_copts(), **kwargs):
    if not deps:
        deps = []

    # TODO(ddunleavy): some of these should be removed from here and added to
    # specific targets.
    deps += [
        clean_dep("@com_google_protobuf//:protobuf"),
        "//xla:xla_proto_cc_impl",
        "//xla:xla_data_proto_cc_impl",
        "//xla/service:hlo_proto_cc_impl",
        "//xla/service:memory_space_assignment_proto_cc_impl",
        "//xla/service/gpu:backend_configs_cc_impl",
        "//xla/stream_executor:dnn_proto_cc_impl",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
        "@tsl//tsl/protobuf:protos_all_cc_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/util:determinism",
    ]
    native.cc_binary(deps = deps, copts = copts, **kwargs)

def xla_cc_test(
        name,
        deps = [],
        extra_copts = [],
        **kwargs):
    native.cc_test(
        name = name,
        copts = extra_copts,
        deps = deps + if_tsl_link_protobuf(
                   [],
                   [
                       clean_dep("@com_google_protobuf//:protobuf"),
                       # TODO(zacmustin): remove these in favor of more granular dependencies in each test.
                       "//xla:xla_proto_cc_impl",
                       "//xla:xla_data_proto_cc_impl",
                       "//xla/service:hlo_proto_cc_impl",
                       "//xla/service:memory_space_assignment_proto_cc_impl",
                       "//xla/service/gpu:backend_configs_cc_impl",
                       "//xla/stream_executor:dnn_proto_cc_impl",
                       "//xla/stream_executor:stream_executor_impl",
                       "//xla/stream_executor:device_id_utils",
                       "//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
                       "//xla/stream_executor/gpu:gpu_init_impl",
                       "@tsl//tsl/profiler/utils:time_utils_impl",
                       "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
                       "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
                       "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
                       "@tsl//tsl/protobuf:dnn_proto_cc_impl",
                       "@tsl//tsl/protobuf:protos_all_cc_impl",
                       "@tsl//tsl/platform:env_impl",
                       "@tsl//tsl/framework:allocator",
                       "@tsl//tsl/framework:allocator_registry_impl",
                       "@tsl//tsl/util:determinism",
                   ],
               ) +
               if_cuda_is_configured([
                   "//xla/stream_executor/cuda:cuda_stream",
                   "//xla/stream_executor/cuda:all_runtime",
                   "//xla/stream_executor/cuda:stream_executor_cuda",
               ]) +
               if_rocm_is_configured([
                   "//xla/stream_executor/gpu:gpu_stream",
                   "//xla/stream_executor/rocm:all_runtime",
                   "//xla/stream_executor/rocm:stream_executor_rocm",
               ]),
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

register_extension_info(
    extension = xla_cc_test,
    label_regex_for_dep = "{extension_name}",
)
