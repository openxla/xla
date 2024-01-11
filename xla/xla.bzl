"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@tsl//tsl:tsl.bzl",
    "if_oss",
    "tsl_copts",
    _tsl_clean_dep = "clean_dep",
)
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "if_static",
    "tf_exec_properties",
)
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

def clean_dep(target):
    """Returns string to 'target' in @{org_tensorflow,xla} repository.

    This is distinct from the clean_dep which appears in @{org_tensorflow,tsl}.
    TODO(ddunleavy,jakeharmon): figure out what to do with this after vendoring.
    """

    # A repo-relative label is resolved relative to the file in which the
    # Label() call appears, i.e. @org_tensorflow.
    return str(Label(target))

def xla_py_proto_library(**_kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    pass

ORC_JIT_MEMORY_MAPPER_TARGETS = []

def xla_py_test_deps():
    return []

# TODO(ddunleavy): some of these should be removed from here and added to
# specific targets.
# We actually shouldn't need this anymore post vendoring. If we build without
# `framework_shared_object` in the bazelrc all of this should be able to go
# away. The problem is making sure that all these impl deps are `if_static`'d
# appropriately throughout XLA.
_XLA_SHARED_OBJECT_SENSITIVE_DEPS = [
    "//xla:autotune_results_proto_cc_impl",
    "//xla:autotuning_proto_cc_impl",
    "//xla:xla_data_proto_cc_impl",
    "//xla:xla_proto_cc_impl",
    "//xla/service:buffer_assignment_proto_cc_impl",
    "//xla/service:hlo_proto_cc_impl",
    "//xla/service/gpu:backend_configs_cc_impl",
    "//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
    "//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl",
    "//xla/stream_executor:device_description_proto_cc_impl",
    "//xla/stream_executor:stream_executor_impl",
    "//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
    "@tsl//tsl/framework:allocator",
    "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
    "@tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
    "@tsl//tsl/protobuf:protos_all_cc_impl",
    "@tsl//tsl/util:determinism",
] + if_oss([
    _tsl_clean_dep("@com_google_protobuf//:protobuf"),
]) + if_static(extra_deps = [], otherwise = [
    "@tsl//tsl/framework:allocator_registry_impl",
    "@tsl//tsl/platform:env_impl",
    "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
    "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
]) + if_cuda_is_configured([
    "//xla/stream_executor/cuda:all_runtime",
    "//xla/stream_executor/cuda:cuda_stream",
    "//xla/stream_executor/cuda:stream_executor_cuda",
]) + if_rocm_is_configured([
    "//xla/stream_executor/gpu:gpu_stream",
    "//xla/stream_executor/rocm:all_runtime",
    "//xla/stream_executor/rocm:stream_executor_rocm",
])

def xla_cc_binary(deps = [], copts = tsl_copts(), **kwargs):
    native.cc_binary(deps = deps + _XLA_SHARED_OBJECT_SENSITIVE_DEPS, copts = copts, **kwargs)

def xla_cc_test(name, deps = [], **kwargs):
    native.cc_test(
        name = name,
        deps = deps + _XLA_SHARED_OBJECT_SENSITIVE_DEPS,
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

def auto_sharding_deps():
    return ["//xla/hlo/experimental/auto_sharding:auto_sharding_impl"]

def auto_sharding_solver_deps():
    return ["//xla/hlo/experimental/auto_sharding:auto_sharding_solver_impl"]

def xla_export_hlo_deps():
    return []

def xla_nvml_deps():
    return ["@local_config_cuda//cuda:nvml_headers"]

def xla_cub_deps():
    return ["@local_config_cuda//cuda:cub_headers"]

def xla_symbol_repository_deps():
    return []
