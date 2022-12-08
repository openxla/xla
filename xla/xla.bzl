"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "@tsl//:tsl.bzl",
    "clean_dep",
    # "if_tsl_link_protobuf",
)

# load("@tsl//platform:build_config.bzl", "tsl_cc_test")
load(
    "//third_party/tensorflow:tensorflow.bzl",
    "lrt_if_needed",
    "tf_binary_additional_srcs",
    "tf_binary_dynamic_kernel_deps",
    "tf_binary_dynamic_kernel_dsos",
    "tf_copts",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)
load(
    "//third_party/tensorflow/core/platform:build_config_root.bzl",
    "tf_exec_properties",
)
load(
    "//third_party/tensorflow/core/platform:rules_cc.bzl",
    "cc_test",
)

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("//third_party/tensorflow:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        clean_dep("//third_party/tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

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

def xla_cc_binary(deps = None, **kwargs):
    if not deps:
        deps = []

    # TODO(ddunleavy): some of these should be removed from here and added to
    # specific targets.
    deps += [
        clean_dep("//google/protobuf"),
        "//xla:xla_proto_cc_impl",
        "//xla:xla_data_proto_cc_impl",
        "//xla/service:hlo_proto_cc_impl",
        "//xla/service/gpu:backend_configs_cc_impl",
        "//xla/stream_executor:dnn_proto_cc_impl",
        "@tsl//platform:env_impl",
        "@tsl//profiler/utils:time_utils_impl",
        "@tsl//profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//protobuf:protos_all_cc_impl",
    ]
    native.cc_binary(deps = deps, **kwargs)

def xla_cc_test(
        name,
        srcs,
        deps,
        data = [],
        extra_copts = [],
        suffix = "",
        linkopts = lrt_if_needed(),
        kernels = [],
        **kwargs):
    cc_test(
        name = "%s%s" % (name, suffix),
        srcs = srcs + tf_binary_additional_srcs(),
        copts = tf_copts() + extra_copts,
        linkopts = select({
            clean_dep("//third_party/tensorflow:android"): [
                "-pie",
            ],
            clean_dep("//third_party/tensorflow:windows"): [],
            clean_dep("//third_party/tensorflow:macos"): [
                "-lm",
            ],
            "//conditions:default": [
                "-lpthread",
                "-lm",
            ],
            clean_dep("//tensorflow/third_party/compute_library:build_with_acl"): ["-fopenmp"],
        }) + linkopts + _rpath_linkopts(name),
        deps = deps + tf_binary_dynamic_kernel_deps(kernels) + if_mkl_ml(
            [
                clean_dep("//third_party/mkl:intel_binary_blob"),
            ],
        ) + if_tsl_link_protobuf(
            [],
            [
                # TODO(zacmustin): remove these in favor of more granular dependencies in each test.
                "//xla/service:cpu_plugin",
            ],
        ),
        data = data +
               tf_binary_dynamic_kernel_dsos() +
               tf_binary_additional_srcs(),
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

"""+ if_tsl_link_protobuf(
            [],
            [
                # TODO(zacmustin): remove these in favor of more granular dependencies in each test.
                "//xla:xla_proto_cc_impl",
                "//xla:xla_data_proto_cc_impl",
                "//xla/service:hlo_proto_cc_impl",
                "//xla/service/gpu:backend_configs_cc_impl",
                "//xla/stream_executor:device_description_proto_cc_impl",
                "//xla/stream_executor:dnn_proto_cc_impl",
                "//xla/stream_executor:stream_executor_impl",
                "//xla/stream_executor/cuda:cublas_plugin",
                "//xla/stream_executor/gpu:gpu_init_impl",
                "@tsl//framework:allocator",
                "@tsl//framework:allocator_registry_impl",
                "@tsl//platform:env_impl",
                "@tsl//platform:tensor_float_32_utils",
                "@tsl//profiler/utils:time_utils_impl",
                "@tsl//profiler/backends/cpu:annotation_stack_impl",
                "@tsl//profiler/backends/cpu:traceme_recorder_impl",
                "@tsl//protobuf:autotuning_proto_cc_impl",
                "@tsl//protobuf:dnn_proto_cc_impl",
                "@tsl//protobuf:protos_all_cc_impl",
                "@tsl//util:determinism",
            ],
        )
"""
