load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl/platform:build_config.bzl", "pyx_library")
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//xla:xla.bzl",
    "xla_cc_test",
    "xla_py_test_deps",
)
load(
    "@tsl//tsl:tsl.bzl",
    "if_cuda_or_rocm",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@tsl//tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//third_party/tensorflow:pytype.default.bzl", "pytype_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//third_party/tensorflow:internal",
        "//xla:friends",
    ],
)

pytype_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":xla_extension"],
)

exports_files(["xla_client.pyi"])

pyx_library(
    name = "custom_call_for_test",
    testonly = True,
    srcs = ["custom_call_for_test.pyx"],
)

py_test(
    name = "xla_client_backend_independent_test",
    srcs = ["xla_client_backend_independent_test.py"],
    python_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

py_library(
    name = "xla_client_test",
    testonly = 1,
    srcs = ["xla_client_test.py"],
    srcs_version = "PY3",
    visibility = [":friends"],
    deps = [
        ":xla_client",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":custom_call_for_test",
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_test(
    name = "xla_client_test_gpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=gpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "config-cuda-only",
        "no_oss",
        "requires-gpu-nvidia",
    ],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "status_casters",
    hdrs = ["status_casters.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":exceptions",
        "//xla:status",
        "//xla:statusor",
        "@pybind11",
    ],
)

cc_library(
    name = "exceptions",
    hdrs = ["exceptions.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//xla:status",
    ],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        ":exceptions",
        ":status_casters",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/python/lib/core:bfloat16_lib",
    ],
)

cc_library(
    name = "python_ref_manager",
    srcs = ["python_ref_manager.cc"],
    hdrs = ["python_ref_manager.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cc_library(
    name = "python_utils",
    hdrs = ["python_utils.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//xla:status_macros",
        "//xla:util",
        "@pybind11",
    ],
)

cc_library(
    name = "traceback",
    srcs = ["traceback.cc"],
    hdrs = ["traceback.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        ":exceptions",
        ":python_ref_manager",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "pprof_profile_builder",
    srcs = ["pprof_profile_builder.cc"],
    hdrs = ["pprof_profile_builder.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":traceback",
        "//xla:statusor",
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@pybind11",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:profile_proto_cc",
    ],
)

cc_library(
    name = "py_client",
    srcs = [
        "callback.cc",
        "py_array.cc",
        "py_buffer.cc",
        "py_client.cc",
        "py_executable.cc",
        "py_values.cc",
        "sharded_device_array.cc",
        "sharding.cc",
    ] + if_cuda_or_rocm([
        "py_client_gpu.cc",
    ]),
    hdrs = [
        "callback.h",
        "py_array.h",
        "py_buffer.h",
        "py_client.h",
        "py_executable.h",
        "py_values.h",
        "sharded_device_array.h",
        "sharding.h",
    ] + if_cuda_or_rocm([
        "py_client_gpu.h",
    ]),
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = if_cuda(["GOOGLE_CUDA=1"]) +
              select({
                  ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
                  "//conditions:default": [],
              }),
    features = ["-use_header_modules"],
    deps = [
        ":exceptions",
        "@tsl//tsl/python/lib/core:numpy",
        ":pprof_profile_builder",
        ":python_ref_manager",
        ":python_utils",
        ":status_casters",
        ":traceback",
        ":transfer_guard_lib",
        ":types",
        ":util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:pjrt_future",
        "//xla:comparison_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt:transpose",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/platform:fingerprint",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + select({
        ":ifrt_enabled": [
            "@llvm-project//llvm:Support",
            "//xla/python/ifrt",
            "//xla/python/pjrt_ifrt",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "dlpack",
    srcs = ["dlpack.cc"],
    hdrs = ["dlpack.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = select({
        ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],
    deps = [
        ":py_client",
        ":python_ref_manager",
        ":traceback",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//xla:types",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@pybind11",
    ],
)

cc_library(
    name = "jax_jit",
    srcs = ["jax_jit.cc"],
    hdrs = ["jax_jit.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = select({
        ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],
    visibility = [":friends"],  # For the functions to access C++ flags/thread-local variables
    deps = [
        ":exceptions",
        ":py_client",
        ":python_ref_manager",
        ":python_utils",
        ":pytree",
        ":types",
        ":util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
        "//third_party/python_runtime:headers",  # build_cleaner: keep
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:lru_cache",
        "//xla/pjrt:pjrt_client",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/profiler/lib:traceme",
    ] + select({
        ":ifrt_enabled": [
            "//xla/python/ifrt",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "custom_call_sharding",
    srcs = ["custom_call_sharding.cc"],
    hdrs = ["custom_call_sharding.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":status_casters",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service:hlo_sharding_util",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
    ],
)

cc_library(
    name = "ops",
    srcs = ["ops.cc"],
    hdrs = ["ops.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:approx_topk",
        "//xla/client/lib:approx_topk_shape",
        "//xla/client/lib:comparators",
        "//xla/client/lib:lu_decomposition",
        "//xla/client/lib:math",
        "//xla/client/lib:qr",
        "//xla/client/lib:self_adjoint_eig",
        "//xla/client/lib:sorting",
        "//xla/client/lib:svd",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cc_library(
    name = "outfeed_receiver",
    srcs = ["outfeed_receiver.cc"],
    hdrs = ["outfeed_receiver.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:sharding_builder",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "pjit",
    srcs = ["pjit.cc"],
    hdrs = ["pjit.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = select({
        ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":jax_jit",
        ":py_client",
        ":status_casters",
        ":util",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
        "//xla/pjrt:lru_cache",
        "@tsl//tsl/profiler/lib:traceme",
    ] + select({
        ":ifrt_enabled": [
            "//xla/python/ifrt",
            "//xla/python/pjrt_ifrt",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "pmap_lib",
    srcs = ["pmap_lib.cc"],
    hdrs = ["pmap_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = select({
        ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":exceptions",
        ":jax_jit",
        ":py_client",
        ":python_utils",
        ":types",
        ":util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/profiler/lib:traceme",
    ] + select({
        ":ifrt_enabled": [
            "//xla/python/ifrt",
            "//xla/python/pjrt_ifrt",
        ],
        "//conditions:default": [],
    }),
)

xla_cc_test(
    name = "outfeed_receiver_test_cpu",
    size = "small",
    srcs = ["outfeed_receiver_test.cc"],
    deps = [
        ":outfeed_receiver",
        "//xla:statusor",
        "//xla:test",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:xla_builder",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/service:platform_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "outfeed_receiver_py",
    srcs = ["outfeed_receiver_py.cc"],
    hdrs = ["outfeed_receiver_py.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":outfeed_receiver",
        ":py_client",
        ":types",
        "//xla/client:xla_builder",
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
    ],
)

cc_library(
    name = "pytree",
    srcs = ["pytree.cc"],
    hdrs = ["pytree.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        ":exceptions",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":types",
        "//xla:status",
        "//xla/client:xla_computation",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:all_passes",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@pybind11",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    # TODO(b/172353882): figure out why compatible_with is needed to avoid some internal errors.
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":types",
        "//xla:status",
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/cpu:python_tracer",
        "//xla/python/profiler/internal:traceme_wrapper",
        "@pybind11",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/rpc:profiler_server_impl",
        "@tsl//tsl/profiler/rpc/client:capture_profile",
        "@tsl//tsl/profiler/rpc/client:profiler_client",
    ],
)

cc_library(
    name = "transfer_guard_lib",
    srcs = ["transfer_guard_lib.cc"],
    hdrs = ["transfer_guard_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        "//xla:status",
        "//xla:util",
        "@com_google_absl//absl/base:core_headers",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = select({
        ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
        "//xla:status",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
    ] + select({
        ":ifrt_enabled": [
            "//xla/python/ifrt",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "weakref_lru_cache",
    srcs = ["weakref_lru_cache.cc"],
    hdrs = ["weakref_lru_cache.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/pjrt:lru_cache",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
    ],
)

cc_library(
    name = "xla_compiler",
    srcs = ["xla_compiler.cc"],
    hdrs = ["xla_compiler.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":py_client",
        ":types",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:call_inliner",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_target_registry",
        "//xla/service:flatten_call_graph",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_proto_cc",
        "//xla/service:name_uniquer",
        "//xla/service:platform_util",
        "//xla/service:tuple_simplifier",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
        "@tsl//tsl/lib/strings:proto_serialization",
    ],
)

# TODO(phawkins): the configuration settings here are overly confusing. The right fix is to split
# xla_extension.so so that each backend is a separate plugin, however that must wait for a clean
# ABI separation between devices.
config_setting(
    name = "link_gpu_plugin",
    define_values = {"xla_python_enable_gpu": "true"},
)

bool_flag(
    name = "enable_gpu",
    build_setting_default = True,
)

config_setting(
    name = "gpu_enabled",
    flag_values = {
        ":enable_gpu": "True",
    },
)

bool_flag(
    name = "enable_tpu",
    build_setting_default = True,
)

config_setting(
    name = "tpu_enabled",
    flag_values = {
        ":enable_tpu": "True",
    },
)

bool_flag(
    name = "enable_plugin_device",
    build_setting_default = False,
)

config_setting(
    name = "plugin_device_enabled",
    flag_values = {
        ":enable_plugin_device": "True",
    },
)

bool_flag(
    name = "enable_ifrt",
    build_setting_default = False,
)

config_setting(
    name = "ifrt_enabled",
    flag_values = {
        ":enable_ifrt": "True",
    },
)

# If this flag is enabled, it sets RPATH on the xla_extension to values that are suitable for
# finding NVIDIA's CUDA libraries when they are installed as pip packages.
bool_flag(
    name = "jax_cuda_pip_rpaths",
    build_setting_default = False,
)

config_setting(
    name = "use_jax_cuda_pip_rpaths",
    flag_values = {
        ":jax_cuda_pip_rpaths": "True",
    },
)

# We cannot nest select and if_cuda_is_configured so we introduce
# a standalone cc_library target.
cc_library(
    name = "gpu_plugin_deps",
    deps = [
        "//xla/service:gpu_plugin",
    ] + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]),
)

tsl_pybind_extension(
    name = "xla_extension",
    srcs = [
        "xla.cc",
    ],
    defines = select({
        ":gpu_enabled": ["XLA_PYTHON_ENABLE_GPU=1"],
        "//conditions:default": [],
    }) + select({
        ":tpu_enabled": ["XLA_PYTHON_ENABLE_TPU=1"],
        "//conditions:default": [],
    }) + select({
        ":plugin_device_enabled": ["XLA_PYTHON_ENABLE_PLUGIN_DEVICE=1"],
        "//conditions:default": [],
    }) + select({
        ":ifrt_enabled": ["JAX_ENABLE_IFRT=1"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cublas/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cufft/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cudnn/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cusolver/lib",
        ],
        "//conditions:default": [],
    }),
    pytype_deps = [
        "//third_party/py/numpy",
    ],
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        ":dlpack",
        ":custom_call_sharding",
        ":jax_jit",
        ":mlir",
        "@tsl//tsl/python/lib/core:numpy",
        ":ops",
        ":util",
        ":pmap_lib",
        ":pjit",
        ":weakref_lru_cache",
        ":pprof_profile_builder",
        ":profiler",
        ":transfer_guard_lib",
        ":py_client",
        ":pytree",
        ":python_ref_manager",
        ":traceback",
        ":outfeed_receiver_py",
        ":types",
        ":xla_compiler",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:interpreter_device",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:service",
        "//xla/stream_executor/tpu:pjrt_api",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/distributed_runtime/preemption:preemption_sync_manager",
        "@tsl//tsl/python/lib/core:bfloat16_lib",
    ] + select({
        ":gpu_enabled": [
            "//xla/pjrt/gpu:se_gpu_pjrt_client",
        ],
        "//conditions:default": [],
    }) + select({
        ":link_gpu_plugin": [
            ":gpu_plugin_deps",
        ],
        "//conditions:default": [],
    }) + select({
        ":tpu_enabled": [
            "//xla/pjrt:pjrt_c_api_client",
            "//xla/pjrt:tpu_client",
        ],
        "//conditions:default": [],
    }) + select({
        ":plugin_device_enabled": [
            "//xla/pjrt:pjrt_plugin_device_client",
        ],
        "//conditions:default": [],
    }) + select({
        ":ifrt_enabled": [
            "//xla/python/ifrt",
            "//xla/python/pjrt_ifrt",
        ],
        "//conditions:default": [],
    }),
)
