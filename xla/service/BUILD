# Description:
#   XLA service implementation.

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_py_proto_library",
    "xla_py_test_deps",
)
load(
    "@tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@tsl//tsl:tsl.bzl", "if_libtpu")
load("@tsl//tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable", "internal_hlo_deps")
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//xla/service:xla_compile.bzl", "xla_aot_compile_cpu", "xla_aot_compile_gpu")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

tf_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = ["//xla:xla_data_proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "hlo_profile_printer_data",
    srcs = ["hlo_profile_printer_data.proto"],
    cc_api_version = 2,
)

tf_proto_library(
    name = "hlo_execution_profile_data",
    srcs = ["hlo_execution_profile_data.proto"],
    cc_api_version = 2,
    protodeps = [":hlo_profile_printer_data"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "reduce_scatter_utils",
    srcs = ["reduce_scatter_utils.cc"],
    hdrs = ["reduce_scatter_utils.h"],
    deps = [
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "async_collective_creator",
    srcs = ["async_collective_creator.cc"],
    hdrs = ["async_collective_creator.h"],
    deps = [
        ":computation_placer_hdr",
        ":hlo_pass",
        ":shape_inference",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "async_collective_creator_test",
    srcs = ["async_collective_creator_test.cc"],
    deps = [
        ":async_collective_creator",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "async_op_canonicalizer",
    srcs = ["async_op_canonicalizer.cc"],
    hdrs = ["async_op_canonicalizer.h"],
    deps = [
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "async_op_canonicalizer_test",
    srcs = ["async_op_canonicalizer_test.cc"],
    deps = [
        ":async_op_canonicalizer",
        ":hlo_dce",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "all_reduce_key",
    srcs = ["all_reduce_key.cc"],
    hdrs = ["all_reduce_key.h"],
    deps = [
        ":hlo_domain_map",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "all_reduce_promotion",
    srcs = ["all_reduce_promotion.cc"],
    hdrs = ["all_reduce_promotion.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:change_op_data_type",
        "//xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "all_reduce_promotion_test",
    srcs = ["all_reduce_promotion_test.cc"],
    deps = [
        ":all_reduce_promotion",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "all_reduce_reassociate",
    srcs = ["all_reduce_reassociate.cc"],
    hdrs = ["all_reduce_reassociate.h"],
    deps = [
        ":all_reduce_key",
        ":collective_ops_utils",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "all_reduce_reassociate_test",
    srcs = ["all_reduce_reassociate_test.cc"],
    deps = [
        ":all_reduce_reassociate",
        ":hlo_matchers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "all_reduce_folder",
    srcs = ["all_reduce_folder.cc"],
    hdrs = ["all_reduce_folder.h"],
    deps = [
        ":all_reduce_key",
        ":hlo_pass",
        ":hlo_query",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "all_reduce_folder_test",
    srcs = ["all_reduce_folder_test.cc"],
    deps = [
        ":all_reduce_folder",
        ":hlo_matchers",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "bfloat16_support",
    srcs = ["bfloat16_support.cc"],
    hdrs = ["bfloat16_support.h"],
    deps = [
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "broadcast_canonicalizer",
    srcs = ["broadcast_canonicalizer.cc"],
    hdrs = ["broadcast_canonicalizer.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
    ],
)

xla_cc_test(
    name = "broadcast_canonicalizer_test",
    srcs = ["broadcast_canonicalizer_test.cc"],
    deps = [
        ":broadcast_canonicalizer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "bfloat16_conversion_folding",
    srcs = ["bfloat16_conversion_folding.cc"],
    hdrs = ["bfloat16_conversion_folding.h"],
    deps = [
        ":bfloat16_support",
        ":hlo_dataflow_analysis",
        ":hlo_pass",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "bfloat16_conversion_folding_test",
    srcs = ["bfloat16_conversion_folding_test.cc"],
    deps = [
        ":bfloat16_conversion_folding",
        ":bfloat16_support",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "bfloat16_normalization",
    srcs = ["bfloat16_normalization.cc"],
    hdrs = ["bfloat16_normalization.h"],
    deps = [
        ":bfloat16_support",
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_simplifier",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "bfloat16_normalization_test",
    srcs = ["bfloat16_normalization_test.cc"],
    deps = [
        ":bfloat16_normalization",
        ":bfloat16_support",
        ":hlo_creation_utils",
        ":hlo_verifier",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "bfloat16_propagation",
    srcs = ["bfloat16_propagation.cc"],
    hdrs = ["bfloat16_propagation.h"],
    deps = [
        ":bfloat16_support",
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_simplifier",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "bfloat16_propagation_test",
    srcs = ["bfloat16_propagation_test.cc"],
    deps = [
        ":bfloat16_propagation",
        ":bfloat16_support",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    deps = [
        ":hlo_graph_dumper",
        ":hlo_proto_util",
        "//xla:status",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tsl//tsl/lib/io:zlib_compression_options",
        "@tsl//tsl/lib/io:zlib_outputbuffer",
        "@tsl//tsl/lib/strings:proto_serialization",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:padding",
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "hlo_opcode_test",
    srcs = ["hlo_opcode_test.cc"],
    deps = [
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_live_range",
    srcs = [
        "hlo_live_range.cc",
    ],
    hdrs = [
        "hlo_live_range.h",
    ],
    deps = [
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_live_range_test",
    srcs = ["hlo_live_range_test.cc"],
    deps = [
        ":call_graph",
        ":hlo_alias_analysis",
        ":hlo_live_range",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_parser",
        ":hlo_value",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_sharding_util",
    srcs = [
        "hlo_sharding_util.cc",
    ],
    hdrs = [
        "hlo_sharding_util.h",
    ],
    deps = [
        ":call_graph",
        "//xla:array",
        "//xla:literal_util",
        "//xla:protobuf_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_sharding_util_test",
    srcs = [
        "hlo_sharding_util_test.cc",
    ],
    deps = [
        ":hlo_sharding_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "sharding_propagation",
    srcs = [
        "sharding_propagation.cc",
    ],
    hdrs = [
        "sharding_propagation.h",
    ],
    deps = [
        ":call_graph",
        ":custom_call_sharding_helper",
        ":dot_as_convolution_util",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_sharding_util",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:sharding_op_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "sharding_propagation_test",
    srcs = [
        "sharding_propagation_test.cc",
    ],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":sharding_propagation",
        "//xla:protobuf_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sharding_remover",
    srcs = [
        "sharding_remover.cc",
    ],
    hdrs = [
        "sharding_remover.h",
    ],
    deps = [
        ":hlo_pass",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "sharding_remover_test",
    size = "small",
    srcs = [
        "sharding_remover_test.cc",
    ],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":sharding_remover",
        "//xla:status_macros",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "dot_as_convolution_util",
    srcs = [
        "dot_as_convolution_util.cc",
    ],
    hdrs = [
        "dot_as_convolution_util.h",
    ],
    deps = [
        ":shape_inference",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "dynamic_parameter_binding_test",
    srcs = ["dynamic_parameter_binding_test.cc"],
    deps = [
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "dynamic_update_slice_test",
    srcs = ["dynamic_update_slice_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":hlo_parser",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/cpu:parallel_task_assignment",
        "//xla/service/cpu:target_machine_features",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "dfs_hlo_visitor_with_default_test",
    srcs = ["dfs_hlo_visitor_with_default_test.cc"],
    deps = [
        ":hlo_runner",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "pattern_matcher",
    hdrs = ["pattern_matcher.h"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
    ],
)

xla_cc_test(
    name = "pattern_matcher_test",
    srcs = ["pattern_matcher_test.cc"],
    deps = [
        ":pattern_matcher",
        "//xla:shape_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "pattern_matcher_gmock",
    testonly = 1,
    hdrs = ["pattern_matcher_gmock.h"],
    deps = [
        ":pattern_matcher",
        "//xla:test",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "pattern_matcher_gmock_test",
    srcs = ["pattern_matcher_gmock_test.cc"],
    deps = [
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:shape_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_reachability",
    srcs = ["hlo_reachability.cc"],
    hdrs = ["hlo_reachability.h"],
    deps = [
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_reachability_test",
    srcs = ["hlo_reachability_test.cc"],
    deps = [
        ":computation_placer",
        ":hlo_reachability",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_matchers",
    testonly = 1,
    srcs = ["hlo_matchers.cc"],
    hdrs = ["hlo_matchers.h"],
    deps = [
        ":hlo_parser",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_matchers_test",
    srcs = ["hlo_matchers_test.cc"],
    deps = [
        ":hlo_matchers",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "hlo_instruction_test",
    srcs = ["hlo_instruction_test.cc"],
    deps = [
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_sharding_test",
    srcs = ["hlo_sharding_test.cc"],
    deps = [
        ":hlo_parser",
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "call_graph",
    srcs = ["call_graph.cc"],
    hdrs = ["call_graph.h"],
    deps = [
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "call_graph_test",
    srcs = ["call_graph_test.cc"],
    deps = [
        ":call_graph",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "flatten_call_graph",
    srcs = ["flatten_call_graph.cc"],
    hdrs = ["flatten_call_graph.h"],
    deps = [
        ":call_graph",
        ":hlo_pass",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "call_inliner",
    srcs = ["call_inliner.cc"],
    hdrs = ["call_inliner.h"],
    deps = [
        ":call_graph",
        ":hlo_dce",
        ":hlo_domain_isolator",
        ":hlo_pass",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "call_inliner_test",
    size = "small",
    srcs = ["call_inliner_test.cc"],
    deps = [
        ":call_inliner",
        ":hlo_matchers",
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_computation_deduplicator",
    srcs = ["hlo_computation_deduplicator.cc"],
    hdrs = ["hlo_computation_deduplicator.h"],
    deps = [
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "hlo_computation_deduplicator_test",
    size = "small",
    srcs = ["hlo_computation_deduplicator_test.cc"],
    deps = [
        ":hlo_computation_deduplicator",
        ":hlo_matchers",
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "flatten_call_graph_test",
    srcs = ["flatten_call_graph_test.cc"],
    deps = [
        ":call_graph",
        ":flatten_call_graph",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "platform_util",
    srcs = ["platform_util.cc"],
    hdrs = ["platform_util.h"],
    deps = [
        ":compiler",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":compiler",
        ":computation_placer",
        ":platform_util",
        ":stream_pool",
        ":transfer_manager",
        "//third_party/eigen3",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor/host:host_platform_id",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":allocation_tracker",
        ":backend",
        ":channel_tracker",
        ":compilation_cache",
        ":compiler",
        ":computation_layout",
        ":computation_placer",
        ":dump",
        ":dynamic_dimension_inference",
        ":dynamic_padder",
        ":executable",
        ":execution_tracker",
        ":hlo_cost_analysis",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_util",
        ":hlo_proto_util",
        ":platform_util",
        ":source_map_util",
        ":stream_pool",
        ":transfer_manager",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:execution_options_util",
        "//xla:service_interface",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/util:ptr_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "local_service",
    srcs = ["local_service.cc"],
    hdrs = ["local_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":executable",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_util",
        ":platform_util",
        ":service",
        ":shaped_buffer",
        "//xla:execution_options_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "latency_hiding_scheduler",
    srcs = ["latency_hiding_scheduler.cc"],
    hdrs = ["latency_hiding_scheduler.h"],
    deps = [
        ":dump",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_cost_analysis",
        ":hlo_dataflow_analysis",
        ":hlo_pass",
        ":hlo_reachability",
        ":tuple_points_to_analysis",
        "//xla:status",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "latency_hiding_scheduler_test",
    srcs = ["latency_hiding_scheduler_test.cc"],
    deps = [
        ":async_collective_creator",
        ":latency_hiding_scheduler",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "compile_only_service",
    srcs = ["compile_only_service.cc"],
    hdrs = ["compile_only_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":dump",
        ":platform_util",
        ":service",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "cpu_plugin",
    compatible_with = [],
    deps = [
        ":service",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_platform",
    ],
)

cc_library(
    name = "gpu_plugin_impl",
    compatible_with = [],
    deps = if_gpu_is_configured([
        ":service",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_transfer_manager",
        "//xla/stream_executor",
    ]) + if_cuda_is_configured([
        "//xla/service/gpu:nvptx_compiler",
        "//xla/stream_executor/cuda:stream_executor_cuda",
    ]) + if_rocm_is_configured([
        "//xla/service/gpu:amdgpu_compiler",
        "//xla/stream_executor/rocm:stream_executor_rocm",
    ]),
)

cc_library(
    name = "gpu_plugin_stub",
)

alias(
    name = "gpu_plugin_noncuda",
    actual = if_libtpu("gpu_plugin_stub", "gpu_plugin_impl"),
)

alias(
    name = "gpu_plugin",
    actual = if_cuda("gpu_plugin_impl", "gpu_plugin_noncuda"),
)

cc_library(
    name = "interpreter_plugin",
    deps = [
        ":service",
        "//xla/backends/interpreter:compiler",
        "//xla/backends/interpreter:interpreter_transfer_manager",
        "//xla/backends/interpreter:platform",
        "//xla/stream_executor",
    ],
)

cc_library(
    name = "shaped_buffer",
    srcs = ["shaped_buffer.cc"],
    hdrs = ["shaped_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "shaped_buffer_test",
    srcs = ["shaped_buffer_test.cc"],
    deps = [
        ":cpu_plugin",
        ":platform_util",
        ":shaped_buffer",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
        "@tsl//tsl/util:ptr_util",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = [
        "executable.h",
        "service_executable_run_options.h",
    ],
    deps = [
        ":computation_layout",
        ":dump",
        "//xla/hlo/ir:hlo",
        ":hlo_execution_profile",
        ":hlo_graph_dumper",
        ":hlo_proto_cc",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        ":stream_pool",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:shape_tree",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory_allocator",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/lib/strings:proto_serialization",
    ] + internal_hlo_deps(),
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":buffer_assignment",
        ":buffer_value",
        ":computation_placer",
        ":executable",
        ":hlo_module_config",
        ":logical_buffer",
        ":metrics_hook_interface",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/stream_executor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "llvm_compiler",
    srcs = ["llvm_compiler.cc"],
    hdrs = ["llvm_compiler.h"],
    deps = [
        ":compiler",
        "@llvm-project//llvm:Core",
        "@tsl//tsl/platform:denormal",
    ],
)

cc_library(
    name = "transfer_manager",
    srcs = ["transfer_manager.cc"],
    hdrs = ["transfer_manager.h"],
    deps = [
        ":compiler",
        ":executable",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:notification",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "allocation_tracker",
    srcs = ["allocation_tracker.cc"],
    hdrs = ["allocation_tracker.h"],
    deps = [
        ":backend",
        ":transfer_manager",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "execution_tracker",
    srcs = ["execution_tracker.cc"],
    hdrs = ["execution_tracker.h"],
    deps = [
        ":backend",
        ":stream_pool",
        "//xla:executable_run_options",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "channel_tracker",
    srcs = ["channel_tracker.cc"],
    hdrs = ["channel_tracker.h"],
    deps = [
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "name_uniquer",
    srcs = ["name_uniquer.cc"],
    hdrs = ["name_uniquer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "name_uniquer_test",
    srcs = ["name_uniquer_test.cc"],
    deps = [
        ":name_uniquer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "buffer_assignment",
    srcs = [
        "buffer_assignment.cc",
    ],
    hdrs = [
        "buffer_assignment.h",
    ],
    deps = [
        ":buffer_value_containers",
        ":heap_simulator",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_live_range",
        ":hlo_proto_cc",
        ":hlo_value",
        ":logical_buffer",
        ":memory_space_assignment",
        ":tuple_points_to_analysis",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "buffer_assignment_test",
    srcs = ["buffer_assignment_test.cc"],
    deps = [
        ":async_op_canonicalizer",
        ":buffer_assignment",
        ":buffer_value",
        ":call_graph",
        ":copy_insertion",
        ":cpu_plugin",
        ":flatten_call_graph",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_parser",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_ordering",
    srcs = ["hlo_ordering.cc"],
    hdrs = ["hlo_ordering.h"],
    deps = [
        ":call_graph",
        ":hlo_dataflow_analysis",
        ":hlo_proto_cc",
        ":hlo_reachability",
        ":hlo_value",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_ordering_test",
    size = "small",
    srcs = ["hlo_ordering_test.cc"],
    deps = [
        ":hlo_dataflow_analysis",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "heap_simulator",
    srcs = ["heap_simulator.cc"],
    hdrs = ["heap_simulator.h"],
    deps = [
        ":buffer_value",
        ":buffer_value_containers",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_live_range",
        ":hlo_ordering",
        ":hlo_proto_cc",
        ":memory_space_assignment_repacking",
        ":tuple_points_to_analysis",
        "//xla:comparison_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "heap_simulator_test",
    srcs = ["heap_simulator_test.cc"],
    deps = [
        ":buffer_value",
        ":heap_simulator",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        ":tuple_points_to_analysis",
        "//xla:literal",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_module_group_test",
    srcs = ["hlo_module_group_test.cc"],
    # TODO(b/148211710) Test fails in OSS.
    tags = ["no_oss"],
    deps = [
        ":hlo_matchers",
        ":hlo_module_group_metadata",
        ":hlo_proto_cc",
        "//xla:test",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_module_group_metadata",
    srcs = ["hlo_module_group_metadata.cc"],
    hdrs = ["hlo_module_group_metadata.h"],
    deps = [
        ":hlo_alias_analysis",
        ":tuple_points_to_analysis",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hlo_module_util",
    srcs = ["hlo_module_util.cc"],
    hdrs = ["hlo_module_util.h"],
    deps = [
        ":compiler",
        ":hlo_module_config",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_module_group_util",
    srcs = ["hlo_module_group_util.cc"],
    hdrs = ["hlo_module_group_util.h"],
    deps = [
        ":hlo_module_group_metadata",
        ":hlo_reachability",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_schedule_test",
    srcs = ["hlo_schedule_test.cc"],
    deps = [
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_input_output_alias_config_test",
    srcs = ["hlo_input_output_alias_config_test.cc"],
    deps = [
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_memory_scheduler",
    srcs = ["hlo_memory_scheduler.cc"],
    hdrs = ["hlo_memory_scheduler.h"],
    deps = [
        ":heap_simulator",
        ":hlo_alias_analysis",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/lib/gtl:map_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_memory_scheduler_test",
    srcs = ["hlo_memory_scheduler_test.cc"],
    deps = [
        ":heap_simulator",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_parser",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_query",
    srcs = ["hlo_query.cc"],
    hdrs = ["hlo_query.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "fusion_queue",
    hdrs = ["fusion_queue.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":fusion_queue",
        ":hlo_dataflow_analysis",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_reachability",
        ":pattern_matcher",
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":instruction_fusion",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_pass",
        ":hlo_reachability",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hlo_creation_utils",
    srcs = ["hlo_creation_utils.cc"],
    hdrs = [
        "hlo_creation_utils.h",
    ],
    deps = [
        ":hlo_module_config",
        ":shape_inference",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:comparators",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fusion_node_indexing_evaluation",
    srcs = ["fusion_node_indexing_evaluation.cc"],
    hdrs = ["fusion_node_indexing_evaluation.h"],
    deps = [
        ":elemental_ir_emitter",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "fusion_node_indexing_evaluation_test",
    srcs = ["fusion_node_indexing_evaluation_test.cc"],
    deps = [
        ":fusion_node_indexing_evaluation",
        ":hlo_parser",
        ":instruction_fusion",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_creation_utils_test",
    srcs = ["hlo_creation_utils_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "batchnorm_expander",
    srcs = ["batchnorm_expander.cc"],
    hdrs = ["batchnorm_expander.h"],
    deps = [
        ":hlo_pass",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "op_expander_pass",
    srcs = ["op_expander_pass.cc"],
    hdrs = ["op_expander_pass.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "gather_expander",
    srcs = ["gather_expander.cc"],
    hdrs = ["gather_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        ":while_util",
        "//xla:literal_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "optimization_barrier_expander",
    srcs = ["optimization_barrier_expander.cc"],
    hdrs = ["optimization_barrier_expander.h"],
    deps = [
        ":op_expander_pass",
    ],
)

cc_library(
    name = "comparison_expander",
    srcs = ["comparison_expander.cc"],
    hdrs = ["comparison_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        "//xla:statusor",
        "//xla:util",
        "//xla/client/lib:comparators",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "scatter_expander",
    srcs = ["scatter_expander.cc"],
    hdrs = ["scatter_expander.h"],
    deps = [
        ":call_inliner",
        ":hlo_creation_utils",
        ":op_expander_pass",
        ":while_util",
        "//xla:literal_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "scatter_expander_test",
    srcs = ["scatter_expander_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":scatter_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "triangular_solve_expander",
    srcs = ["triangular_solve_expander.cc"],
    hdrs = ["triangular_solve_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:constants",
        "//xla/client/lib:math",
        "//xla/client/lib:matrix",
        "//xla/client/lib:slicing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "triangular_solve_expander_test",
    size = "medium",
    srcs = ["triangular_solve_expander_test.cc"],
    shard_count = 12,
    deps = [
        ":triangular_solve_expander",
        "//xla:literal",
        "//xla:reference_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cholesky_expander",
    srcs = ["cholesky_expander.cc"],
    hdrs = ["cholesky_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/client/lib:constants",
        "//xla/client/lib:loops",
        "//xla/client/lib:math",
        "//xla/client/lib:matrix",
        "//xla/client/lib:slicing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "qr_expander",
    srcs = ["qr_expander.cc"],
    hdrs = ["qr_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/client/lib:constants",
        "//xla/client/lib:loops",
        "//xla/client/lib:math",
        "//xla/client/lib:matrix",
        "//xla/client/lib:qr",
        "//xla/client/lib:slicing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "real_imag_expander",
    srcs = ["real_imag_expander.cc"],
    hdrs = ["real_imag_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
    ],
)

xla_cc_test(
    name = "real_imag_expander_test",
    size = "small",
    srcs = ["real_imag_expander_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":real_imag_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "eigh_expander",
    srcs = ["eigh_expander.cc"],
    hdrs = ["eigh_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/client/lib:comparators",
        "//xla/client/lib:constants",
        "//xla/client/lib:loops",
        "//xla/client/lib:math",
        "//xla/client/lib:matrix",
        "//xla/client/lib:slicing",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "convolution_4d_expander",
    srcs = ["convolution_4d_expander.cc"],
    hdrs = ["convolution_4d_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "convolution_4d_expander_test",
    srcs = ["convolution_4d_expander_test.cc"],
    deps = [
        "convolution_4d_expander",
        ":hlo_matchers",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "convolution_pred_expander",
    srcs = ["convolution_pred_expander.cc"],
    hdrs = ["convolution_pred_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":op_expander_pass",
        ":pattern_matcher",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "convolution_pred_expander_test",
    srcs = ["convolution_pred_expander_test.cc"],
    deps = [
        ":convolution_pred_expander",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "batchnorm_expander_test",
    size = "small",
    srcs = ["batchnorm_expander_test.cc"],
    deps = [
        ":batchnorm_expander",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "algebraic_simplifier",
    srcs = ["algebraic_simplifier.cc"],
    hdrs = ["algebraic_simplifier.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        ":hlo_query",
        ":pattern_matcher",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tree_reduction_rewriter",
    srcs = ["tree_reduction_rewriter.cc"],
    hdrs = ["tree_reduction_rewriter.h"],
    deps = [
        ":hlo_pass",
        ":shape_inference",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:padding",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "algebraic_simplifier_test",
    srcs = ["algebraic_simplifier_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo_creation_utils",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":shape_inference",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "algebraic_simplifier_overflow_test",
    srcs = ["algebraic_simplifier_overflow_test.cc"],
    deps = [
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "simplify_fp_conversions",
    srcs = ["simplify_fp_conversions.cc"],
    hdrs = ["simplify_fp_conversions.h"],
    deps = [
        ":hlo_pass",
        "//xla:comparison_util",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "simplify_fp_conversions_test",
    srcs = ["simplify_fp_conversions_test.cc"],
    deps = [
        ":hlo_matchers",
        ":simplify_fp_conversions",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "logistic_expander",
    srcs = ["logistic_expander.cc"],
    hdrs = ["logistic_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        ":op_expander_pass",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "logistic_expander_test",
    srcs = ["logistic_expander_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":logistic_expander",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":shape_inference",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "collectives_schedule_linearizer",
    srcs = ["collectives_schedule_linearizer.cc"],
    hdrs = ["collectives_schedule_linearizer.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "collectives_schedule_linearizer_test",
    srcs = ["collectives_schedule_linearizer_test.cc"],
    deps = [
        ":collectives_schedule_linearizer",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_runner",
        ":pattern_matcher",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "collective_combiner_utils",
    hdrs = ["collective_combiner_utils.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_reachability",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "collective_decomposer_utils",
    srcs = ["collective_decomposer_utils.cc"],
    hdrs = ["collective_decomposer_utils.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_module_config",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "all_gather_broadcast_reorder",
    srcs = ["all_gather_broadcast_reorder.cc"],
    hdrs = ["all_gather_broadcast_reorder.h"],
    deps = [
        ":hlo_pass",
        ":hlo_query",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "bitcast_dtypes_expander",
    srcs = ["bitcast_dtypes_expander.cc"],
    hdrs = ["bitcast_dtypes_expander.h"],
    deps = [
        ":hlo_pass",
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/client:xla_builder",
        "//xla/client/lib:arithmetic",
        "//xla/client/lib:broadcast",
        "//xla/client/lib:constants",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "bitcast_dtypes_expander_test",
    srcs = ["bitcast_dtypes_expander_test.cc"],
    deps = [
        ":bitcast_dtypes_expander",
        ":hlo_matchers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "all_gather_broadcast_reorder_test",
    srcs = ["all_gather_broadcast_reorder_test.cc"],
    deps = [
        ":all_gather_broadcast_reorder",
        ":hlo_matchers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "all_gather_combiner",
    srcs = ["all_gather_combiner.cc"],
    hdrs = ["all_gather_combiner.h"],
    deps = [
        ":collective_combiner_utils",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":hlo_sharding_util",
        ":shape_inference",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "all_gather_combiner_test",
    srcs = ["all_gather_combiner_test.cc"],
    deps = [
        ":all_gather_combiner",
        ":hlo_matchers",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "all_reduce_combiner",
    srcs = ["all_reduce_combiner.cc"],
    hdrs = ["all_reduce_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":hlo_sharding_util",
        ":shape_inference",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "all_reduce_combiner_test",
    srcs = ["all_reduce_combiner_test.cc"],
    deps = [
        ":all_reduce_combiner",
        ":hlo_matchers",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "all_reduce_contiguous",
    srcs = ["all_reduce_contiguous.cc"],
    hdrs = ["all_reduce_contiguous.h"],
    deps = [
        ":hlo_pass",
        ":hlo_query",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "all_reduce_contiguous_test",
    srcs = ["all_reduce_contiguous_test.cc"],
    deps = [
        ":all_reduce_contiguous",
        ":hlo_matchers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "reduce_scatter_combiner",
    srcs = ["reduce_scatter_combiner.cc"],
    hdrs = ["reduce_scatter_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":collective_ops_utils",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_reachability",
        ":shape_inference",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "reduce_scatter_combiner_test",
    srcs = ["reduce_scatter_combiner_test.cc"],
    deps = [
        ":hlo_matchers",
        ":reduce_scatter_combiner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "all_reduce_simplifier",
    srcs = ["all_reduce_simplifier.cc"],
    hdrs = ["all_reduce_simplifier.h"],
    deps = [
        ":hlo_pass",
        ":hlo_replication_analysis",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "all_reduce_simplifier_test",
    srcs = ["all_reduce_simplifier_test.cc"],
    deps = [
        ":all_reduce_simplifier",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "reduce_scatter_decomposer",
    srcs = ["reduce_scatter_decomposer.cc"],
    hdrs = ["reduce_scatter_decomposer.h"],
    deps = [
        ":collective_decomposer_utils",
        ":collective_ops_utils",
        ":hlo_creation_utils",
        ":hlo_module_config",
        ":hlo_pass",
        ":hlo_query",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "reduce_scatter_decomposer_test",
    srcs = ["reduce_scatter_decomposer_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":hlo_matchers",
        ":hlo_parser",
        ":reduce_scatter_decomposer",
        "//xla:literal_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "reduce_scatter_reassociate",
    srcs = ["reduce_scatter_reassociate.cc"],
    hdrs = ["reduce_scatter_reassociate.h"],
    deps = [
        ":all_reduce_key",
        ":collective_ops_utils",
        ":hlo_domain_map",
        ":hlo_pass",
        ":hlo_query",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "reduce_scatter_reassociate_test",
    srcs = ["reduce_scatter_reassociate_test.cc"],
    deps = [
        ":hlo_matchers",
        ":reduce_scatter_reassociate",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "batch_dot_simplification",
    srcs = ["batch_dot_simplification.cc"],
    hdrs = ["batch_dot_simplification.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "batch_dot_simplification_test",
    srcs = ["batch_dot_simplification_test.cc"],
    deps = [
        ":batch_dot_simplification",
        ":hlo_matchers",
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "gather_expander_test",
    srcs = ["gather_expander_test.cc"],
    deps = [
        ":gather_expander",
        ":hlo_query",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "conditional_simplifier",
    srcs = ["conditional_simplifier.cc"],
    hdrs = ["conditional_simplifier.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "conditional_simplifier_test",
    srcs = ["conditional_simplifier_test.cc"],
    deps = [
        ":conditional_simplifier",
        ":hlo_matchers",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "conditional_code_motion",
    srcs = ["conditional_code_motion.cc"],
    hdrs = ["conditional_code_motion.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo_cse",
        ":hlo_dce",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":hlo_verifier",
        ":tuple_simplifier",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "conditional_code_motion_test",
    srcs = ["conditional_code_motion_test.cc"],
    deps = [
        ":conditional_code_motion",
        ":hlo_matchers",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "convolution_group_converter",
    srcs = ["convolution_group_converter.cc"],
    hdrs = ["convolution_group_converter.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "convolution_group_converter_test",
    size = "small",
    srcs = ["convolution_group_converter_test.cc"],
    deps = [
        ":convolution_group_converter",
        ":hlo_matchers",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "space_to_batch_converter",
    srcs = ["space_to_batch_converter.cc"],
    hdrs = ["space_to_batch_converter.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        ":pattern_matcher",
        ":shape_inference",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/core:bitmap",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "space_to_batch_converter_test",
    size = "small",
    srcs = ["space_to_batch_converter_test.cc"],
    deps = [
        ":hlo_matchers",
        ":space_to_batch_converter",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "while_loop_analysis",
    srcs = ["while_loop_analysis.cc"],
    hdrs = ["while_loop_analysis.h"],
    deps = [
        ":hlo_reachability",
        ":pattern_matcher",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base",
    ],
)

xla_cc_test(
    name = "while_loop_analysis_test",
    srcs = ["while_loop_analysis_test.cc"],
    deps = [
        ":hlo_parser",
        ":while_loop_analysis",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "while_loop_simplifier",
    srcs = ["while_loop_simplifier.cc"],
    hdrs = ["while_loop_simplifier.h"],
    deps = [
        ":call_inliner",
        ":hlo_pass",
        ":hlo_query",
        ":pattern_matcher",
        ":while_loop_analysis",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:union_find",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "while_loop_simplifier_test",
    srcs = ["while_loop_simplifier_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo_cse",
        ":hlo_dce",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":tuple_simplifier",
        ":while_loop_simplifier",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "while_loop_trip_count_annotator",
    srcs = ["while_loop_trip_count_annotator.cc"],
    hdrs = ["while_loop_trip_count_annotator.h"],
    deps = [
        ":hlo_pass",
        ":while_loop_analysis",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "while_loop_trip_count_annotator_test",
    srcs = ["while_loop_trip_count_annotator_test.cc"],
    deps = [
        ":pattern_matcher",
        ":while_loop_simplifier",
        ":while_loop_trip_count_annotator",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "defuser",
    srcs = ["defuser.cc"],
    hdrs = ["defuser.h"],
    deps = [
        ":call_graph",
        ":hlo_pass",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "defuser_test",
    srcs = ["defuser_test.cc"],
    deps = [
        ":defuser",
        ":hlo_matchers",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "despecializer_test",
    srcs = ["despecializer_test.cc"],
    deps = [
        ":despecializer",
        ":hlo_matchers",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["dot_decomposer.cc"],
    hdrs = ["dot_decomposer.h"],
    deps = [
        ":hlo_pass",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "dot_decomposer_test",
    srcs = ["dot_decomposer_test.cc"],
    deps = [
        ":dot_decomposer",
        ":hlo_matchers",
        ":hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "dot_merger",
    srcs = ["dot_merger.cc"],
    hdrs = ["dot_merger.h"],
    deps = [
        ":hlo_pass",
        ":shape_inference",
        "//xla/hlo/ir:hlo",
        "//xla/service/graphcycles",
    ],
)

xla_cc_test(
    name = "dot_merger_test",
    srcs = ["dot_merger_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":dot_merger",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "convert_mover",
    srcs = ["convert_mover.cc"],
    hdrs = ["convert_mover.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
        "//xla/service/graphcycles",
    ],
)

xla_cc_test(
    name = "convert_mover_test",
    srcs = ["convert_mover_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":convert_mover",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "all_to_all_decomposer",
    srcs = ["all_to_all_decomposer.cc"],
    hdrs = ["all_to_all_decomposer.h"],
    deps = [
        ":hlo_pass",
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "all_gather_decomposer",
    srcs = ["all_gather_decomposer.cc"],
    hdrs = ["all_gather_decomposer.h"],
    deps = [
        ":collective_decomposer_utils",
        ":collective_ops_utils",
        ":hlo_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "all_gather_decomposer_test",
    srcs = ["all_gather_decomposer_test.cc"],
    deps = [
        ":all_gather_decomposer",
        ":hlo_matchers",
        ":hlo_parser",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "tuple_simplifier",
    srcs = ["tuple_simplifier.cc"],
    hdrs = ["tuple_simplifier.h"],
    deps = [
        ":hlo_pass",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "tuple_simplifier_test",
    srcs = ["tuple_simplifier_test.cc"],
    deps = [
        ":hlo_matchers",
        ":tuple_simplifier",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "reshape_mover",
    srcs = ["reshape_mover.cc"],
    hdrs = ["reshape_mover.h"],
    deps = [
        ":hlo_pass",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "reshape_decomposer",
    srcs = ["reshape_decomposer.cc"],
    hdrs = ["reshape_decomposer.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla:status",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "reduce_decomposer",
    srcs = ["reduce_decomposer.cc"],
    hdrs = ["reduce_decomposer.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla:status",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "reduce_decomposer_test",
    srcs = ["reduce_decomposer_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":hlo_parser",
        ":reduce_decomposer",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "reshape_decomposer_test",
    srcs = ["reshape_decomposer_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":hlo_parser",
        ":reshape_decomposer",
        ":reshape_mover",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "dynamic_window_utils",
    srcs = ["dynamic_window_utils.cc"],
    hdrs = ["dynamic_window_utils.h"],
    deps = [
        ":shape_inference",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "dynamic_dimension_inference",
    srcs = ["dynamic_dimension_inference.cc"],
    hdrs = ["dynamic_dimension_inference.h"],
    deps = [
        ":dynamic_window_utils",
        ":hlo_creation_utils",
        ":tuple_util",
        ":while_util",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "dynamic_dimension_simplifier",
    srcs = ["dynamic_dimension_simplifier.cc"],
    hdrs = ["dynamic_dimension_simplifier.h"],
    deps = [
        ":hlo_pass",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "dynamic_dimension_simplifier_test",
    srcs = ["dynamic_dimension_simplifier_test.cc"],
    deps = [
        ":dynamic_dimension_simplifier",
        ":hlo_creation_utils",
        ":hlo_parser",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":shape_inference",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "dynamic_padder",
    srcs = ["dynamic_padder.cc"],
    hdrs = ["dynamic_padder.h"],
    deps = [
        ":dynamic_dimension_inference",
        ":dynamic_window_utils",
        ":hlo_creation_utils",
        ":hlo_dce",
        ":hlo_pass",
        ":pattern_matcher",
        ":shape_inference",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/lib/monitoring:gauge",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "dynamic_padder_test",
    srcs = ["dynamic_padder_test.cc"],
    deps = [
        ":dynamic_dimension_inference",
        ":dynamic_dimension_simplifier",
        ":dynamic_padder",
        ":hlo_dce",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":tuple_simplifier",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
        "@tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

xla_cc_test(
    name = "dynamic_dimension_inference_test",
    srcs = ["dynamic_dimension_inference_test.cc"],
    deps = [
        ":dynamic_dimension_inference",
        ":hlo_matchers",
        ":hlo_runner",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "reshape_mover_test",
    srcs = ["reshape_mover_test.cc"],
    deps = [
        ":hlo_matchers",
        ":reshape_mover",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "computation_placer",
    srcs = ["computation_placer.cc"],
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
    ] + if_libtpu([":tpu_computation_placer"]),
    alwayslink = True,  # Contains per-platform computation placer registration
)

cc_library(
    name = "computation_placer_hdr",
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "//xla:array2d",
        "//xla:status",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tpu_computation_placer",
    srcs = ["tpu_computation_placer.cc"],
    hdrs = ["tpu_computation_placer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":computation_placer_hdr",
        "//xla:statusor",
        "//xla/stream_executor/tpu:status_helper",
        "//xla/stream_executor/tpu:tpu_api",
        "//xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//xla/stream_executor/tpu:tpu_platform_hdr",
        "//xla/stream_executor/tpu:tpu_platform_id",
        "//xla/stream_executor/tpu:tpu_topology_external",
    ],
    alwayslink = True,  # Contains TPU computation placer registration
)

cc_library(
    name = "human_readable_profile_builder",
    srcs = ["human_readable_profile_builder.cc"],
    hdrs = ["human_readable_profile_builder.h"],
    deps = [
        "//xla:metric_table_report",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:numbers",
    ],
)

cc_library(
    name = "generic_transfer_manager",
    srcs = ["generic_transfer_manager.cc"],
    hdrs = ["generic_transfer_manager.h"],
    deps = [
        ":transfer_manager",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "hlo_cost_analysis",
    srcs = ["hlo_cost_analysis.cc"],
    hdrs = ["hlo_cost_analysis.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/lib/gtl:map_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_cost_analysis_test",
    srcs = ["hlo_cost_analysis_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo_cost_analysis",
        ":hlo_parser",
        ":local_service",
        ":service",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/client:padding",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_execution_profile",
    srcs = ["hlo_execution_profile.cc"],
    hdrs = ["hlo_execution_profile.h"],
    deps = [
        ":hlo_cost_analysis",
        ":hlo_execution_profile_data_cc",
        ":hlo_profile_printer",
        ":human_readable_profile_builder",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "hlo_execution_profile_test",
    srcs = ["hlo_execution_profile_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo_cost_analysis",
        ":hlo_execution_profile",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_computation_test",
    srcs = ["hlo_computation_test.cc"],
    deps = [
        ":hlo_matchers",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_test",
    srcs = ["hlo_module_test.cc"],
    deps = [
        ":computation_placer_hdr",
        ":hlo_matchers",
        ":hlo_memory_scheduler",
        ":hlo_module_config",
        ":test_compilation_environment_proto_cc",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/lib/strings:proto_serialization",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_metadata_test",
    srcs = ["hlo_module_metadata_test.cc"],
    deps = [
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "buffer_value",
    srcs = ["buffer_value.cc"],
    hdrs = ["buffer_value.h"],
    deps = [
        ":hlo_proto_cc",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "buffer_value_containers",
    hdrs = ["buffer_value_containers.h"],
    deps = [
        ":buffer_value",
        ":logical_buffer",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/lib/gtl:compactptrset",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "logical_buffer",
    srcs = ["logical_buffer.cc"],
    hdrs = ["logical_buffer.h"],
    deps = [
        ":buffer_value",
        ":hlo_proto_cc",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/gtl:int_type",
    ],
)

cc_library(
    name = "hlo_value",
    srcs = ["hlo_value.cc"],
    hdrs = ["hlo_value.h"],
    deps = [
        ":buffer_value",
        "//xla:lazy",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_dataflow_analysis",
    srcs = ["hlo_dataflow_analysis.cc"],
    hdrs = ["hlo_dataflow_analysis.h"],
    deps = [
        ":call_graph",
        ":hlo_phi_graph",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_dataflow_analysis_test",
    srcs = ["hlo_dataflow_analysis_test.cc"],
    deps = [
        ":async_op_canonicalizer",
        ":flatten_call_graph",
        ":hlo_creation_utils",
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":instruction_fusion",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_phi_graph",
    srcs = ["hlo_phi_graph.cc"],
    hdrs = ["hlo_phi_graph.h"],
    deps = [
        ":call_graph",
        ":hlo_value",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_phi_graph_test",
    srcs = ["hlo_phi_graph_test.cc"],
    deps = [
        ":hlo_dataflow_analysis",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_phi_graph",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_replication_analysis",
    srcs = ["hlo_replication_analysis.cc"],
    hdrs = ["hlo_replication_analysis.h"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_replication_analysis_test",
    srcs = ["hlo_replication_analysis_test.cc"],
    deps = [
        ":hlo_replication_analysis",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_liveness_analysis",
    srcs = ["hlo_liveness_analysis.cc"],
    hdrs = ["hlo_liveness_analysis.h"],
    deps = [
        ":call_graph",
        ":hlo_value",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_liveness_analysis_test",
    srcs = ["hlo_liveness_analysis_test.cc"],
    deps = [
        ":hlo_liveness_analysis",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_buffer",
    srcs = ["hlo_buffer.cc"],
    hdrs = ["hlo_buffer.h"],
    deps = [
        ":hlo_value",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_alias_analysis",
    srcs = ["hlo_alias_analysis.cc"],
    hdrs = ["hlo_alias_analysis.h"],
    deps = [
        ":hlo_buffer",
        ":hlo_dataflow_analysis",
        ":hlo_ordering",
        ":hlo_value",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_alias_analysis_test",
    srcs = ["hlo_alias_analysis_test.cc"],
    deps = [
        ":flatten_call_graph",
        ":hlo_alias_analysis",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_ordering",
        ":instruction_fusion",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "logical_buffer_analysis",
    srcs = ["logical_buffer_analysis.cc"],
    hdrs = ["logical_buffer_analysis.h"],
    deps = [
        ":logical_buffer",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tuple_points_to_analysis",
    srcs = ["tuple_points_to_analysis.cc"],
    hdrs = ["tuple_points_to_analysis.h"],
    deps = [
        ":hlo_dataflow_analysis",
        ":logical_buffer",
        ":logical_buffer_analysis",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/gtl:compactptrset",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "tuple_points_to_analysis_test",
    srcs = ["tuple_points_to_analysis_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_matchers",
        ":instruction_fusion",
        ":tuple_points_to_analysis",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "compilation_cache",
    srcs = ["compilation_cache.cc"],
    hdrs = ["compilation_cache.h"],
    deps = [
        ":executable",
        ":hlo_module_config",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:strcat",
    ],
)

cc_library(
    name = "layout_assignment",
    srcs = [
        "layout_assignment.cc",
    ],
    hdrs = [
        "layout_assignment.h",
    ],
    deps = [
        ":call_graph",
        ":computation_layout",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_points_to_analysis",
        ":tuple_simplifier",
        "//xla:permutation_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "copy_insertion",
    srcs = ["copy_insertion.cc"],
    hdrs = [
        "compile_time_cap.h",
        "copy_insertion.h",
    ],
    deps = [
        ":dump",
        ":hlo_alias_analysis",
        ":hlo_buffer",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_simplifier",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "loop_schedule_linearizer",
    srcs = ["loop_schedule_linearizer.cc"],
    hdrs = ["loop_schedule_linearizer.h"],
    deps = [
        ":dump",
        ":hlo_alias_analysis",
        ":hlo_dce",
        ":hlo_graph_dumper",
        ":hlo_ordering",
        ":hlo_pass",
        ":logical_buffer",
        ":tuple_simplifier",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service/graphcycles",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "copy_insertion_test",
    srcs = ["copy_insertion_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_runner",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "loop_schedule_linearizer_test",
    srcs = ["loop_schedule_linearizer_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo_graph_dumper",
        ":hlo_matchers",
        ":hlo_runner",
        ":loop_schedule_linearizer",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "memory_space_assignment_utils",
    srcs = ["memory_space_assignment_utils.cc"],
    hdrs = ["memory_space_assignment_utils.h"],
    deps = [
        ":heap_simulator",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "memory_space_assignment_tuning_utils",
    srcs = ["memory_space_assignment_tuning_utils.cc"],
    hdrs = ["memory_space_assignment_tuning_utils.h"],
    deps = [
        ":heap_simulator",
        ":hlo_module_config",
        ":memory_space_assignment_utils",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "memory_space_assignment_repacking",
    hdrs = ["memory_space_assignment_repacking.h"],
    deps = [
        "//xla:statusor",
        "//xla:types",
    ],
)

cc_library(
    name = "memory_space_assignment_best_fit_repacker",
    srcs = ["memory_space_assignment_best_fit_repacker.cc"],
    hdrs = ["memory_space_assignment_best_fit_repacker.h"],
    deps = [
        ":heap_simulator",
        ":memory_space_assignment_repacking",
    ],
)

xla_cc_test(
    name = "memory_space_assignment_best_fit_repacker_test",
    srcs = ["memory_space_assignment_best_fit_repacker_test.cc"],
    deps = [
        ":memory_space_assignment_best_fit_repacker",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "memory_space_assignment",
    srcs = ["memory_space_assignment.cc"],
    hdrs = ["memory_space_assignment.h"],
    deps = [
        ":heap_simulator",
        ":hlo_cost_analysis",
        ":memory_space_assignment_repacking",
        ":memory_space_assignment_tuning_utils",
        ":memory_space_assignment_utils",
        ":tuple_util",
        "//xla:debug_options_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "memory_space_assignment_test",
    srcs = ["memory_space_assignment_test.cc"],
    deps = [
        ":hlo_matchers",
        ":instruction_hoister",
        ":memory_space_assignment",
        ":memory_space_assignment_utils",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "memory_space_propagation",
    srcs = ["memory_space_propagation.cc"],
    hdrs = ["memory_space_propagation.h"],
    deps = [
        ":hlo_dataflow_analysis",
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "memory_space_propagation_test",
    srcs = ["memory_space_propagation_test.cc"],
    deps = [
        ":hlo_parser",
        ":memory_space_propagation",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_dce",
    srcs = ["hlo_dce.cc"],
    hdrs = ["hlo_dce.h"],
    deps = [
        ":hlo_pass",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_module_dce",
    srcs = ["hlo_module_dce.cc"],
    hdrs = ["hlo_module_dce.h"],
    deps = [
        ":hlo_dce",
        ":hlo_liveness_analysis",
        ":hlo_pass",
        ":tuple_simplifier",
        ":while_loop_simplifier",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_verifier",
    srcs = ["hlo_verifier.cc"],
    hdrs = ["hlo_verifier.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_pass",
        ":shape_inference",
        "//xla:comparison_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_verifier_test",
    srcs = ["hlo_verifier_test.cc"],
    deps = [
        ":hlo_module_config",
        ":hlo_parser",
        ":hlo_verifier",
        ":layout_assignment",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_rematerialization",
    srcs = ["hlo_rematerialization.cc"],
    hdrs = ["hlo_rematerialization.h"],
    deps = [
        ":buffer_value",
        ":call_graph",
        ":flatten_call_graph",
        ":hlo_dataflow_analysis",
        ":hlo_dce",
        ":hlo_memory_scheduler",
        ":hlo_ordering",
        ":hlo_query",
        ":hlo_value",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_rematerialization_test_utils",
    testonly = 1,
    hdrs = ["hlo_rematerialization_test_utils.h"],
    deps = [
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/flags:flag",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_rematerialization_test_utils_test",
    srcs = ["hlo_rematerialization_test_utils_test.cc"],
    deps = [
        ":flatten_call_graph",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization_test_utils",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_rematerialization_test",
    srcs = ["hlo_rematerialization_test.cc"],
    deps = [
        ":flatten_call_graph",
        ":hlo_matchers",
        ":hlo_ordering",
        ":hlo_rematerialization",
        ":hlo_rematerialization_test_utils",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_dce_test",
    srcs = ["hlo_dce_test.cc"],
    deps = [
        ":hlo_dce",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_module_dce_test",
    srcs = ["hlo_module_dce_test.cc"],
    deps = [
        ":hlo_module_dce",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "layout_assignment_test",
    srcs = ["layout_assignment_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":computation_layout",
        ":hlo_parser",
        ":layout_assignment",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:literal",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_pass",
    hdrs = [
        "hlo_pass_fix.h",
        "hlo_pass_interface.h",
    ],
    deps = [
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_pass_pipeline",
    srcs = [
        "hlo_pass_pipeline.cc",
    ],
    hdrs = [
        "hlo_pass_pipeline.h",
    ],
    deps = [
        ":compilation_stats",
        ":dump",
        ":hlo_graph_dumper",
        ":hlo_pass",
        ":hlo_proto_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_pass_pipeline_test",
    srcs = ["hlo_pass_pipeline_test.cc"],
    deps = [
        ":hlo_parser",
        ":hlo_pass_pipeline",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_cse",
    srcs = ["hlo_cse.cc"],
    hdrs = ["hlo_cse.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_cse_test",
    srcs = ["hlo_cse_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo_cse",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_constant_folding",
    srcs = ["hlo_constant_folding.cc"],
    hdrs = ["hlo_constant_folding.h"],
    deps = [
        ":hlo_pass",
        ":hlo_query",
        ":slow_operation_alarm",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_constant_folding_test",
    srcs = ["hlo_constant_folding_test.cc"],
    deps = [
        ":hlo_constant_folding",
        ":hlo_matchers",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_domain_map",
    srcs = ["hlo_domain_map.cc"],
    hdrs = ["hlo_domain_map.h"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hlo_domain_verifier",
    srcs = ["hlo_domain_verifier.cc"],
    hdrs = ["hlo_domain_verifier.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hlo_domain_isolator",
    srcs = ["hlo_domain_isolator.cc"],
    hdrs = ["hlo_domain_isolator.h"],
    deps = [
        ":hlo_domain_remover",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_domain_remover",
    srcs = ["hlo_domain_remover.cc"],
    hdrs = ["hlo_domain_remover.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_domain_verifier",
        ":hlo_graph_dumper",
        ":hlo_pass",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_domain_test",
    srcs = ["hlo_domain_test.cc"],
    deps = [
        ":call_inliner",
        ":hlo_domain_isolator",
        ":hlo_domain_remover",
        ":hlo_domain_verifier",
        ":hlo_parser",
        "//xla:debug_options_flags",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/service:sharding_propagation",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "hlo_element_type_converter",
    srcs = ["hlo_element_type_converter.cc"],
    hdrs = ["hlo_element_type_converter.h"],
    deps = [
        ":hlo_pass",
        ":hlo_query",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_element_type_converter_test",
    srcs = ["hlo_element_type_converter_test.cc"],
    deps = [
        ":hlo_element_type_converter",
        ":hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "conditional_canonicalizer",
    srcs = ["conditional_canonicalizer.cc"],
    hdrs = ["conditional_canonicalizer.h"],
    deps = [
        ":hlo_pass",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "conditional_canonicalizer_test",
    srcs = ["conditional_canonicalizer_test.cc"],
    deps = [
        ":conditional_canonicalizer",
        ":hlo_matchers",
        ":hlo_parser",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "maybe_owning_device_memory",
    srcs = [
        "maybe_owning_device_memory.cc",
    ],
    hdrs = [
        "maybe_owning_device_memory.h",
    ],
    deps = [
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:ir_builder_mixin",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "elemental_ir_emitter_test",
    srcs = ["elemental_ir_emitter_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-nvidia",
        "requires-gpu-sm70-only",
        "requires-gpu-sm80-only",
    ]},
    backends = [
        "cpu",
        "gpu",
    ],
    tags = [
        "no_windows",  # TODO(b/152037541)
    ],
    deps = [
        ":hlo_parser",
        "//xla:execution_options_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_module_config",
    srcs = ["hlo_module_config.cc"],
    hdrs = ["hlo_module_config.h"],
    deps = [
        ":computation_layout",
        ":computation_placer",
        ":hlo_proto_cc",
        "//xla:debug_options_flags",
        "//xla:shape_layout",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "computation_layout",
    srcs = ["computation_layout.cc"],
    hdrs = ["computation_layout.h"],
    deps = [
        "//xla:shape_layout",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_graph_dumper",
    srcs = ["hlo_graph_dumper.cc"],
    hdrs = ["hlo_graph_dumper.h"],
    deps = [
        ":pattern_matcher",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/lib/gtl:map_util",
        "@tsl//tsl/lib/io:zlib_compression_options",
        "@tsl//tsl/lib/io:zlib_outputbuffer",
        "@tsl//tsl/platform:base64",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/platform:status",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "hlo_graph_dumper_test",
    srcs = ["hlo_graph_dumper_test.cc"],
    deps = [
        ":hlo_graph_dumper",
        "//xla:literal_util",
        "//xla:test",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "transpose_folding",
    srcs = ["transpose_folding.cc"],
    hdrs = ["transpose_folding.h"],
    deps = [
        ":hlo_pass",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "transpose_folding_test",
    srcs = ["transpose_folding_test.cc"],
    deps = [
        ":hlo_matchers",
        ":shape_inference",
        ":transpose_folding",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "zero_sized_hlo_elimination",
    srcs = ["zero_sized_hlo_elimination.cc"],
    hdrs = ["zero_sized_hlo_elimination.h"],
    deps = [
        ":hlo_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "zero_sized_hlo_elimination_test",
    srcs = ["zero_sized_hlo_elimination_test.cc"],
    deps = [
        ":shape_inference",
        ":zero_sized_hlo_elimination",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "stream_pool",
    srcs = ["stream_pool.cc"],
    hdrs = ["stream_pool.h"],
    deps = [
        "//xla:types",
        "//xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "stream_pool_test",
    srcs = ["stream_pool_test.cc"],
    deps = [
        ":stream_pool",
        "//xla:test_helpers",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_platform",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_proto_util",
    srcs = ["hlo_proto_util.cc"],
    hdrs = ["hlo_proto_util.h"],
    deps = [
        ":buffer_assignment",
        ":hlo_proto_cc",
        ":hlo_verifier",
        "//xla:status",
        "//xla:util",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "hlo_proto_util_test",
    srcs = ["hlo_proto_util_test.cc"],
    deps = [
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_runner_interface",
    srcs = ["hlo_runner_interface.cc"],
    hdrs = ["hlo_runner_interface.h"],
    deps = [
        ":computation_placer",
        ":executable",
        ":hlo_parser",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_runner",
    srcs = ["hlo_runner.cc"],
    hdrs = ["hlo_runner.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_placer",
        ":executable",
        ":hlo_module_util",
        ":hlo_parser",
        ":hlo_runner_interface",
        ":transfer_manager",
        "//third_party/eigen3",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:blocking_counter",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_runner_pjrt",
    srcs = ["hlo_runner_pjrt.cc"],
    hdrs = ["hlo_runner_pjrt.h"],
    deps = [
        ":executable",
        ":hlo_runner_interface",
        "//xla:statusor",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/service:hlo_module_util",
        "//xla/tests:pjrt_client_registry",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_profile_printer",
    srcs = ["hlo_profile_printer.cc"],
    hdrs = ["hlo_profile_printer.h"],
    deps = [
        ":hlo_profile_printer_data_cc",
        ":human_readable_profile_builder",
        "//xla:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sort_simplifier",
    srcs = ["sort_simplifier.cc"],
    hdrs = ["sort_simplifier.h"],
    deps = [
        ":hlo_pass",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "sort_simplifier_test",
    srcs = ["sort_simplifier_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":sort_simplifier",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "stable_sort_expander",
    srcs = ["stable_sort_expander.cc"],
    hdrs = ["stable_sort_expander.h"],
    deps = [
        ":hlo_pass",
        ":op_expander_pass",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "stable_sort_expander_test",
    srcs = ["stable_sort_expander_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":stable_sort_expander",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "tuple_util",
    srcs = ["tuple_util.cc"],
    hdrs = ["tuple_util.h"],
    deps = [
        ":hlo_value",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "tuple_util_test",
    srcs = ["tuple_util_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_module_config",
        ":hlo_parser",
        ":tuple_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "root_instruction_sinker",
    srcs = ["root_instruction_sinker.cc"],
    hdrs = ["root_instruction_sinker.h"],
    deps = [
        ":hlo_pass",
        ":tuple_util",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "root_instruction_sinker_test",
    srcs = ["root_instruction_sinker_test.cc"],
    deps = [
        ":hlo_matchers",
        ":root_instruction_sinker",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "while_util",
    srcs = ["while_util.cc"],
    hdrs = ["while_util.h"],
    deps = [
        ":call_inliner",
        ":hlo_creation_utils",
        ":tuple_util",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "while_util_test",
    srcs = ["while_util_test.cc"],
    deps = [
        ":hlo_matchers",
        ":while_util",
        "//xla:test",
        "//xla:util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "while_loop_all_reduce_code_motion",
    srcs = ["while_loop_all_reduce_code_motion.cc"],
    hdrs = ["while_loop_all_reduce_code_motion.h"],
    deps = [
        ":call_graph",
        ":collective_ops_utils",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_replication_analysis",
        "//xla:literal_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_all_reduce_code_motion_test",
    srcs = ["while_loop_all_reduce_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_verifier",
        ":while_loop_all_reduce_code_motion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "while_loop_concat_code_motion",
    srcs = ["while_loop_concat_code_motion.cc"],
    hdrs = ["while_loop_concat_code_motion.h"],
    deps = [
        ":hlo_dce",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":tuple_simplifier",
        ":while_loop_simplifier",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "while_loop_concat_code_motion_test",
    srcs = ["while_loop_concat_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_verifier",
        ":while_loop_concat_code_motion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "while_loop_invariant_code_motion",
    srcs = ["while_loop_invariant_code_motion.cc"],
    hdrs = [
        "compile_time_cap.h",
        "while_loop_invariant_code_motion.h",
    ],
    deps = [
        ":hlo_dce",
        ":hlo_pass",
        ":tuple_util",
        ":while_loop_analysis",
        ":while_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "while_loop_invariant_code_motion_test",
    srcs = ["while_loop_invariant_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":while_loop_invariant_code_motion",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "while_loop_expensive_invariant_code_motion",
    srcs = ["while_loop_expensive_invariant_code_motion.cc"],
    hdrs = ["while_loop_expensive_invariant_code_motion.h"],
    deps = [
        ":hlo_pass",
        ":tuple_util",
        ":while_loop_analysis",
        ":while_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

xla_cc_test(
    name = "while_loop_expensive_invariant_code_motion_test",
    srcs = ["while_loop_expensive_invariant_code_motion_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":while_loop_expensive_invariant_code_motion",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "while_loop_constant_sinking",
    srcs = ["while_loop_constant_sinking.cc"],
    hdrs = ["while_loop_constant_sinking.h"],
    deps = [
        ":hlo_pass",
        ":while_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

xla_cc_test(
    name = "while_loop_constant_sinking_test",
    srcs = ["while_loop_constant_sinking_test.cc"],
    deps = [
        ":hlo_matchers",
        ":while_loop_constant_sinking",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "despecializer",
    srcs = ["despecializer.cc"],
    hdrs = ["despecializer.h"],
    deps = [
        ":bfloat16_normalization",
        ":defuser",
        ":hlo_memory_scheduler",
        ":hlo_pass",
        ":hlo_pass_pipeline",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "source_map_util",
    srcs = [],
    hdrs = ["source_map_util.h"],
    deps = [
        ":executable",
        "//xla:status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "indexed_array_analysis",
    srcs = ["indexed_array_analysis.cc"],
    hdrs = ["indexed_array_analysis.h"],
    deps = [
        ":hlo_pass",
        "//xla:util",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/util:ptr_util",
    ],
)

xla_cc_test(
    name = "indexed_array_analysis_test",
    srcs = ["indexed_array_analysis_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":indexed_array_analysis",
        "//xla:test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_parser",
    srcs = ["hlo_parser.cc"],
    hdrs = ["hlo_parser.h"],
    deps = [
        ":computation_layout",
        ":hlo_lexer",
        ":shape_inference",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@tsl//tsl/lib/gtl:map_util",
    ],
)

xla_cc_test(
    name = "hlo_parser_test",
    size = "small",
    srcs = ["hlo_parser_test.cc"],
    deps = [
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:shape_util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = [
        "hlo_lexer.h",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:regexp",
    ],
)

cc_library(
    name = "map_inliner",
    srcs = ["map_inliner.cc"],
    hdrs = ["map_inliner.h"],
    deps = [
        ":hlo_pass",
        ":hlo_query",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "optimize_input_output_buffer_alias",
    srcs = ["optimize_input_output_buffer_alias.cc"],
    hdrs = ["optimize_input_output_buffer_alias.h"],
    deps = [
        ":hlo_pass",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "optimize_input_output_buffer_alias_test",
    srcs = ["optimize_input_output_buffer_alias_test.cc"],
    deps = [
        ":optimize_input_output_buffer_alias",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "ar_crs_combiner",
    srcs = ["ar_crs_combiner.cc"],
    hdrs = ["ar_crs_combiner.h"],
    deps = [
        ":call_graph",
        ":hlo_pass",
        ":hlo_query",
        ":hlo_replication_analysis",
        ":pattern_matcher",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compilation_stats",
    srcs = ["compilation_stats.cc"],
    hdrs = ["compilation_stats.h"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "dynamic_index_splitter",
    srcs = ["dynamic_index_splitter.cc"],
    hdrs = ["dynamic_index_splitter.h"],
    deps = [
        ":hlo_pass",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "dynamic_index_splitter_test",
    srcs = ["dynamic_index_splitter_test.cc"],
    deps = [
        ":dynamic_index_splitter",
        ":hlo_matchers",
        "//xla:statusor",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "ar_crs_combiner_test",
    srcs = ["ar_crs_combiner_test.cc"],
    deps = [
        ":ar_crs_combiner",
        ":hlo_matchers",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "map_inliner_test",
    srcs = ["map_inliner_test.cc"],
    deps = [
        ":hlo_matchers",
        ":map_inliner",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_test(
    name = "hlo_casting_utils_test",
    srcs = ["hlo_casting_utils_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "conditional_to_select",
    srcs = ["conditional_to_select.cc"],
    hdrs = ["conditional_to_select.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "conditional_to_select_test",
    srcs = ["conditional_to_select_test.cc"],
    deps = [
        ":conditional_to_select",
        ":hlo_matchers",
        "//xla:literal",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "slice_sinker",
    srcs = ["slice_sinker.cc"],
    hdrs = ["slice_sinker.h"],
    deps = [
        ":hlo_pass",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "custom_call_target_registry",
    srcs = ["custom_call_target_registry.cc"],
    hdrs = ["custom_call_target_registry.h"],
    visibility = ["//visibility:public"],
)

# Exposes the public interface only and hides internal details. Suitable for
# linking into a static library or binary.
cc_library(
    name = "custom_call_status",
    srcs = [
        "custom_call_status.cc",
        "custom_call_status_internal.h",
    ],
    hdrs = ["custom_call_status.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "custom_call_status_hdrs",
    srcs = [
        "custom_call_status.h",
        "custom_call_status_internal.h",
    ],
    visibility = [":friends"],
)

filegroup(
    name = "custom_call_status_srcs",
    srcs = ["custom_call_status.cc"],
    visibility = [":friends"],
)

# Internal version that exposes internal details and private interfaces. For
# use by other XLA code only.
cc_library(
    name = "custom_call_status_internal",
    hdrs = [
        "custom_call_status_internal.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        ":__subpackages__",
        "//third_party/tensorflow/compiler/tf2xla:__pkg__",
    ],
    deps = [
        ":custom_call_status",
        "@com_google_absl//absl/strings",
    ],
)

# Public headers only, suitable for inclusion in a shared library of CustomCall
# target functions.
cc_library(
    name = "custom_call_status_public_headers",
    hdrs = ["custom_call_status.h"],
    visibility = ["//visibility:public"],
)

xla_cc_test(
    name = "custom_call_status_test",
    srcs = ["custom_call_status_test.cc"],
    deps = [
        ":custom_call_status_internal",
        ":custom_call_status_test_c_caller",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "custom_call_status_test_c_caller",
    testonly = True,
    srcs = ["custom_call_status_test_c_caller.c"],
    hdrs = ["custom_call_status_test_c_caller.h"],
    deps = [":custom_call_status"],
)

xla_cc_test(
    name = "slice_sinker_test",
    srcs = ["slice_sinker_test.cc"],
    deps = [
        ":hlo_dce",
        ":hlo_parser",
        ":hlo_pass",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        ":slice_sinker",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "rng_expander",
    srcs = ["rng_expander.cc"],
    hdrs = ["rng_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/client:xla_builder",
        "//xla/client/lib:prng",
    ],
)

cc_library(
    name = "rng_bit_generator_expander",
    srcs = ["rng_bit_generator_expander.cc"],
    hdrs = ["rng_bit_generator_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client/lib:prng",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "slow_operation_alarm",
    srcs = ["slow_operation_alarm.cc"],
    hdrs = ["slow_operation_alarm.h"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "collective_ops_utils",
    srcs = ["collective_ops_utils.cc"],
    hdrs = ["collective_ops_utils.h"],
    deps = [
        ":computation_placer",
        ":global_device_id",
        ":pattern_matcher",
        "//xla:executable_run_options",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/functional:function_ref",
        "@tsl//tsl/platform:blocking_counter",
    ],
)

xla_cc_test(
    name = "collective_ops_utils_test",
    srcs = ["collective_ops_utils_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":computation_placer",
        ":global_device_id",
        "//xla:xla_data_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "topk_rewriter",
    srcs = ["topk_rewriter.cc"],
    hdrs = ["topk_rewriter.h"],
    deps = [
        ":hlo_pass",
        ":pattern_matcher",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "topk_rewriter_test",
    srcs = ["topk_rewriter_test.cc"],
    deps = [
        ":hlo_dce",
        ":hlo_matchers",
        ":topk_rewriter",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_macros_cpu",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "operand_upcaster",
    srcs = ["operand_upcaster.cc"],
    hdrs = ["operand_upcaster.h"],
    deps = [
        ":hlo_creation_utils",
        ":op_expander_pass",
        ":shape_inference",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "operand_upcaster_test",
    srcs = ["operand_upcaster_test.cc"],
    deps = [
        ":hlo_matchers",
        ":operand_upcaster",
        "//xla:shape_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "result_caster",
    srcs = ["result_caster.cc"],
    hdrs = ["result_caster.h"],
    deps = [
        ":op_expander_pass",
        ":shape_inference",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "result_caster_test",
    srcs = ["result_caster_test.cc"],
    deps = [
        ":hlo_matchers",
        ":result_caster",
        "//xla:shape_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "global_device_id",
    srcs = ["global_device_id.cc"],
    hdrs = ["global_device_id.h"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/gtl:int_type",
    ],
)

cc_library(
    name = "convert_operand_folding",
    srcs = ["convert_operand_folding.cc"],
    hdrs = ["convert_operand_folding.h"],
    deps = [
        ":op_expander_pass",
        "//xla:comparison_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
    ],
)

xla_cc_test(
    name = "convert_operand_folding_test",
    srcs = ["convert_operand_folding_test.cc"],
    deps = [
        ":convert_operand_folding",
        ":hlo_matchers",
        "//xla:shape_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xla_debug_info_manager",
    srcs = [
        "xla_debug_info_manager.cc",
    ],
    hdrs = [
        "xla_debug_info_manager.h",
    ],
    deps = [
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "xla_debug_info_manager_test",
    srcs = ["xla_debug_info_manager_test.cc"],
    deps = [
        ":hlo_proto_cc",
        ":xla_debug_info_manager",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_py_proto_library(
    name = "hlo_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":hlo_proto"],
)

py_library(
    name = "generate_test_hlo_checks",
    srcs = ["generate_test_hlo_checks.py"],
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "generate_test_hlo_checks_test",
    srcs = ["generate_test_hlo_checks_test.py"],
    python_version = "PY3",
    # TODO(b/200806426): Test fails in OSS.
    tags = [
        "no_oss",
        "nopip",
    ],
    deps = [
        ":generate_test_hlo_checks",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "mapped_ptr_container_sorter",
    hdrs = ["mapped_ptr_container_sorter.h"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "mapped_ptr_container_sorter_test",
    srcs = ["mapped_ptr_container_sorter_test.cc"],
    deps = [
        ":mapped_ptr_container_sorter",
        "//xla:test",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/functional:bind_front",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "rendezvous",
    srcs = ["rendezvous.cc"],
    hdrs = ["rendezvous.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "compilation_environments",
    srcs = ["compilation_environments.cc"],
    hdrs = ["compilation_environments.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:logging",  # fixdeps: keep
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "custom_call_sharding_helper",
    srcs = ["custom_call_sharding_helper.cc"],
    hdrs = ["custom_call_sharding_helper.h"],
    deps = ["//xla/hlo/ir:hlo"],
)

tf_proto_library(
    name = "test_compilation_environment_proto",
    testonly = 1,
    srcs = ["test_compilation_environment.proto"],
    cc_api_version = 2,
)

xla_cc_test(
    name = "compilation_environments_test",
    srcs = ["compilation_environments_test.cc"],
    deps = [
        ":compilation_environments",
        ":test_compilation_environment_proto_cc",
        "//xla:test",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "layout_normalization",
    srcs = ["layout_normalization.cc"],
    hdrs = ["layout_normalization.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_creation_utils",
        ":hlo_pass",
        ":pattern_matcher",
        ":shape_inference",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:padding",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "instruction_hoister",
    srcs = ["instruction_hoister.cc"],
    hdrs = ["instruction_hoister.h"],
    deps = [
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "scatter_simplifier",
    srcs = ["scatter_simplifier.cc"],
    hdrs = ["scatter_simplifier.h"],
    deps = [
        ":gather_scatter_utils",
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "scatter_simplifier_test",
    srcs = ["scatter_simplifier_test.cc"],
    deps = [
        ":hlo_pass",
        ":hlo_pass_pipeline",
        ":scatter_simplifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "select_and_scatter_expander",
    srcs = ["select_and_scatter_expander.cc"],
    hdrs = ["select_and_scatter_expander.h"],
    deps = [
        ":call_inliner",
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "select_and_scatter_expander_test",
    srcs = ["select_and_scatter_expander_test.cc"],
    deps = [
        ":hlo_matchers",
        ":select_and_scatter_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "layout_normalization_test",
    srcs = [
        "layout_normalization_test.cc",
    ],
    deps = [
        ":layout_normalization",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/stream_executor/lib",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "change_op_data_type",
    srcs = ["change_op_data_type.cc"],
    hdrs = ["change_op_data_type.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "gather_scatter_utils",
    srcs = ["gather_scatter_utils.cc"],
    hdrs = ["gather_scatter_utils.h"],
    deps = [
        ":hlo_creation_utils",
        "//xla:permutation_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gather_simplifier",
    srcs = ["gather_simplifier.cc"],
    hdrs = ["gather_simplifier.h"],
    deps = [
        ":gather_scatter_utils",
        ":hlo_creation_utils",
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "stochastic_convert_decomposer",
    srcs = ["stochastic_convert_decomposer.cc"],
    hdrs = ["stochastic_convert_decomposer.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_pass",
        ":shape_inference",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "stochastic_convert_decomposer_test",
    srcs = ["stochastic_convert_decomposer_test.cc"],
    deps = [
        ":hlo_matchers",
        ":hlo_parser",
        ":stochastic_convert_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "metrics_hook_interface",
    hdrs = ["metrics_hook_interface.h"],
    deps = ["@com_google_absl//absl/strings"],
)

xla_cc_test(
    name = "gather_simplifier_test",
    srcs = ["gather_simplifier_test.cc"],
    deps = [
        ":gather_simplifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "change_op_data_type_test",
    srcs = ["change_op_data_type_test.cc"],
    deps = [
        ":change_op_data_type",
        ":hlo_matchers",
        ":hlo_parser",
        ":pattern_matcher",
        ":pattern_matcher_gmock",
        "//xla:permutation_util",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_binary(
    name = "xla_compile",
    srcs = if_cuda_is_configured(["xla_compile_main.cc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":compiler",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "//xla:statusor",
        "//xla/mlir_hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/gpu:executable_proto_cc",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:nvptx_compiler_impl",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/util:command_line_flags",
        "@stablehlo//:register",
    ]),
)

# A simple test of xla_aot_compile which generates an output file from an mhlo file.
xla_aot_compile_cpu(
    name = "xla_aot_compile_test_cpu_executable",
    module = "xla_aot_compile_test.mlir",
)

xla_aot_compile_cpu(
    name = "xla_aot_compile_stablehlo_test_cpu_executable",
    module = "xla_aot_compile_stablehlo_test.mlir",
)

xla_aot_compile_gpu(
    name = "xla_aot_compile_test_gpu_executable",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.prototxt",
    module = "xla_aot_compile_test.mlir",
)

xla_aot_compile_gpu(
    name = "xla_aot_compile_test_gpu_executable_constant",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.prototxt",
    module = "xla_aot_compile_test_constant.mlir",
)

xla_cc_test(
    name = "xla_aot_compile_cpu_test",
    srcs = if_cuda_is_configured(["xla_aot_compile_cpu_test.cc"]),
    data = if_cuda_is_configured([
        ":xla_aot_compile_test_cpu_executable",
    ]),
    tags = ["no_oss"],
    deps = if_cuda_is_configured([
        ":cpu_plugin",
        ":platform_util",
        "@com_google_googletest//:gtest_main",
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service/cpu:cpu_compiler",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:resource_loader",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ]),
)

xla_cc_test(
    name = "xla_aot_compile_stablehlo_cpu_test",
    srcs = if_cuda_is_configured(["xla_aot_compile_stablehlo_cpu_test.cc"]),
    data = if_cuda_is_configured([
        ":xla_aot_compile_stablehlo_test_cpu_executable",
    ]),
    tags = ["no_oss"],
    deps = if_cuda_is_configured([
        ":cpu_plugin",
        ":platform_util",
        "@com_google_googletest//:gtest_main",
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:literal_test_util",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:resource_loader",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ]),
)

xla_cc_test(
    name = "xla_aot_compile_gpu_test",
    srcs = if_cuda_is_configured(["xla_aot_compile_gpu_test.cc"]),
    data = if_cuda_is_configured([
        ":xla_aot_compile_test_gpu_executable",
        ":xla_aot_compile_test_gpu_executable_constant",
    ]),
    env = {
        "XLA_FLAGS": "--xla_gpu_enable_xla_runtime_executable",
    },
    tags = [
        "gpu",
        "no_oss",
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false positives in msan.
        "requires-gpu-sm60-only",
    ],
    deps = if_cuda_is_configured([
        ":gpu_plugin_impl",
        ":platform_util",
        "@com_google_googletest//:gtest_main",
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service/cpu:cpu_compiler",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:resource_loader",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ]),
)
