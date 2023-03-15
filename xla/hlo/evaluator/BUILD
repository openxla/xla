# Description:
#   XLA evaluator implementation.

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "hlo_evaluator",
    srcs = [
        "hlo_evaluator.cc",
        "hlo_evaluator_typed_visitor.h",
        "hlo_evaluator_typed_visitor_bfloat16.cc",
        "hlo_evaluator_typed_visitor_bool.cc",
        "hlo_evaluator_typed_visitor_complex128.cc",
        "hlo_evaluator_typed_visitor_complex64.cc",
        "hlo_evaluator_typed_visitor_double.cc",
        "hlo_evaluator_typed_visitor_float.cc",
        "hlo_evaluator_typed_visitor_float8.cc",
        "hlo_evaluator_typed_visitor_half.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int4.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        "//xla:array2d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/service:compilation_environments",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:hlo_query",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/service:tuple_points_to_analysis",
        "//xla/service/cpu:runtime_single_threaded_matmul",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/core:bitmap",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:types",
    ],
)

xla_cc_test(
    name = "hlo_evaluator_test",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo_evaluator",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:reference_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:test",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_element_type_converter",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ],
)
