# Description:
#   XLA evaluator implementation.

load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")

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
        "hlo_evaluator_typed_visitor_half.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        "//tensorflow/tsl/lib/core:bitmap",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:types",
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
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:hlo_query",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/service:tuple_points_to_analysis",
        "//xla/service/cpu:runtime_single_threaded_matmul",
        "//xla/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_evaluator_test",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo_evaluator",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
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
    ],
)
