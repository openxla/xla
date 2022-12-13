# Tools and utilities that aid in XLA development and usage.

load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@tsl//tsl:tsl.bzl",
    "if_cuda_or_rocm",
)
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_py_proto_library",
)
load(
    "@tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//xla:internal"],
)

xla_cc_binary(
    name = "hex_floats_to_packed_literal",
    srcs = ["hex_floats_to_packed_literal.cc"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/io:buffered_inputstream",
        "@tsl//tsl/lib/io:random_inputstream",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/util:command_line_flags",
    ],
)

xla_cc_binary(
    name = "show_signature",
    srcs = ["show_signature.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service:hlo_proto_cc",
        "//xla/service:interpreter_plugin",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "replay_computation_library",
    srcs = ["replay_computation.cc"],
    deps = [
        "//third_party/eigen3",
        "//xla:debug_options_flags",
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_computation",
        "//xla/client/lib:testing",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/lib/io:record_reader",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:tstring",
        "@tsl//tsl/util:command_line_flags",
    ],
    alwayslink = True,
)

xla_cc_binary(
    name = "replay_computation_cpu",
    deps = [
        ":replay_computation_library",
        "//xla/service:cpu_plugin",
    ],
)

xla_cc_binary(
    name = "replay_computation_gpu",
    tags = ["gpu"],
    deps = [
        ":replay_computation_library",
        "//xla/service:gpu_plugin",
    ],
)

xla_cc_binary(
    name = "replay_computation_interpreter",
    deps = [
        ":replay_computation_library",
        "//xla/service:interpreter_plugin",
    ],
)

xla_cc_binary(
    name = "show_literal",
    srcs = ["show_literal.cc"],
    deps = [
        "//xla:literal",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "convert_computation",
    srcs = ["convert_computation.cc"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla/service:hlo_proto_cc",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_binary(
    name = "show_text_literal",
    srcs = ["show_text_literal.cc"],
    deps = [
        "//xla:literal",
        "//xla:statusor",
        "//xla:text_literal_reader",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
    ],
)

xla_cc_binary(
    name = "dumped_computation_to_text",
    srcs = ["dumped_computation_to_text.cc"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/client:xla_computation",
        "//xla/service",
        "//xla/service:hlo_proto_cc",
        "//xla/service:interpreter_plugin",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "dumped_computation_to_operation_list",
    srcs = ["dumped_computation_to_operation_list.cc"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service",
        "//xla/service:hlo_proto_cc",
        "//xla/service:interpreter_plugin",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "hlo_proto_to_json",
    srcs = ["hlo_proto_to_json.cc"],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/service:hlo_proto_cc",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/util:command_line_flags",
    ],
)

xla_cc_test(
    name = "hlo_extractor_test",
    srcs = ["hlo_extractor_test.cc"],
    deps = [
        ":hlo_extractor",
        "//xla/service:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_extractor",
    srcs = ["hlo_extractor.cc"],
    hdrs = ["hlo_extractor.h"],
    deps = [
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_verifier",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

xla_cc_binary(
    name = "interactive_graphviz",
    srcs = ["interactive_graphviz.cc"],
    deps = [
        ":hlo_extractor",
        "//xla/service:hlo_graph_dumper",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_runner",
        "//xla/service:local_service",
        "//xla/service:platform_util",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:subprocess",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@tsl//tsl/util:command_line_flags",
    ] + if_cuda_or_rocm([
        "//xla/service:gpu_plugin",
    ]),
)

sh_test(
    name = "interactive_graphviz_build_only_test",
    srcs = ["interactive_graphviz_test.sh"],
    data = [":interactive_graphviz"],
)

cc_library(
    name = "hlo_module_loader",
    srcs = ["hlo_module_loader.cc"],
    hdrs = ["hlo_module_loader.h"],
    visibility = ["//xla:friends"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:regexp",
    ],
)

xla_cc_test(
    name = "hlo_module_loader_test",
    srcs = ["hlo_module_loader_test.cc"],
    deps = [
        ":hlo_module_loader",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "prepare_reference_module",
    srcs = ["prepare_reference_module.cc"],
    hdrs = ["prepare_reference_module.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:despecializer",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/lib",
        "@tsl//tsl/platform:errors",
    ],
)

tf_proto_library(
    name = "run_hlo_module_proto",
    srcs = ["run_hlo_module.proto"],
    cc_api_version = 2,
    protodeps = [
        "//xla:xla_data_proto",
    ],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "run_hlo_module_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":run_hlo_module_proto"],
)

cc_library(
    name = "run_hlo_module_lib",
    srcs = ["run_hlo_module.cc"],
    hdrs = ["run_hlo_module.h"],
    deps = [
        ":hlo_control_flow_flattening",
        ":hlo_module_loader",
        ":prepare_reference_module",
        ":run_hlo_module_proto_cc",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client/lib:testing",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_verifier",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "run_hlo_module",
    testonly = True,
    srcs = ["run_hlo_module_main.cc"],
    tags = [
        "noasan",  # Exceeds linker limit.
    ],
    deps = [
        ":run_hlo_module_lib",
        "@com_google_absl//absl/strings",
        "//xla/service:cpu_plugin",
        "//xla/service:hlo_runner",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "//xla:debug_options_flags",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/util:command_line_flags",
    ] + if_cuda_or_rocm([
        "//xla/service:gpu_plugin",
    ]),
)

cc_library(
    name = "hlo_control_flow_flattening",
    srcs = ["hlo_control_flow_flattening.cc"],
    hdrs = ["hlo_control_flow_flattening.h"],
    deps = [
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_pass",
        "//xla/service:tuple_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_control_flow_flattening_test",
    srcs = ["hlo_control_flow_flattening_test.cc"],
    deps = [
        ":hlo_control_flow_flattening",
        "//xla/service:collective_ops_utils",
        "//xla/service:despecializer",
        "//xla/service:hlo_matchers",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

# This target is used to reproduce miscompiles in OSS outside of TF, and it can
# not have any dependencies apart from the standard library.
cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    tags = ["nofixdeps"],
    deps = [],
)

xla_cc_binary(
    name = "compute_cost",
    srcs = ["compute_cost.cc"],
    deps = [
        ":hlo_module_loader",
        "//xla/service:hlo_cost_analysis",
        "@tsl//tsl/platform:platform_port",
    ],
)
