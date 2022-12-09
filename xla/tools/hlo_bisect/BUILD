# Description:
#   A tool for reducing a HLO module that produces incorrect results.
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")
load("//xla:xla.bzl", "xla_cc_binary")

package(
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "hlo_bisect",
    testonly = True,
    srcs = ["hlo_bisect.cc"],
    deps = [
        ":hlo_bisect_utils",
        "//xla:debug_options_flags",
        "//xla/service:backend",
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
        "//xla/service:interpreter_plugin",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "hlo_bisect_state",
    srcs = ["hlo_bisect_state.cc"],
    hdrs = ["hlo_bisect_state.h"],
    deps = [
        "//xla:literal",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_dce",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "hlo_bisect_state_test",
    srcs = ["hlo_bisect_state_test.cc"],
    deps = [
        ":hlo_bisect_state",
        "//xla:literal",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "hlo_bisect_utils",
    testonly = True,
    srcs = ["hlo_bisect_utils.cc"],
    hdrs = ["hlo_bisect_utils.h"],
    deps = [
        ":hlo_bisect_state",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_util",
        "//xla/service:hlo_runner",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_verifier",
        "//xla/service:platform_util",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tools:prepare_reference_module",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:subprocess",
    ],
)
