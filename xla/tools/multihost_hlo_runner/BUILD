load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_binary(
    name = "hlo_runner_main",
    srcs = ["hlo_runner_main.cc"],
    deps = [
        ":hlo_runner_flags",
        ":functional_hlo_runner",
        "@com_google_absl//absl/strings",
        "//xla:debug_options_flags",
        # This is needed to run on GPU.
        "//third_party/tensorflow/compiler/jit:xla_gpu_jit",
        # "//xla/stream_executor/cuda:cuda_platform",
        "//xla:status",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "functional_hlo_runner",
    srcs = ["functional_hlo_runner.cc"],
    hdrs = ["functional_hlo_runner.h"],
    deps = [
        "//xla:literal",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/service:hlo_parser",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_control_flow_flattening",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_runner_flags",
    srcs = ["hlo_runner_flags.cc"],
    hdrs = ["hlo_runner_flags.h"],
    deps = [
        ":functional_hlo_runner",
        "//xla/pjrt:pjrt_executable",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "functional_hlo_runner_test",
    srcs = ["functional_hlo_runner_test.cc"],
    backend_tags = {
        # This test is tagged "manual" because it requires multiple (2) GPUs.
        "gpu": [
            "manual",
            "multi_gpu",
            "no_oss",
            "notap",
        ],
    },
    backends = ["gpu"],
    data = [
        "data/sharded_2_devices.hlo",
        "data/single_device.hlo",
    ],
    deps = [
        ":functional_hlo_runner",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)
