load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "hlo_runner_base_lib",
    srcs = [
        "hlo_runner.cc",
    ],
    hdrs = [
        "hlo_runner.h",
    ],
    deps = [
        "//third_party/tensorflow/compiler/jit:xla_gpu_jit",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/service:computation_placer",
        "//xla/service:global_device_id",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_control_flow_flattening",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_runner_lib",
    hdrs = [
        "hlo_runner.h",
    ],
    deps = [
        ":hlo_runner_base_lib",
        "//xla:status",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/service:computation_placer_hdr",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "hlo_runner_test_1x1",
    srcs = ["hlo_runner_test.cc"],
    backends = ["gpu"],
    data = [
        ":test_data/test_hlo.pb",
        ":test_data/test_hlo.txt",
    ],
    shard_count = 10,
    deps = [
        "hlo_runner_lib",
        "//xla:literal_util",
        "//xla/service:hlo_parser",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

xla_test(
    name = "hlo_runner_test_2x2",
    srcs = ["hlo_runner_multichip_test.cc"],
    backend_tags = {
        # This test is tagged "manual" because it requires 4 GPUs.
        "gpu": [
            "manual",
            "multi_gpu",
            "no_oss",
            "notap",
        ],
    },
    backends = ["gpu"],
    deps = [
        "hlo_runner_lib",
        "//xla:literal_util",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/lib/core:status_test_util",
    ],
)

xla_test(
    name = "hlo_partition_id_test",
    srcs = ["hlo_partition_id_test.cc"],
    backend_tags = {
        # This test is tagged "manual" because it requires 2 GPUs.
        "gpu": [
            "manual",
            "multi_gpu",
            "no_oss",
            "notap",
        ],
    },
    backends = ["gpu"],
    deps = [
        "hlo_runner_lib",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "multihost_collective_ops_test",
    srcs = ["multihost_collective_ops_test.cc"],
    backend_tags = {
        # This test is tagged "manual" because it requires 8 GPUs.
        "gpu": [
            "manual",
            "multi_gpu",
            "no_oss",
            "notap",
        ],
    },
    backends = ["gpu"],
    deps = [
        "hlo_runner_lib",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_runner_utils_lib",
    srcs = [
        "hlo_runner_utils.cc",
    ],
    hdrs = [
        "hlo_runner_utils.h",
    ],
    deps = [
        ":hlo_runner_base_lib",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_placer_hdr",
        "//xla/tests:test_utils",
    ],
)
