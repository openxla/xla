load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# copybara:uncomment package(default_applicable_licenses = ["//third_party/tensorflow:license"])

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/tpu:c_api_decl",  # TODO(b/238999986): Remove this.
    ],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

# TODO(269360974): remove this target after pjrt_c_api_client no longer needs
# to access wrapped objects.
cc_library(
    name = "pjrt_c_api_wrapper_impl_header",
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_future",
    ],
)

cc_library(
    name = "pjrt_c_api_wrapper_impl",
    srcs = ["pjrt_c_api_wrapper_impl.cc"],
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor/tpu:c_api_conversions",  # TODO(b/238999986): Remove this.
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//xla:shape_util",
        "//xla:status",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu",
    srcs = ["pjrt_c_api_cpu.cc"],
    hdrs = ["pjrt_c_api_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu",
    srcs = ["pjrt_c_api_gpu.cc"],
    hdrs = ["pjrt_c_api_gpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//xla/pjrt/gpu:gpu_helpers",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
    ],
)

cc_library(
    name = "pjrt_c_api_test_common",
    testonly = 1,
    srcs = ["pjrt_c_api_test.cc"],
    hdrs = ["pjrt_c_api_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/pjrt:pjrt_client",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "pjrt_c_api_test_gpu",
    srcs = ["pjrt_c_api_test_gpu.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":pjrt_c_api_gpu",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_test_common",
        "//xla/pjrt:pjrt_client",
        "//xla/service:gpu_plugin",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_cpu_test",
    srcs = ["pjrt_c_api_cpu_test.cc"],
    deps = [
        ":pjrt_c_api_cpu",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//xla/pjrt:pjrt_client",
        "@com_google_googletest//:gtest_main",
    ],
)
