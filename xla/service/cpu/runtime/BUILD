load("@tsl//platform:rules_cc.bzl", "cc_library")

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
    name = "collectives",
    srcs = ["collectives.cc"],
    hdrs = ["collectives.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service/cpu:cpu_runtime",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:custom_call_status_internal",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_proto_cc",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "fft_call",
    srcs = ["fft_call.cc"],
    hdrs = ["fft_call.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:xla_proto_cc",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service:hlo_proto_cc",
        "//xla/service/cpu:runtime_fft",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "xfeed",
    srcs = ["xfeed.cc"],
    hdrs = ["xfeed.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla/runtime:custom_call",
        "//xla/runtime:custom_call_registry",
        "//xla/runtime:executable",
        "//xla/service/cpu:cpu_runtime",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
