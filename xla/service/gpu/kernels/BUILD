load("//xla/tests:build_defs.bzl", "xla_test")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

cc_library(
    name = "custom_fusion",
    srcs = ["custom_fusion.cc"],
    hdrs = ["custom_fusion.h"],
    deps = [
        ":custom_kernel",
        "//xla:status",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "custom_kernel",
    srcs = ["custom_kernel.cc"],
    hdrs = ["custom_kernel.h"],
    deps = [
        "//xla:statusor",
        "//xla/stream_executor",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

# copybara:uncomment_begin(google-only)
# # TODO(ezhulenev): We currently do not have a CUTLASS dependency in open source BUILD.
#
# cc_library(
#     name = "cutlass_gemm_fusion",
#     srcs = ["cutlass_gemm_fusion.cc"],
#     deps = [
#         ":custom_fusion",
#         ":custom_kernel",
#         ":cutlass_gemm_kernel",
#         "@com_google_absl//absl/status",
#         "//xla:statusor",
#         "//xla:xla_data_proto_cc",
#         "//xla/hlo/ir:hlo",
#         "@tsl//tsl/platform:errors",
#         "@tsl//tsl/platform:logging",
#         "@tsl//tsl/platform:statusor",
#     ],
#     alwayslink = 1,  # static fusion registration
# )
#
# xla_test(
#     name = "cutlass_gemm_fusion_test",
#     srcs = ["cutlass_gemm_fusion_test.cc"],
#     backends = ["gpu"],
#     deps = [
#         ":cutlass_gemm_fusion",
#         "@com_google_absl//absl/strings",
#         "//xla:debug_options_flags",
#         "//xla:error_spec",
#         "//xla/tests:hlo_test_base",
#         "@tsl//tsl/platform:test",
#         "@tsl//tsl/platform:test_main",
#     ],
# )
#
# cuda_library(
#     name = "cutlass_gemm_kernel",
#     srcs = ["cutlass_gemm_kernel.cu.cc"],
#     hdrs = ["cutlass_gemm_kernel.h"],
#     visibility = ["//visibility:private"],
#     deps = [
#         ":custom_kernel",
#         "//third_party/gpus/cutlass",
#         "//xla:statusor",
#         "//xla:xla_data_proto_cc",
#         "//xla/stream_executor",
#     ],
# )
#
# xla_test(
#     name = "cutlass_gemm_test",
#     srcs = if_cuda_is_configured(["cutlass_gemm_test.cc"]),
#     backends = ["gpu"],
#     deps = [
#         ":cutlass_gemm_kernel",
#         "//xla:types",
#         "//xla:xla_data_proto_cc",
#         "//xla/stream_executor",
#         "//xla/stream_executor:multi_platform_manager",
#         "//xla/stream_executor:platform",
#         "//xla/stream_executor/cuda:cuda_platform",
#         "@tsl//tsl/lib/core:status_test_util",
#         "@tsl//tsl/platform:status",
#         "@tsl//tsl/platform:test",
#         "@tsl//tsl/platform:test_benchmark",
#         "@tsl//tsl/platform:test_main",
#     ],
# )
#
# copybara:uncomment_end(google-only)
