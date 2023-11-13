
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
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
