load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl/platform:build_config.bzl", "tf_platform_deps")
load("//xla/mlir/backends/experimental:build_config.bzl", "if_experimental", "if_not_experimental")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# cc_library(
#     name = "compiler",
#     srcs = if_experimental(["compiler.cc"]),
#     hdrs = if_experimental(["compiler.h"]),
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         "@com_google_absl//absl/base",
#         "//xla:status",
#         "//xla:util",
#         "@tsl//tsl/platform",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#     ] + tf_platform_deps(
#         "compiler",
#         platform_dir = "//xla/service/gpu/experimental/",
#     ) + if_experimental([
#         "//third_party/iree/compiler/bindings/c:headers",
#         "//third_party/iree/compiler/bindings/c:loader",
#         "//third_party/iree/llvm-external-projects/iree-dialects:IREEInputDialect",
#     ]),
# )
#
# cc_library(
#     name = "executable",
#     srcs = if_experimental(["executable.cc"]),
#     hdrs = ["executable.h"],
#     compatible_with = [],
#     defines = if_not_experimental(["XLA_DISABLE_EXPERIMENTAL_RUNTIME=1"]),
#     deps = [
#         ":compiler",
#         ":module",
#         ":vm",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/strings",
#         "//xla:status",
#         "//xla:statusor",
#         "//xla:util",
#         "//xla:xla_proto_cc",
#         "//xla/service:buffer_assignment",
#         "//xla/service:executable",
#         "//xla/service/gpu:buffer_allocations",
#         "//xla/service/llvm_ir:llvm_util",
#         "@llvm-project//mlir:IR",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/base",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/hal/drivers/cuda",
#         "//third_party/iree/runtime/src/iree/modules/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm/bytecode:module",
#     ]),
# )
#
# cc_library(
#     name = "gemm",
#     srcs = if_experimental(["gemm.cc"]),
#     hdrs = if_experimental(["gemm.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":vm",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "//xla:status",
#         "//xla:statusor",
#         "//xla:xla_data_proto_cc",
#         "//xla/service:executable",
#         "//xla/service/gpu:matmul_utils",
#         "@tsl//tsl/profiler/lib:scoped_annotation",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "kernel",
#     srcs = if_experimental(["kernel.cc"]),
#     hdrs = if_experimental(["kernel.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":vm",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "//xla:status",
#         "//xla:statusor",
#         "//xla/service:executable",
#         "//xla/service/gpu:launch_dimensions",
#         "//xla/service/gpu:stream_executor_util",
#         "//xla/stream_executor:device_memory",
#         "@tsl//tsl/profiler/lib:scoped_annotation",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "memcpy",
#     srcs = if_experimental(["memcpy.cc"]),
#     hdrs = if_experimental(["memcpy.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":vm",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "//xla:status",
#         "//xla:statusor",
#         "//xla/service:executable",
#         "//xla/service/gpu:launch_dimensions",
#         "//xla/service/gpu:stream_executor_util",
#         "//xla/stream_executor:device_memory",
#         "@tsl//tsl/profiler/lib:scoped_annotation",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "module",
#     srcs = if_experimental(["module.cc"]),
#     hdrs = if_experimental(["module.h"]),
#     compatible_with = [],
#     deps = [
#         ":gemm",
#         ":hal",
#         ":kernel",
#         ":memcpy",
#         ":vm",
#         "@com_google_absl//absl/log",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/base",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm:cc",
#     ]),
# )
#
# cc_library(
#     name = "hal",
#     srcs = if_experimental(["hal.cc"]),
#     hdrs = if_experimental(["hal.h"]),
#     compatible_with = [],
#     deps = [
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/types:span",
#         "//xla:shape_util",
#         "//xla/stream_executor:device_memory",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/hal",
#     ]),
# )
#
# cc_library(
#     name = "vm",
#     srcs = if_experimental(["vm.cc"]),
#     hdrs = if_experimental(["vm.h"]),
#     compatible_with = [],
#     deps = [
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/strings:str_format",
#     ] + if_experimental([
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm:cc",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#     ]),
# )
#
# copybara:uncomment_end_and_comment_begin
cc_library(
    name = "executable",
    hdrs = ["executable.h"],
    defines = ["XLA_DISABLE_EXPERIMENTAL_RUNTIME=1"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla/service:executable",
        "//xla/service/gpu:buffer_allocations",
        "@llvm-project//mlir:IR",
    ],
)
# copybara:comment_end
