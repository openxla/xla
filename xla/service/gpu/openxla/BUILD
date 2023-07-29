load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@tsl//tsl/platform:build_config.bzl", "tf_platform_deps")
load("//xla/mlir/backends/openxla:build_config.bzl", "if_not_openxla", "if_openxla")

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
    name = "compiler",
    srcs = if_openxla(["compiler.cc"]),
    hdrs = if_openxla(["compiler.h"]),
    # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
    # compatible with `non_prod` constraint.
    compatible_with = [],
    deps = [
        "//xla:status",
        "//xla:util",
        "@com_google_absl//absl/base",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform",
    ] + tf_platform_deps(
        "compiler",
        platform_dir = "//xla/service/gpu/openxla/",
    ) + if_openxla([
        "@iree_core//compiler/bindings/c:headers",
        "@iree_core//compiler/bindings/c:loader",
        "@iree_core//llvm-external-projects/iree-dialects:IREEInputDialect",
    ]),
)

cc_library(
    name = "executable",
    srcs = if_openxla(["executable.cc"]),
    hdrs = ["executable.h"],
    compatible_with = [],
    defines = if_not_openxla(["XLA_DISABLE_OPENXLA_RUNTIME=1"]),
    deps = [
        ":compiler",
        ":module",
        ":vm",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
    ] + if_openxla([
        "@iree_core//runtime/src/iree/base",
        "@iree_core//runtime/src/iree/hal",
        "@iree_core//runtime/src/iree/hal/drivers/cuda",
        "@iree_core//runtime/src/iree/modules/hal",
        "@iree_core//runtime/src/iree/modules/hal:types",
        "@iree_core//runtime/src/iree/vm",
        "@iree_core//runtime/src/iree/vm/bytecode:module",
    ]),
)

cc_library(
    name = "gemm",
    srcs = if_openxla(["gemm.cc"]),
    hdrs = if_openxla(["gemm.h"]),
    compatible_with = [],
    deps = [
        ":hal",
        ":vm",
        "//xla:status",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/service:executable",
        "//xla/service/gpu:matmul_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@tsl//tsl/profiler/lib:scoped_annotation",
    ] + if_openxla([
        "@iree_core//runtime/src/iree/hal",
        "@iree_core//runtime/src/iree/vm",
    ]),
)

cc_library(
    name = "kernel",
    srcs = if_openxla(["kernel.cc"]),
    hdrs = if_openxla(["kernel.h"]),
    compatible_with = [],
    deps = [
        ":hal",
        ":vm",
        "//xla:status",
        "//xla:statusor",
        "//xla/service:executable",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@tsl//tsl/profiler/lib:scoped_annotation",
    ] + if_openxla([
        "@iree_core//runtime/src/iree/hal",
        "@iree_core//runtime/src/iree/vm",
    ]),
)

cc_library(
    name = "module",
    srcs = if_openxla(["module.cc"]),
    hdrs = if_openxla(["module.h"]),
    compatible_with = [],
    deps = [
        ":gemm",
        ":hal",
        ":kernel",
        ":vm",
        "@com_google_absl//absl/log",
    ] + if_openxla([
        "@iree_core//runtime/src/iree/base",
        "@iree_core//runtime/src/iree/hal",
        "@iree_core//runtime/src/iree/modules/hal:types",
        "@iree_core//runtime/src/iree/vm",
        "@iree_core//runtime/src/iree/vm:cc",
    ]),
)

cc_library(
    name = "hal",
    srcs = if_openxla(["hal.cc"]),
    hdrs = if_openxla(["hal.h"]),
    compatible_with = [],
    deps = [
        "//xla:shape_util",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ] + if_openxla([
        "@iree_core//runtime/src/iree/hal",
    ]),
)

cc_library(
    name = "vm",
    srcs = if_openxla(["vm.cc"]),
    hdrs = if_openxla(["vm.h"]),
    compatible_with = [],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
    ] + if_openxla([
        "@iree_core//runtime/src/iree/vm",
        "@iree_core//runtime/src/iree/vm:cc",
        "@iree_core//runtime/src/iree/hal",
        "@iree_core//runtime/src/iree/modules/hal:types",
    ]),
)
