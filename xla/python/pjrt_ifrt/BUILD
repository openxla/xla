load("//xla:xla.bzl", "xla_cc_test")

package_group(
    name = "friends",
    includes = [
        "//xla/python:friends",
    ],
    packages = [
        "//xla/python/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "//xla/python/pjrt_ifrt/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        ":internal",
    ],
)

exports_files([
    "BUILD",
])

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
cc_library(
    name = "xla_ifrt",
    srcs = [
        "xla_compiler.cc",
    ],
    hdrs = [
        "xla_compiler.h",
    ],
    deps = [
        "//xla/python/ifrt",
        "@llvm-project//llvm:Support",
    ],
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
cc_library(
    name = "xla_executable_impl_test_lib",
    testonly = 1,
    srcs = ["xla_executable_impl_test_lib.cc"],
    deps = [
        ":xla_ifrt",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:statusor",
    ],
    alwayslink = 1,
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
xla_cc_test(
    name = "xla_executable_test_no_impl",
    srcs = [],
    deps = [
        ":xla_executable_impl_test_lib",
        "//xla/python/ifrt:no_impl_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "pjrt_ifrt",
    srcs = [
        "pjrt_array.cc",
        "pjrt_client.cc",
        "pjrt_compiler.cc",
        "pjrt_executable.cc",
        "pjrt_host_callback.cc",
        "pjrt_tuple.cc",
    ],
    hdrs = [
        "pjrt_array.h",
        "pjrt_client.h",
        "pjrt_compiler.h",
        "pjrt_executable.h",
        "pjrt_host_callback.h",
        "pjrt_tuple.h",
    ],
    deps = [
        ":xla_ifrt",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_computation",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:utils",
        "//xla/python/ifrt",
        "//xla/service:hlo_proto_cc",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@tf_runtime//:ref_count",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tfrt_cpu_client_test_lib",
    testonly = 1,
    srcs = ["tfrt_cpu_client_test_lib.cc"],
    deps = [
        ":pjrt_ifrt",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/python/ifrt:test_util",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "pjrt_array_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["pjrt_array_impl_test_tfrt_cpu.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:array_impl_test_lib",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "pjrt_client_impl_test_tfrt_cpu",
    size = "small",
    srcs = [],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:client_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "pjrt_executable_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["pjrt_executable_impl_test_tfrt_cpu.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        ":xla_executable_impl_test_lib",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "pjrt_tuple_impl_test_tfrt_cpu",
    size = "small",
    srcs = [],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:tuple_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
