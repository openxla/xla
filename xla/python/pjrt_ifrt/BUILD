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
    default_visibility = [
        ":friends",
        ":internal",
    ],
    licenses = ["notice"],
)

exports_files([
    "BUILD",
])

cc_library(
    name = "pjrt_ifrt",
    srcs = [
        "pjrt_array.cc",
        "pjrt_client.cc",
        "pjrt_compiler.cc",
        "pjrt_executable.cc",
    ],
    hdrs = [
        "pjrt_array.h",
        "pjrt_client.h",
        "pjrt_compiler.h",
        "pjrt_executable.h",
    ],
    deps = [
        "//xla:literal",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_computation",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/python/ifrt",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
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
    srcs = [],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:array_impl_test_lib",
        "@com_google_googletest//:gtest_main",
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
        "//xla/python/ifrt:executable_impl_test_lib",
        "@com_google_googletest//:gtest",
    ],
)
