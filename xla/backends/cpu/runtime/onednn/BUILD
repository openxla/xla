load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    name = "onednn_interop",
    hdrs = ["onednn_interop.h"],
    deps = [
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@onednn//:mkl_dnn",
    ],
)

cc_library(
    name = "onednn_threadpool",
    hdrs = ["onednn_threadpool.h"],
    deps = [
        ":onednn_interop",
        "//xla/backends/cpu/runtime:parallel_loop_runner",
        "@onednn//:mkl_dnn",
    ],
)

xla_cc_test(
    name = "onednn_threadpool_test",
    srcs = select({
        "//third_party/intel_dnnl:use_custom_thread_pool": ["onednn_threadpool_test.cc"],
        "//conditions:default": [],
    }),
    deps = [
        ":onednn_interop",
        ":onednn_threadpool",
        "//xla/backends/cpu/runtime:parallel_loop_runner",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
        "@onednn//:mkl_dnn",
        "@pthreadpool",
    ],
)
