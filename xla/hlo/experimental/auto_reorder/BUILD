load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.bzl",  "xla_cc_test","xla_cc_binary")
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages= [
        # "//xla/hlo/experimental/auto_reorder",
        "//xla/service/gpu/...",
        "//xla/hlo/utils/...",
    ],
    
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)
cc_library(
    name = "auto_reorder_solver",
    srcs = ["auto_reorder_solver.cc",
    ],
    hdrs = [
        "auto_reorder_solver.h",
    ],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/hlo/utils:common_ortools_deps",
        "@tsl//tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/platform:types",
    ]
)
# All header files that are used in the build must be declared in
#   the hdrs or srcs of cc_* rules.
#   This is enforced.

cc_library(
    name = "auto_reorder",
    srcs = [
        "auto_reorder.cc",
    ],
    hdrs = [
        "auto_reorder.h",
        "auto_reorder_solver.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/service/gpu/model:analytical_latency_estimator",
        "//xla/service:backend",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:statusor",
        ":auto_reorder_solver"
    ],
)

xla_cc_test(
    name = "auto_reorder_test",
    srcs = ["auto_reorder_test.cc"],
    deps = [
        ":auto_reorder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:statusor",
    ],
)