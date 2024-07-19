# Optimized sharding annotations

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google", "if_libtpu", "internal_visibility", "tsl_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
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
    name = "auto_parallel",
    srcs = ["auto_parallel.cc"],
    hdrs = ["auto_parallel.h"],
    visibility = ["//visibility:public"],
    deps = [
        "module_cost_evaluator",
        "resharding_cost_evaluator",
        "//xla/service:hlo_pass", 
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:sharding_propagation",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "sharding_strategy_solver",
    srcs = ["sharding_strategy_solver.cc"],
    hdrs = ["sharding_strategy_solver.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/hlo/ir:hlo"
    ]
)

cc_library(
    name = "module_cost_evaluator",
    srcs = ["module_cost_evaluator.cc"],
    hdrs = ["module_cost_evaluator.h"],
    visibility = ["//visibility:private"],
    deps = [
        "shape_utils",
        "//xla/hlo/ir:hlo"
    ]
)

cc_library(
    name = "resharding_cost_evaluator",
    srcs = ["resharding_cost_evaluator.cc"],
    hdrs = ["resharding_cost_evaluator.h"],
    visibility = ["//visibility:private"],
    deps = [
        "shape_utils",
        "//xla/hlo/ir:hlo"
    ]
)

cc_library(
    name = "shape_utils",
    srcs = ["shape_utils.cc"],
    hdrs = ["shape_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla:shape_util"
    ]
)