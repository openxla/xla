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
        "debug",
        "instruction_strategies",
        "sharding_strategy",
        "sharding_strategy_solver",
        "module_cost_evaluator",
        "resharding_cost_matrix",
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
    hdrs = ["sharding_strategy_solver.h", "fix_log.h"],
    visibility = ["//visibility:private"],
    deps = [
        "instruction_strategies",
        "sharding_strategy",
        "solver_builder",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:logging",
    ]
)

cc_library(
    name = "solver_builder",
    srcs = ["simple_solver_builder.cc", "complete_solver_builder.cc"],
    hdrs = [
        "solver_builder.h", 
        "simple_solver_builder.h", 
        "complete_solver_builder.h", 
        "fix_log.h"
    ],
    visibility = ["//visibility:private"],
    deps = [
        "instruction_strategies",
        "sharding_strategy",
        "variable_matrix",
        "//xla/hlo/ir:hlo",
        "@com_google_ortools//ortools/linear_solver",
        "@tsl//tsl/platform:logging",
    ]
)

cc_library(
    name = "instruction_strategies",
    srcs = ["instruction_strategies.cc"],
    hdrs = ["instruction_strategies.h"],
    visibility = ["//visibility:private"],
    deps = [
        "sharding_strategy",
        "module_cost_evaluator",
        "instruction_to_module",
        "sharding_enumeration",
        "sharding_strategy_evaluator",
        "resharding_cost_matrix",
        "//xla/hlo/ir:hlo",
    ]
)

cc_library(
    name = "sharding_strategy_evaluator",
    srcs = ["sharding_strategy_evaluator.cc"],
    hdrs = ["sharding_strategy_evaluator.h"],
    visibility = ["//visibility:private"],
    deps = [
        "device_mesh",
        "module_cost_evaluator",
        "sharding_strategy",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass", 
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:sharding_propagation",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
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
    name = "resharding_cost_matrix",
    srcs = ["resharding_cost_matrix.cc"],
    hdrs = ["resharding_cost_matrix.h"],
    visibility = ["//visibility:private"],
    deps = [
        "resharding_cost_evaluator",
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
    name = "sharding_enumeration",
    srcs = ["sharding_enumeration.cc"],
    hdrs = ["sharding_enumeration.h"],
    visibility = ["//visibility:private"],
    deps = [
        "device_mesh",
        "sharding_strategy",
        "//xla/hlo/ir:hlo",
    ]
)

cc_library(
    name = "sharding_strategy",
    srcs = ["sharding_strategy.cc"],
    hdrs = ["sharding_strategy.h"],
    visibility = ["//visibility:private"],
    deps = [
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

cc_library(
    name = "instruction_to_module",
    srcs = ["instruction_to_module.cc"],
    hdrs = ["instruction_to_module.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/hlo/ir:hlo"
    ]
)

cc_library(
    name = "device_mesh",
    srcs = ["device_mesh.cc"],
    hdrs = ["device_mesh.h"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "variable_matrix",
    srcs = ["variable_matrix.cc"],
    hdrs = ["variable_matrix.h", "fix_log.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_ortools//ortools/linear_solver",
        "@tsl//tsl/platform:logging",
    ]
)

cc_library(
    name = "debug",
    srcs = ["debug.cc"],
    hdrs = ["debug.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/hlo/ir:hlo"
    ]
)