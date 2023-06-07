load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "if_static",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "interpreter_transfer_manager",
    srcs = ["interpreter_transfer_manager.cc"],
    hdrs = ["interpreter_transfer_manager.h"],
    deps = [
        "//xla/backends/interpreter:platform_id",
        "//xla/service:generic_transfer_manager",
        "//xla/service:transfer_manager",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":executable",
        ":platform_id",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:algebraic_simplifier",
        "//xla/service:batchnorm_expander",
        "//xla/service:cholesky_expander",
        "//xla/service:comparison_expander",
        "//xla/service:compiler",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_target_registry",
        "//xla/service:dynamic_index_splitter",
        "//xla/service:eigh_expander",
        "//xla/service:executable",
        "//xla/service:flatten_call_graph",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:layout_assignment",
        "//xla/service:map_inliner",
        "//xla/service:qr_expander",
        "//xla/service:reshape_mover",
        "//xla/service:topk_rewriter",
        "//xla/service:triangular_solve_expander",
        "//xla/service:while_loop_simplifier",
        "//xla/stream_executor",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "platform_id",
    srcs = ["platform_id.cc"],
    hdrs = ["platform_id.h"],
    deps = ["//xla/stream_executor"] + if_static(
        ["@com_google_protobuf//:protobuf"],
        ["@com_google_protobuf//:protobuf_headers"],
    ),
)

cc_library(
    name = "executable_base",
    srcs = ["executable_base.cc"],
    hdrs = ["executable_base.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:executable",
        "//xla/service:hlo_execution_profile",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:event",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    deps = [
        ":executable_base",
        ":executor",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_execution_profile",
        "//xla/service:hlo_module_config",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":executor",
        ":platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:status",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_stream",
        "//xla/stream_executor/host:host_timer",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)
