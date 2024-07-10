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