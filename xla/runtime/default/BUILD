load("@tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/runtime:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "async_values_cache",
    hdrs = ["async_values_cache.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/concurrency:async_value",
    ],
)

cc_library(
    name = "memory_mapper",
    hdrs = ["memory_mapper.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [],
)
