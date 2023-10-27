load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "api_headers",
    srcs = ["api.h"],
)

filegroup(
    name = "c_api_headers",
    srcs = ["c_api.h"],
)

cc_library(
    name = "api",
    hdrs = [":api_headers"],
    visibility = ["//visibility:private"],
    deps = [":c_api"],
)

cc_library(
    name = "c_api",
    hdrs = ["c_api.h"],
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    deps = [":c_api"],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    deps = [
        ":api",
        ":c_api",
    ],
)
