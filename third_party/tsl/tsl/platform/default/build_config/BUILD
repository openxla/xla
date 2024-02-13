# Description:
# Platform-specific build configurations.

load("//tsl:tsl.bzl", "set_external_visibility", "tsl_copts")
load("//tsl/platform:rules_cc.bzl", "cc_library")

package(default_visibility = set_external_visibility(["//tsl:internal"]))

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "tensorflow_platform_specific",
    copts = tsl_copts(),
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    deps = [],
)
