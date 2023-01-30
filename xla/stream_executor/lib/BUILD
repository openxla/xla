load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "lib",
)
