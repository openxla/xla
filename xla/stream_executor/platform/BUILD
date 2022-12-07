load("@tsl//platform:rules_cc.bzl", "cc_library")
load("@tsl//platform:build_config.bzl", "tf_stream_executor_deps")
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
    name = "platform",
    textual_hdrs = [
        "initialize.h",
        "logging.h",
        "platform.h",
        "port.h",
    ],
    deps = [
        "@tsl//platform:macros",
        "@tsl//platform:types",
        "@tsl//platform:logging",
    ] + tf_stream_executor_deps("platform", "//xla/stream_executor/platform/"),
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
    ] + tf_stream_executor_deps("dso_loader", "//xla/stream_executor/platform/"),
)
