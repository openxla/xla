load("@tsl//platform:rules_cc.bzl", "cc_library")
load("@tsl//:tsl.bzl", "if_windows")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("@tsl//platform:build_config_root.bzl", "if_static")

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
    srcs = glob(
        [
            "**/*.cc",
        ],
        exclude = [
            "**/*test*",
        ],
    ),
    hdrs = glob(["**/*.h"]),
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xla/stream_executor/platform",
        "@tsl//platform:env",
        "@tsl//platform:stacktrace",
        "@tsl//platform:status",
        "@tsl//platform:statusor",
        "@tsl//protobuf:error_codes_proto_impl_cc",
    ] + if_static([
        "@tsl//platform:env_impl",
    ]),
)
