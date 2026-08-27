load("//xla:xla.default.bzl", "xla_cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "add_zero_eliminator",
    srcs = ["add_zero_eliminator.cc"],
    hdrs = ["add_zero_eliminator.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "first_hlo_pass_test",
    srcs = ["add_zero_eliminator_test.cc"],
    deps = [
        ":add_zero_eliminator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)
