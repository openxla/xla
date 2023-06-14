load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "sharding_param_to_op_sharding",
    srcs = ["sharding_param_to_op_sharding.cc"],
    hdrs = ["sharding_param_to_op_sharding.h"],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/python/ifrt/ir",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "sharding_param_to_op_sharding_test",
    srcs = ["sharding_param_to_op_sharding_test.cc"],
    deps = [
        ":sharding_param_to_op_sharding",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt/ir",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:statusor",
    ],
)
