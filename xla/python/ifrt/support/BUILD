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
        "//xla:xla_data_proto_cc",
        "//xla/python/ifrt/ir",
        "@llvm-project//llvm:Support",
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
        "//xla/python/ifrt/ir",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)
