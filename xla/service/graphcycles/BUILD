load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = [
        "//third_party/tensorflow/compiler/tf2xla:internal",
        "//xla:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "graphcycles",
    srcs = ["graphcycles.cc"],
    hdrs = ["graphcycles.h"],
    deps = [
        ":ordered_set",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ordered_set",
    hdrs = ["ordered_set.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "graphcycles_test",
    srcs = ["graphcycles_test.cc"],
    deps = [
        ":graphcycles",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
        "@tsl//tsl/platform:test_main",
    ],
)

tf_cc_test(
    name = "ordered_set_test",
    srcs = ["ordered_set_test.cc"],
    deps = [
        ":ordered_set",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
        "@tsl//tsl/platform:test_main",
    ],
)
