# Description:
#   Intel oneAPI DPL (Data Parallel C++ Library)

load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE.txt"])

cc_library(
    name = "onedpl",
    hdrs = glob(["include/oneapi/dpl/**"]),
    includes = ["include"],
)

cc_library(
    name = "libs",
    visibility = ["//visibility:public"],
    deps = [
        ":onedpl",
    ],
    alwayslink = True,
)
