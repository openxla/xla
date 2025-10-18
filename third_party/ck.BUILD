package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "ck_header_files",
    srcs = glob([
        "include/ck_tile/**",
    ]),
)


cc_library(
    name = "ck",
    hdrs = [
        ":ck_header_files",
    ],
    includes = [
        "include",
    ],
)
