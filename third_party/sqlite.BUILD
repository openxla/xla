load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["unencumbered"])  # Public Domain

cc_library(
    name = "sqlite3",
    srcs = ["sqlite3.c"],
    hdrs = [
        "sqlite3.h",
        "sqlite3ext.h",
    ],
    defines = [
        "SQLITE_OMIT_DEPRECATED",
    ],
    visibility = ["//visibility:public"],
)
