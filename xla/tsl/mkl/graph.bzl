"""Starlark macros for oneDNN Graph API.

Contains build rules that builds with empty srcs, hdrs, and deps if not build with Graph API.
These rules have to be outside of mkl/build_defs.bzl, otherwise we would have cyclic dependency
(xla.bzl depends on tsl which depends on mkl/build_defs.bzl).
"""

load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl/mkl:build_defs.bzl", "if_graph_api")

def onednn_graph_cc_library(srcs = [], hdrs = [], deps = [], **kwargs):
    """cc_library rule that has empty src, hdrs and deps if not building with Graph API."""
    native.cc_library(
        srcs = if_graph_api(srcs),
        hdrs = if_graph_api(hdrs),
        deps = if_graph_api(deps),
        **kwargs
    )

def onednn_graph_cc_test(srcs = [], deps = [], **kwargs):
    """xla_cc_test rule that has empty src and deps if not building with Graph API."""
    xla_cc_test(
        srcs = if_graph_api(srcs),
        deps = if_graph_api(deps) + ["@com_google_googletest//:gtest_main"],
        **kwargs
    )
