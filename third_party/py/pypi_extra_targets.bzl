load("@rules_cc//cc:cc_library.bzl", "cc_library")

def extra_numpy_targets():
    cc_library(
        name = "numpy_headers_2",
        hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
        strip_include_prefix="site-packages/numpy/_core/include/",
    )
    cc_library(
        name = "numpy_headers_1",
        hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
        strip_include_prefix="site-packages/numpy/core/include/",
    )
    cc_library(
        name = "numpy_headers",
        deps = [":numpy_headers_2", ":numpy_headers_1"],
    )
