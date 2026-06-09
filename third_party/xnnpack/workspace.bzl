"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "84b99b4806c2cca4ee31b0e2212664deaa65d3c71d369da19951a972ee6562e4",
        strip_prefix = "XNNPACK-f5d613307ef88ef784395640f4d48ac1a1e42b21",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/f5d613307ef88ef784395640f4d48ac1a1e42b21.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
