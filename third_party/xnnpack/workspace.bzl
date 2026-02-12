"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "bc7149f31eb4e48868f14aa915de8b1962ed8af208b95ed6b86293db6effa5ba",
        strip_prefix = "XNNPACK-33bda67b6cddba7e74e57fae3b2e18abe1a0213a",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/33bda67b6cddba7e74e57fae3b2e18abe1a0213a.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
