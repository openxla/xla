"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "15c92386bd17e8118688ab93533ad73921860dcf0be463745843b8dcb76c30fa",
        strip_prefix = "XNNPACK-7ef5fdca247dfc94e9ac8a93fac74e558565efac",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/7ef5fdca247dfc94e9ac8a93fac74e558565efac.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
