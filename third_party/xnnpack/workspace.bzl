"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "f7007af8c6cee24976ee4c89ae7264909a70f503aaea3387b7d5e30f15f41e18",
        strip_prefix = "XNNPACK-e01449b8dbe859f4c0a773b8dce0c4b65449555b",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/e01449b8dbe859f4c0a773b8dce0c4b65449555b.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
