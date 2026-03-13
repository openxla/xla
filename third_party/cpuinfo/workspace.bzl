"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "2d74c44c80d9419702ed83bb362ac764e71720093138ef06a34f73de829cce27",
        strip_prefix = "cpuinfo-f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0.zip"),
    )
