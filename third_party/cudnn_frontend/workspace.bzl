"""cuDNN frontend is a C++ API for cuDNN."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "54fdc7347326b2e9938e6883a437da43223c5b95ef26e641da8fc8b981936bfc",
        strip_prefix = "cudnn-frontend-1.16.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.16.0.zip"),
    )
