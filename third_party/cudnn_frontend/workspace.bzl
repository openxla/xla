"""cuDNN frontend is a C++ API for cuDNN."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "8630dd3fc96b351eed9683ca69fbe4c89397934aa6ad7b29677010020f286780",
        strip_prefix = "cudnn-frontend-1.18.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.18.0.zip"),
    )
