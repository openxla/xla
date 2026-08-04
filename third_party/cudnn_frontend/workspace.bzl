"""cuDNN frontend is a C++ API for cuDNN."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    native.new_local_repository(
        name = "cudnn_frontend_archive",
        build_file = Label("//third_party:cudnn_frontend.BUILD"),
        path = "/opt/cudnn_frontend"
    )
