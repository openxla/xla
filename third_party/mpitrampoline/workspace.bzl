"""Provides the repository macro to import mpitrampoline."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports mpitrampoline."""

    mpitrampoline_COMMIT = "25efb0f7a4cd00ed82bafb8b1a6285fc50d297ed"
    mpitrampoline_SHA256 = "5a36656205c472bdb639bffebb0f014523b32dda0c2cbedd9ce7abfc9e879e84"

    tf_http_archive(
        name = "mpitrampoline",
        sha256 = mpitrampoline_SHA256,
        strip_prefix = "MPItrampoline-{commit}".format(commit = mpitrampoline_COMMIT),
        urls = tf_mirror_urls("https://github.com/eschnett/mpitrampoline/archive/{commit}.tar.gz".format(commit = mpitrampoline_COMMIT)),
        patch_file = ["//third_party/mpitrampoline:gen.patch"],
        build_file = "//third_party/mpitrampoline:mpitrampoline.BUILD",
    )
