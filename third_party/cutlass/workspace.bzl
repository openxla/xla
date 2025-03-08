"""Provides the repository macro to import CUTLASS."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    CUTLASS_COMMIT = "afa7b7241aabe598b725c65480bd9fa71121732c"
    CUTLASS_SHA256 = "84cf3fcc47c440a8dde016eb458f8d6b93b3335d9c3a7a16f388333823f1eae0"

    tf_http_archive(
        name = "cutlass",
        sha256 = CUTLASS_SHA256,
        strip_prefix = "cutlass-{commit}".format(commit = CUTLASS_COMMIT),
        urls = tf_mirror_urls("https://github.com/chsigg/cutlass/archive/{commit}.tar.gz".format(commit = CUTLASS_COMMIT)),
        build_file = "//third_party/cutlass:cutlass.BUILD",
        patch_file = [
            "//third_party/cutlass:set_slice3x3-set_slice_3x3-1784.patch",
        ],
    )
