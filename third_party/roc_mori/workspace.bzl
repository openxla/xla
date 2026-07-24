"""MORI - ROCm Modular RDMA Interface.

Hermetic Bazel build of @roc_mori from a GitHub source tarball, with all
BUILD files supplied as overlays from this directory (Option 4 — Bazel
label list for hermetic overlays).

To update to a new commit:
  1. Change _MORI_COMMIT to the new full SHA.
  2. Run any bazel build that touches @roc_mori — Bazel will print the
     actual sha256 in the error output; paste it back here. (A stale or
     empty _MORI_SHA256 both cause Bazel to report the correct hash.)
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Vendored from mori:main on 7-July-2026
_MORI_COMMIT = "a0c52deed69e76b58c11def63c4299305a6303b4"
_MORI_SHA256 = "436b644927957b9de0f34e606b6da426d13b349a6f7665954dbc72ed6e3deea0"

def xla_repo():
    """Registers @roc_mori, fetched from GitHub and overlaid with our BUILD files."""
    if not _MORI_SHA256:
        fail(
            "_MORI_SHA256 is empty in third_party/roc_mori/workspace.bzl. " +
            "Run a build to have Bazel print the actual sha256 for commit " +
            "{}, then paste it into _MORI_SHA256.".format(_MORI_COMMIT),
        )
    tf_http_archive(
        name = "roc_mori",
        sha256 = _MORI_SHA256,
        strip_prefix = "mori-{}".format(_MORI_COMMIT),
        urls = tf_mirror_urls(
            "https://github.com/ROCm/mori/archive/{}.zip".format(_MORI_COMMIT),
        ),
        # Top-level BUILD (exports headers-only targets and the config header).
        build_file = "//third_party/roc_mori:BUILD.roc_mori",
        # Additional BUILD overlays, one per MORI sub-library we have bazelified.
        # Adding a new sub-library (e.g. src/application) is a single extra
        # entry here plus a sibling BUILD.<name> file in this directory.
        link_files = {
            "//third_party/roc_mori:BUILD.src_shmem": "src/shmem/BUILD.bazel",
            "//third_party/roc_mori:BUILD.src_application": "src/application/BUILD.bazel",
            "//third_party/roc_mori:BUILD.src_collective": "src/collective/BUILD.bazel",
        },
    )
