"""MORI - ROCm Modular RDMA Interface.

Hermetic Bazel build of @roc_mori from a GitHub source tarball, with all
BUILD files supplied as overlays from this directory (Option 4 — Bazel
label list for hermetic overlays).

To update to a new commit:
  1. Change _MORI_COMMIT to the new full SHA.
  2. Clear _MORI_SHA256 (set to "").
  3. Run any bazel build that touches @roc_mori — Bazel will print the
     actual sha256 in the error output; paste it back here.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Branch: pemeliya/multi_threading_support
_MORI_COMMIT = "db99d38d73f46f3123f2691887d3338c72d607af"
_MORI_SHA256 = ""

def xla_repo():
    """Registers @roc_mori, fetched from GitHub and overlaid with our BUILD files."""
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
