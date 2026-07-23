"""Provides the repository macro to import Triton."""

load("//third_party/triton:common/series.bzl", "common_patch_list")
load("//third_party/triton:intel_xpu/workspace.bzl", "triton_archive")
load("//third_party/triton:oss_only/series.bzl", "oss_only_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "72259b1cc3c543c361dcd185a6ff89662e8ed52f"
    TRITON_SHA256 = "35744577b837c66cf934b3b1d31b1496e3c205c0fb431b8bdcc76f4c0245312c"
    triton_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        triton_commit = TRITON_COMMIT,
        patch_file = common_patch_list + oss_only_patch_list,
    )
