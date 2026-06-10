"""Repository helpers for selecting the Triton source archive."""

load("//third_party:repo.bzl", "tf_mirror_urls")

XPU_TRITON_COMMIT = "0ba44beb6746a3f4935198bfcc353ee0abe0b5ac"
XPU_TRITON_SHA256 = "c7ac24d81221f33c57e02246a342bc26dc00a11f12217137efa6c0d220cb01f1"

def _use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

def _triton_archive_impl(repository_ctx):
    patch_files = repository_ctx.attr.patch_file
    sha256 = repository_ctx.attr.triton_sha256
    commit = repository_ctx.attr.triton_commit
    strip_prefix = "triton-" + commit
    url = tf_mirror_urls("https://github.com/triton-lang/triton/archive/{}.tar.gz".format(commit))

    if _use_xpu_triton(repository_ctx):
        commit = XPU_TRITON_COMMIT
        sha256 = XPU_TRITON_SHA256
        patch_files = [
            "//third_party/triton:oss_only/build_files.patch",
            "//third_party/triton:intel_xpu/intel_build.patch",
        ]
        strip_prefix = "intel-xpu-backend-for-triton-" + commit
        url = tf_mirror_urls("https://github.com/intel/intel-xpu-backend-for-triton/archive/{}.tar.gz".format(commit))

    for patch_file in patch_files:
        repository_ctx.path(Label(patch_file))

    repository_ctx.download_and_extract(
        url = url,
        sha256 = sha256,
        stripPrefix = strip_prefix,
    )
    for patch_file in patch_files:
        repository_ctx.patch(repository_ctx.path(Label(patch_file)), strip = 1)

_triton_archive = repository_rule(
    implementation = _triton_archive_impl,
    attrs = {
        "patch_file": attr.string_list(),
        "triton_commit": attr.string(mandatory = True),
        "triton_sha256": attr.string(mandatory = True),
    },
    environ = ["ENABLE_INTEL_XPU_TRITON"],
)

def triton_archive(name, sha256, triton_commit, patch_file):
    _triton_archive(
        name = name,
        triton_sha256 = sha256,
        patch_file = patch_file,
        triton_commit = triton_commit,
    )
