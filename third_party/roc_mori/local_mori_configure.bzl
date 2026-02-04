"""Configuration for locally installed ROC MORI."""

def _mori_impl(repository_ctx):
    """Implementation of the local_mori_configure repository rule."""
    
    # Get the MORI installation path from environment variable
    mori_path = repository_ctx.os.environ.get("ROC_MORI_PATH", "/data/mori")
    mori_lib_path = repository_ctx.os.environ.get(
        "ROC_MORI_LIB_PATH", mori_path + "/python/mori"
    )

    # Create a simple BUILD file that exposes the local installation
    build_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mori_headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    strip_include_prefix = "include",
    include_prefix = "third_party",
    includes = ["include"],
)

# filegroup(
#    name = "libmori_shmem_dev",
#    srcs = ["lib/libmori_shmem.so"],
#    visibility = ["//visibility:public"],
# )

cc_import(
    name = "libmori_shmem",
    shared_library = "lib/libmori_shmem.so",
)

cc_import(
    name = "libmori_application",
    shared_library = "lib/libmori_application.so",
)

cc_library(
    name = "mori_libs",
    deps = [
        ":libmori_shmem",
        ":libmori_application",
    ],
    linkstatic = False,
)

cc_library(
    name = "roc_mori_config",
    hdrs = ["roc_mori_config.h"],
    include_prefix = "third_party",
)
"""
    repository_ctx.file("BUILD", build_content)
    
    # Create symlinks to the actual installation
    repository_ctx.symlink(mori_path + "/include", "include")
    repository_ctx.symlink(mori_lib_path, "lib")

    # Create a simple config header
    config_header = """
#ifndef THIRD_PARTY_ROCM_MORI_CONFIG_H_
#define THIRD_PARTY_ROCM_MORI_CONFIG_H_

constexpr static char XLA_ROCM_MORI_VERSION[] = "local";

#endif  // THIRD_PARTY_ROCM_MORI_CONFIG_H_
"""
    repository_ctx.file("roc_mori_config.h", config_header)

local_mori_configure = repository_rule(
    implementation = _mori_impl,
    environ = ["ROC_MORI_PATH", "ROC_MORI_LIB_PATH"],
    local = True,
)