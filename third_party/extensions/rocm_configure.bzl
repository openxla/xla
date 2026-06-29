"""Module extension for rocm."""

load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("@rules_ml_toolchain//gpu/rocm:hipcc_configure.bzl", "hipcc_configure")

def _rocm_configure_ext_impl(mctx):
    rocm_configure(
        name = "local_config_rocm",
    )
    hipcc_configure(
        name = "config_rocm_hipcc",
        rocm_dist = "@local_config_rocm//rocm:toolchain_data",
    )

rocm_configure_ext = module_extension(
    implementation = _rocm_configure_ext_impl,
)
