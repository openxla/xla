"""Module extension for rocm."""

load("@xla//third_party/gpus:rocm_configure.bzl", "rocm_configure")

rocm_extension = module_extension(
    implementation = lambda mctx: rocm_configure(name = "local_config_rocm"),
)
