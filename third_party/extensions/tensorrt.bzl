"""Module extension for tensorrt."""

load("@xla//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")

tensorrt_extension = module_extension(
    implementation = lambda mctx: tensorrt_configure(name = "local_config_tensorrt"),
)
