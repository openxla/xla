"""Module extension for local clang."""

load("@rules_ml_toolchain//cc/llvms/local:local_clang_configure.bzl", "local_clang_configure")

local_clang_configure_ext = module_extension(
    implementation = lambda mctx: local_clang_configure(name = "local_config_clang"),
)
