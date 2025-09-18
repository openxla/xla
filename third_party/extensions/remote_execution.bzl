"""Module extension for remote_execution."""

load("@xla//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

remote_execution_extension = module_extension(
    implementation = lambda mctx: remote_execution_configure(name = "local_config_remote_execution"),
)
