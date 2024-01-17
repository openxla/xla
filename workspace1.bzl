"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
load("@rules_python//python:repositories.bzl", "py_repositories")
load("@tsl//:workspace1.bzl", "tsl_workspace1")

# buildifier: disable=unnamed-macro
def workspace():
    """Loads a set of TensorFlow dependencies in a WORKSPACE file."""
    tsl_workspace1()

    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_pkg_dependencies()

    closure_repositories()
    py_repositories()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "02e4f3744f1ce3f6e711e261fd322916ddd18cccd38026352f7a4c0351dbda19",
        strip_prefix = "bazel-toolchains-5.1.2",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/refs/tags/v5.1.2.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/refs/tags/v5.1.2.tar.gz",
        ],
    )

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace1 = workspace
