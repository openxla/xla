"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@tsl//:workspace1.bzl", "tsl_workspace1")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

# buildifier: disable=unnamed-macro
def workspace():
    """Loads a set of TensorFlow dependencies in a WORKSPACE file."""
    tsl_workspace1()

    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_pkg_dependencies()

    closure_repositories()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "95b02f13e25a67a1b616130097972707a78f5b248d077b4ebda353e356750778",
        strip_prefix = "bazel-toolchains-b18c44ec3e0f996443db5a319edd49535f89d1c9",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/b18c44ec3e0f996443db5a319edd49535f89d1c9.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/b18c44ec3e0f996443db5a319edd49535f89d1c9.tar.gz",
        ],
    )

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace1 = workspace
