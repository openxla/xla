"""Module extension for python."""

load("@xla//third_party/py:python_repo.bzl", "python_repository")

def _python_extension_impl(mctx):
    python_repository(
        name = "python_version_repo",
        requirements_versions = ["3.11", "3.12"],
        requirements_locks = [
            "@xla//:requirements_lock_3_11.txt",
            "@xla//:requirements_lock_3_12.txt",
        ],
    )

python_extension = module_extension(
    implementation = _python_extension_impl,
)
