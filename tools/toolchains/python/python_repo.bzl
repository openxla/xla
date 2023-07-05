"""
Repository rule to set python version.
Can be set via build parameter "--repo_env=TF_PYTHON_VERSION=3.10"
Defaults to 3.9.
"""

def _python_repository_iimpl(repository_ctx):
    repository_ctx.file("BUILD", "")
    repository_ctx.file(
        "py_version.bzl",
        "HERMETIC_PYTHON_VERSION = \"%s\"" %
        repository_ctx.os.environ.get("TF_PYTHON_VERSION", "3.10"),
    )

python_repository = repository_rule(
    implementation = _python_repository_iimpl,
    environ = ["TF_PYTHON_VERSION"],
)
