"""
Repository rule to get python interpreter and python version to be used
in pip_parse.

For hermetic, we set py_interpreter_target.
For non hermetic, we set py_interpreter.
"""

load(
    "//third_party/remote_config:common.bzl",
    "execute",
    "get_python_bin",
)

def _get_python_version(repository_ctx, python_bin):
    """Gets the python minor version."""
    result = execute(
        repository_ctx,
        [
            python_bin,
            "-c",
            'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))',
        ],
        error_msg = "Problem getting python version.",
        error_details = ("Is the Python binary path set up right?"),
    )
    return result.stdout.splitlines()[0]

def _hermetic_impl(repository_ctx):
    if not repository_ctx.attr.version:
        fail("When using hermetic python, specify the version in tf_workspace3")
    repository_ctx.file("BUILD", "")
    repository_ctx.file(
        "py_deps.bzl",
        """
load("@python//:defs.bzl", "interpreter")
py_interpreter_target = interpreter
py_version = "%s"
py_interpreter = ""
hermetic_python = True
""" % repository_ctx.attr.version,
    )

def _non_hermetic_impl(repository_ctx):
    py_bin = get_python_bin(repository_ctx)
    version = _get_python_version(repository_ctx, py_bin)
    repository_ctx.file("BUILD", "")
    repository_ctx.file(
        "py_deps.bzl",
        """
py_interpreter_target = None
py_version = "%s"
py_interpreter = "%s"
hermetic_python = False
""" % (version, py_bin),
    )

def _python_location_repository_impl(repository_ctx):
    if repository_ctx.attr.hermetic:
        _hermetic_impl(repository_ctx)
    else:
        _non_hermetic_impl(repository_ctx)

python_deps_repository = repository_rule(
    implementation = _python_location_repository_impl,
    attrs = {
        "hermetic": attr.bool(),
        "version": attr.string(),
    },
)
