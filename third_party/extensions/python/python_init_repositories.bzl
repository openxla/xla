load("//third_party/py:python_repo.bzl", "python_repository")

def _python_init_extension_impl(module_ctx):
    """
    Implementation of the python_init_extension.

    This extension is designed to be a "singleton," just like the
    original WORKSPACE macro. It should only be used once, typically
    in the root MODULE.bazel file. It finds the first 'init' tag
    and uses it to configure the 'python_version_repo'.
    """

    # Collect all modules with init tag.
    modules = [mod for mod in module_ctx.modules if mod.tags.init]

    # If no module tagged this extension, there's nothing to do.
    if not modules:
        return

    # This macro was a singleton, so we only process the first tag.
    # We'll issue a warning if more than one module tries to configure it.
    if len(modules) > 1:
        print(
            "WARNING: 'python_init_extension' was configured by multiple " +
            "modules. Respecting the one from %s@%s because it's the first one in BFS order from root." % (modules[0].name, modules[0].version),
        )

    init = modules[0].tags.init[0]

    # Call the repository rule, just as the original macro did.
    # The name is hardcoded, as in the original.
    python_repository(
        name = "python_version_repo",
        requirements_versions = init.requirements.keys(),
        requirements_locks = init.requirements.values(),
        local_wheel_workspaces = init.local_wheel_workspaces,
        # Make sure to pass None instead of "", otherwise the default values in python_repository won't work.
        local_wheel_dist_folder = init.local_wheel_dist_folder if init.local_wheel_dist_folder else None,
        default_python_version = init.default_python_version if init.default_python_version else None,
        local_wheel_inclusion_list = init.local_wheel_inclusion_list,
        local_wheel_exclusion_list = init.local_wheel_exclusion_list,
    )

# Defines the attributes that can be passed to the extension
# from the MODULE.bazel file.
_python_init_tag = tag_class(attrs = {
    "requirements": attr.string_dict(
        doc = "A dictionary mapping requirements file labels to their lockfile labels.",
        default = {},
    ),
    "local_wheel_workspaces": attr.string_list(
        doc = "List of local workspaces containing wheels.",
        default = [],
    ),
    "local_wheel_dist_folder": attr.string(
        doc = "Path to a local 'dist' folder for wheels.",
        mandatory=False,
    ),
    "default_python_version": attr.string(
        doc = "Default Python version to use (e.g., 'PY3_11').",
        mandatory=False,
    ),
    "local_wheel_inclusion_list": attr.string_list(
        doc = "Glob patterns for wheels to include.",
        default = ["*"],
    ),
    "local_wheel_exclusion_list": attr.string_list(
        doc = "Glob patterns for wheels to exclude.",
        default = [],
    ),
})

# The public-facing module extension that users will load.
python_init_extension = module_extension(
    implementation = _python_init_extension_impl,
    tag_classes = {"init": _python_init_tag},
)
