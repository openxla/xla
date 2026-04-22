def get_canonical_repo_name(apparent_repo_name):
    """Returns the canonical repo name for the given apparent repo name seen by the module this bzl file belongs to."""
    if not apparent_repo_name.startswith("@"):
        apparent_repo_name = "@" + apparent_repo_name
    return Label(apparent_repo_name).workspace_name
