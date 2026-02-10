""" Provides the CCCL headers for CUDA. """

# CCCL version 3.2.0
# https://github.com/NVIDIA/cccl/releases/tag/v3.2.0
_CCCL_GITHUB_ARCHIVE = {
    "full_path": "https://github.com/NVIDIA/cccl/releases/download/v3.2.0/cccl-src-v3.2.0.tar.gz",
    "sha256": "b5cd66e240201f5a06af2a75eaffdf05a6c63829edada33ff569ada0037f8086",
    "strip_prefix": "cccl-src-v3.2.0",
}

CCCL_DIST_DICT = {
    "cuda_cccl": {
        "linux-x86_64": _CCCL_GITHUB_ARCHIVE,
        "linux-sbsa": _CCCL_GITHUB_ARCHIVE,
    },
}

CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES = {
    "cuda_cccl": {
        "repo_name": "cuda_cccl",
        "version_to_template": {
            "any": "@rules_ml_toolchain//gpu/cuda/build_templates:cuda_cccl_github.BUILD.tpl",
        },
        "local": {
            "local_path_env_var": "LOCAL_CCCL_PATH",
            "source_dirs": ["thrust", "libcudacxx", "cub"],
            "version_to_template": {
                "any": "@rules_ml_toolchain//gpu/cuda/build_templates:cuda_cccl_github.BUILD.tpl",
            },
        },
    },
}
