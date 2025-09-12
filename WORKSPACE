# buildifier: disable=load-on-top
workspace(name = "xla")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Initialize toolchains for ML projects.
#
# A hermetic build system is designed to produce completely reproducible builds for C++.
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "e7e44c4e349a1c1f31398bd2257c51432e73ea0e7e24cce67090b68b0b50007e",
    strip_prefix = "rules_ml_toolchain-55dcd0a52c7e0f9eec9927a32512229c09ac3b3e",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/55dcd0a52c7e0f9eec9927a32512229c09ac3b3e.tar.gz",
    ],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")

# Initialize the XLA repository and all dependencies.
#
# The cascade of load() statements and xla_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

load(":workspace4.bzl", "xla_workspace4")

xla_workspace4()

load(":workspace3.bzl", "xla_workspace3")

xla_workspace3()

# Initialize hermetic Python
load("//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
    },
)

load("//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load(":workspace2.bzl", "xla_workspace2")

xla_workspace2()

load(":workspace1.bzl", "xla_workspace1")

xla_workspace1()

load(":workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)
http_archive(
    name = "sycl_hermetic",
    urls = ["https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/intel-oneapi-base-toolkit-2025.1.3.7.tar.gz"],
    sha256 = "2213104bd122336551aa144512e7ab99e4a84220e77980b5f346edc14ebd458a",
    build_file_content = "package(default_visibility=[\"//visibility:public\"])\n",
)

http_archive(
    name = "level_zero_redist",
    urls = ["https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/level-zero-1.21.10.tar.gz"],
    sha256 = "e0ff1c6cb9b551019579a2dd35c3a611240c1b60918c75345faf9514142b9c34",
    build_file_content = "package(default_visibility=[\"//visibility:public\"])\n",
)

http_archive(
    name = "ze_loader_redist",
    urls = ["https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/ze_loader_libs.tar.gz"],
    sha256 = "71cbfd8ac59e1231f013e827ea8efe6cf5da36fad771da2e75e202423bd6b82e",
    build_file_content = "package(default_visibility=[\"//visibility:public\"])\n",
)

load("//third_party/gpus:sycl_configure.bzl", "sycl_configure")

sycl_configure(name = "local_config_sycl")
