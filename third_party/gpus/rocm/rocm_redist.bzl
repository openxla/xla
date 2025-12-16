rocm_redist = {
    "rocm_7.10.0_gfx94X": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz",
                "sha256": "f53d58a26c8738f633a33582d650be76fe634122b693b3f4f0e7c9dd3f840623",
            },
        ],
        required_softlinks = [struct(target = "llvm/amdgcn", link = "amdgcn")],
        rocm_root = "",
    ),
    "rocm_7.10.0_gfx90X": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx90X-dcgpu-7.10.0.tar.gz",
                "sha256": "060a129cb4b2d04ebc596e936addb91ebcd63efd369939998f50708cfa5688d1",
            },
        ],
        required_softlinks = [struct(target = "llvm/amdgcn", link = "amdgcn")],
        rocm_root = "",
    ),
    "rocm_7.10.0_gfx94X_whl": struct(
        packages = [
            {
                "url": "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/rocm_sdk_libraries_gfx94x_dcgpu-7.10.0a20251009-py3-none-linux_x86_64.whl",
                "sha256": "e4aa688ef0f4c54e57b0746fe7a617d6ee57ce4d19164308803b3f3eaf07fb30",
            },
            {
                "url": "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/rocm_sdk_core-7.10.0a20251009-py3-none-linux_x86_64.whl",
                "sha256": "a284d98122a82464199b633d845909ce57c961f5a21fd890c5343fb27e2a110b",
            },
            {
                "url": "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/rocm_sdk_devel-7.10.0a20251009-py3-none-linux_x86_64.whl",
                "sha256": "21b4ad7fe2d667977e0acd9f77490c2c5296d0039b0f773c337375c4580ce69d",
                "sub_package": "rocm_sdk_devel/_devel.tar",
            },
        ],
        required_softlinks = [struct(target = "_rocm_sdk_devel/llvm/amdgcn", link = "_rocm_sdk_devel/amdgcn")],
        rocm_root = "_rocm_sdk_devel",
    ),
}
