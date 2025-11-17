rocm_redist = {
    "rocm_7.10.0_gfx90X": struct(
        packages = [
            struct(
                url = "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx90X-dcgpu-7.10.0a20251106.tar.gz",
                sha256 = "a9270cac210e02f60a7f180e6a4d2264436cdcce61167440e6e16effb729a8ea",
                sub_package = None,
                strip_prefix = "",
                root = "",
            )
        ],
        required_softlinks = [struct(src = "llvm/amdgcn", dest = "amdgcn")],
        rocm_root = "",
    ),
    "rocm_7.10.0_gfx94X": struct(
        packages = [
            struct(
                url = "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx94X-dcgpu-7.10.0a20251107.tar.gz",
                sha256 = "486dbf647bcf9b78f21d7477f43addc7b2075b1a322a119045db9cdc5eb98380",
                sub_package = None,
                strip_prefix = "",
                root = "",
            )
        ],
        required_softlinks = [struct(src = "llvm/amdgcn", dest = "amdgcn")],
        rocm_root = "",
    ),
    "rocm_7.10.0_gfx94X_whl": struct(
        packages = [
            struct(
                url = "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/rocm_sdk_libraries_gfx94x_dcgpu-7.10.0a20251009-py3-none-linux_x86_64.whl",
                sha256 = "e4aa688ef0f4c54e57b0746fe7a617d6ee57ce4d19164308803b3f3eaf07fb30",
                sub_package = None,
                strip_prefix = "_rocm_sdk_libraries_gfx94X_dcgpu",
                root = "_rocm_sdk_libraries_gfx94X_dcgpu",
            ),
            struct(
                url = "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/rocm_sdk_core-7.10.0a20251009-py3-none-linux_x86_64.whl",
                sha256 = "a284d98122a82464199b633d845909ce57c961f5a21fd890c5343fb27e2a110b",
                sub_package = None,
                strip_prefix = "_rocm_sdk_core",
                root = "_rocm_sdk_core",
            ),
            struct(
                url = "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/rocm_sdk_devel-7.10.0a20251009-py3-none-linux_x86_64.whl",
                sha256 = "21b4ad7fe2d667977e0acd9f77490c2c5296d0039b0f773c337375c4580ce69d",
                sub_package = struct(
                    archive = "_devel.tar",
                    strip_prefix = "_rocm_sdk_devel",
                    root = "_rocm_sdk_devel",
                ),
                strip_prefix = "rocm_sdk_devel",
                root = "",
            ),
        ],
        required_softlinks = [],
        rocm_root = "_rocm_sdk_devel",
    ),
}
