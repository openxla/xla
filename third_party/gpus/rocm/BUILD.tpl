load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_version_number")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:private"])

string_flag(
    name = "rocm_path_type",
    build_setting_default = "system",
    values = [
        "hermetic",
        "multiple",
        "system",
        "link_only",
    ],
)

config_setting(
    name = "build_hermetic",
    flag_values = {
        ":rocm_path_type": "hermetic",
    },
)

config_setting(
    name = "multiple_rocm_paths",
    flag_values = {
        ":rocm_path_type": "multiple",
    },
)

config_setting(
    name = "link_only",
    flag_values = {
        ":rocm_path_type": "link_only",
    },
)

config_setting(
    name = "using_hipcc",
    values = {
        "define": "using_rocm_hipcc=true",
    },
)

cc_library(
    name = "config",
    hdrs = [
        "rocm_config/rocm_config.h",
    ],
    include_prefix = "rocm",
    strip_include_prefix = "rocm_config",
)

cc_library(
    name = "config_hermetic",
    hdrs = [
        "rocm_config_hermetic/rocm_config.h",
    ],
    include_prefix = "rocm",
    strip_include_prefix = "rocm_config_hermetic",
)

cc_library(
    name = "rocm_config",
    visibility = ["//visibility:public"],
    deps = select({
        ":build_hermetic": [
            ":config_hermetic",
        ],
        "//conditions:default": [
            "config",
        ],
    }),
)

# This target is required to
# add includes that are used by rocm headers themself
# through the virtual includes
# cleaner solution would be to adjust the xla code
# and remove include prefix that is used to include rocm headers.
cc_library(
    name = "rocm_headers_includes",
    hdrs = glob([
        "%{rocm_root}/include/**",
    ]),
    strip_include_prefix = "%{rocm_root}/include",
)

cc_library(
    name = "rocm_headers",
    hdrs = glob([
        "%{rocm_root}/include/**",
        "%{rocm_root}/lib/llvm/lib/**/*.h",
    ]),
    defines = ["MIOPEN_BETA_API=1"],
    include_prefix = "rocm",
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_headers_includes",
    ],
)

# Provides -L and -Wl,-rpath flags for ROCm libraries.
# These must live in a cc_library (not a toolchain feature) because
# cc_library linkopts propagate transitively through CcInfo to the
# final linking target, whereas toolchain features do not.
cc_library(
    name = "rocm_rpath",
    linkopts = select({
        ":build_hermetic": [
            "-Wl,-rpath,external/local_config_rocm/rocm/%{rocm_root}/lib",
            "-Lexternal/local_config_rocm/rocm/%{rocm_root}/lib",
        ],
        ":link_only": [
            "-Lexternal/local_config_rocm/rocm/%{rocm_root}/lib",
        ],
        ":multiple_rocm_paths": [
            "-Wl,-rpath=%{rocm_lib_paths}",
            "-Lexternal/local_config_rocm/rocm/%{rocm_root}/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,/opt/rocm/lib",
            "-Lexternal/local_config_rocm/rocm/%{rocm_root}/lib",
        ],
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "hip",
    actual = ":hip_runtime",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hip_runtime",
    linkopts = ["-lamdhip64"],
    visibility = ["//visibility:public"],
    deps = [
        ":hip_runtime_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hip_runtime_libs",
    data = glob(
        [
            "%{rocm_root}/lib/libamdhip64.so*",
        ],
    ),
    deps = [
        ":amd_comgr_libs",
        ":hiprtc_libs",
        ":hsa_rocr_libs",
        ":rocprofiler_register_libs",
        ":system_libs",
    ],
)

cc_library(
    name = "hsa_rocr_libs",
    data = glob(["%{rocm_root}/lib/libhsa-runtime64.so*"]),
    deps = [
        ":rocprofiler_register_libs",
        ":system_libs",
    ],
)

cc_library(
    name = "hiprtc_libs",
    data = glob(
        [
            "%{rocm_root}/lib/libhiprtc.so*",
            "%{rocm_root}/lib/libhiprtc-builtins.so*",
        ],
    ),
    deps = [
        ":amd_comgr_libs",
        ":hsa_rocr_libs",
    ],
)

cc_library(
    name = "amd_comgr_libs",
    data = glob(
        [
            "%{rocm_root}/lib/libamd_comgr_loader.so*",
            "%{rocm_root}/lib/libamd_comgr.so*",
            "%{rocm_root}/lib/llvm/lib/libLLVM.so*",
        ],
    ),
    deps = [
        ":system_libs",
    ],
)

cc_library(
    name = "rocprofiler_register_libs",
    data = glob(
        [
            "%{rocm_root}/lib/librocprofiler-register.so*",
        ],
    ),
)

cc_library(
    name = "rocblas",
    linkopts = ["-lrocblas"],
    visibility = ["//visibility:public"],
    deps = [
        ":rocblas_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocblas_libs",
    data = glob([
        "%{rocm_root}/lib/librocblas.so*",
        "%{rocm_root}/lib/rocblas/**",
    ]),
    deps = [
        ":hip_runtime_libs",
        ":hipblaslt_libs",
        ":roctx_libs",
    ],
)

cc_library(
    name = "hipfft",
    linkopts = ["-lhipfft"],
    visibility = ["//visibility:public"],
    deps = [
        ":hipfft_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipfft_libs",
    data = glob(["%{rocm_root}/lib/libhipfft.so*"]),
    deps = [
        ":hip_runtime_libs",
        ":rocfft_libs",
    ],
)

cc_library(
    name = "rocfft_libs",
    data = glob(["%{rocm_root}/lib/librocfft.so*"]),
    deps = [
        ":hip_runtime_libs",
        ":hiprtc_libs",
    ],
)

cc_library(
    name = "hiprand",
    linkopts = ["-lhiprand"],
    visibility = ["//visibility:public"],
    deps = [
        ":hiprand_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hiprand_libs",
    data = glob(["%{rocm_root}/lib/libhiprand.so*"]),
    deps = [
        ":hip_runtime_libs",
        ":rocrand_libs",
    ],
)

cc_library(
    name = "rocrand_libs",
    data = glob(["%{rocm_root}/lib/librocrand.so*"]),
    deps = [
        ":hip_runtime_libs",
    ],
)

cc_library(
    name = "miopen",
    linkopts = ["-lMIOpen"],
    visibility = ["//visibility:public"],
    deps = [
        ":miopen_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "miopen_libs",
    data = glob([
        "%{rocm_root}/lib/libMIOpen.so*",
        "%{rocm_root}/share/miopen/**",
    ]),
    deps = [
        ":amd_comgr_libs",
        ":hip_runtime_libs",
        ":hipblaslt_libs",
        ":hiprtc_libs",
        ":rocblas_libs",
        ":roctx_libs",
        ":system_libs",
    ],
)

cc_library(
    name = "rccl",
    linkopts = ["-lrccl"],
    visibility = ["//visibility:public"],
    deps = [
        ":rccl_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rccl_libs",
    data = glob(["%{rocm_root}/lib/librccl.so*"]),
    deps = [
        ":hip_runtime_libs",
        ":rocm_smi_libs",
        ":rocprofiler_register_libs",
        ":roctx_libs",
    ],
)

cc_library(
    name = "rocm_smi_libs",
    data = glob([
        "%{rocm_root}/lib/librocm_smi64.so*",
        "%{rocm_root}/lib/libamd_smi.so*",
    ]),
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocprim",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers_includes",
    ],
)

cc_library(
    name = "hipsparse",
    linkopts = ["-lhipsparse"],
    visibility = ["//visibility:public"],
    deps = [
        ":hipsparse_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipsparse_libs",
    data = glob(["%{rocm_root}/lib/libhipsparse*.so*"]),
    deps = [
        ":hip_runtime_libs",
        ":rocsparse_libs",
    ],
)

cc_library(
    name = "rocsparse_libs",
    data = glob(["%{rocm_root}/lib/librocsparse*.so*"]),
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "roctx_libs",
    data = glob([
        "%{rocm_root}/lib/libroctx64.so*",
    ]),
    deps = [":rocm_config"],
)

cc_library(
    name = "roctracer",
    linkopts = ["-lroctracer64"],
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers_includes",
        ":rocm_rpath",
        ":roctracer_libs",
    ],
)

cc_library(
    name = "roctracer_libs",
    data = glob([
        "%{rocm_root}/lib/libroctracer64.so*",
    ]),
    deps = [
        ":hsa_rocr_libs",
    ],
)

cc_library(
    name = "rocprofiler_sdk",
    linkopts = ["-lrocprofiler-sdk"],
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers_includes",
        ":rocm_rpath",
        ":rocprofiler_sdk_libs",
    ],
)

cc_library(
    name = "rocprofiler_sdk_libs",
    data = glob(["%{rocm_root}/lib/librocprofiler-sdk*.so*"]),
    deps = [
        ":amd_comgr_libs",
        ":system_libs",
    ],
)

cc_library(
    name = "rocsolver",
    linkopts = ["-lrocsolver"],
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers_includes",
        ":rocm_rpath",
        ":rocsolver_libs",
    ],
)

cc_library(
    name = "rocsolver_libs",
    data = glob([
        "%{rocm_root}/lib/librocsolver.so*",
        "%{rocm_root}/lib/host-math/lib/*.so*",
    ]),
    deps = [
        ":hip_runtime_libs",
        ":rocblas_libs",
    ],
)

cc_library(
    name = "hipsolver",
    linkopts = ["-lhipsolver"],
    visibility = ["//visibility:public"],
    deps = [
        ":hipsolver_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipsolver_libs",
    data = glob(["%{rocm_root}/lib/libhipsolver.so*"]),
    deps = [
        ":hip_runtime_libs",
        ":rocblas_libs",
        ":rocsolver_libs",
        ":rocsparse_libs",
    ],
)

cc_library(
    name = "hipblas",
    linkopts = ["-lhipblas"],
    visibility = ["//visibility:public"],
    deps = [
        ":hipblas_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipblas_libs",
    data = glob(["%{rocm_root}/lib/libhipblas.so*"]),
    deps = [
        ":rocblas_libs",
        ":rocsolver_libs",
    ],
)

cc_library(
    name = "hipblaslt",
    linkopts = ["-lhipblaslt"],
    visibility = ["//visibility:public"],
    deps = [
        ":hipblaslt_libs",
        ":rocm_headers_includes",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipblaslt_libs",
    data = glob([
        "%{rocm_root}/lib/hipblaslt/**",
        "%{rocm_root}/lib/libhipblaslt.so*",
        "%{rocm_root}/lib/librocroller.so*",
    ]),
    deps = [
        ":hip_runtime_libs",
        ":roctx_libs",
    ],
)

cc_library(
    name = "system_libs",
    data = glob([
        "%{rocm_root}/lib/rocm_sysdeps/lib/*.so*",
        "%{rocm_root}/lib/rocm_sysdeps/share/**",
    ]),
)

filegroup(
    name = "rocm_root",
    srcs = [
        "%{rocm_root}/bin/clang-offload-bundler",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "toolchain_data",
    srcs = glob([
        "%{rocm_root}/bin/hipcc",
        "%{rocm_root}/lib/llvm/**",
        "%{rocm_root}/share/hip/**",
        "%{rocm_root}/amdgcn/**",
        "%{rocm_root}/lib/rocm_sysdeps/lib/*.so*",
        "%{rocm_root}/lib/libamd_comgr_loader.so*",
        "%{rocm_root}/lib/libamd_comgr.so*",
    ]),
    visibility = ["//visibility:public"],
)

#TODO(rocm) Be more specific
filegroup(
    name = "all_files",
    srcs = glob(["%{rocm_root}/**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rocminfo",
    srcs = glob([
        "%{rocm_root}/bin/rocminfo",
        "%{rocm_root}/lib/libhsa-runtime64.so*",
        "%{rocm_root}/lib/rocm_sysdeps/lib/*",
        "%{rocm_root}/lib/librocprofiler-register.so.*",
    ]),
    visibility = ["//visibility:public"],
)

platform(
    name = "linux_x64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://%{rocm_rbe_docker_image}",
        "Pool": "%{rocm_rbe_pool}",
        "OSFamily": "Linux",
    },
)
