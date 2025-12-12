load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_version_number", "select_threshold")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:private"])

string_flag(
    name = "rocm_path_type",
    build_setting_default = "system",
    values = [
        "hermetic",
        "multiple",
        "system",
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
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":hip",
        ":hipblas",
        ":hipblaslt",
        ":hipfft",
        ":hiprand",
        ":hipsolver",
        ":hipsparse",
        ":hsa_rocr",
        ":miopen",
        ":rocblas",
        ":rocm_config",
        ":rocprofiler_register",
        ":rocsolver",
        ":rocsparse",
        ":roctracer",
    ],
)

cc_library(
    name = "hsa_rocr",
    hdrs = glob(["%{rocm_root}/include/hsa/**"]),
    data = glob(["%{rocm_root}/lib/libhsa-runtime64.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lhsa-runtime64"],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

# workaround to bring data to the same fs layout as expected in the rocm libs
# rocblas assumes that miopen db files are located in ../share/miopen/db directory
# hibplatslt assumes that tensile files are located in ../hipblaslt/library directory
cc_library(
    name = "rocm_rpath",
    linkopts = select({
        ":build_hermetic": [
            "-Wl,-rpath,external/local_config_rocm/rocm/%{rocm_root}/lib",
            "-Wl,-rpath,external/local_config_rocm/rocm/%{rocm_root}/lib/llvm/lib",
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

cc_library(
    name = "hip",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_hip",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hiprtc",
    data = glob(
        [
            "%{rocm_root}/lib/libhiprtc.so*",
            "%{rocm_root}/lib/libhiprtc-builtins.so*",
        ],
    ),
    linkopts = [
        "-lhiprtc",
        "-lpthread",
    ],
    deps = [":rocm_rpath"],
)

cc_library(
    name = "rocm_hip",
    hdrs = glob(["%{rocm_root}/include/hip/**"]),
    data = glob(
        [
            "%{rocm_root}/lib/libamdhip64.so*",
        ],
        exclude = [
            # exclude files like libamdhip64.so.7.1.25445-7484b05b13 -> misplaced
            "%{rocm_root}/**/*.so.*.*",
        ],
    ),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lamdhip64"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":amd_comgr",
        ":hiprtc",
        ":hsa_rocr",
        ":rocm_config",
        ":rocm_rpath",
        ":rocm_smi",
        ":rocprofiler_register",
        ":system_libs",
    ],
)

# Used by jax_rocm_plugin to minimally link to hip runtime.
cc_library(
    name = "hip_runtime",
    srcs = glob(
        [
            "%{rocm_root}/lib/libamdhip*.so*",
        ],
        exclude = [
            # exclude files like libamdhip64.so.7.1.25445-7484b05b13 -> misplaced
            "%{rocm_root}/**/*.so.*.*",
        ],
    ),
    hdrs = glob(["%{rocm_root}/include/hip/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":amd_comgr",
        ":hiprtc",
        ":rocm_config",
        ":rocm_rpath",
        ":rocprofiler_register",
        ":system_libs",
    ],
)

cc_library(
    name = "rocblas",
    hdrs = glob(["%{rocm_root}/include/rocblas/**"]),
    data = glob([
        "%{rocm_root}/lib/librocblas.so*",
        "%{rocm_root}/lib/librocroller.so*",
        "%{rocm_root}/lib/rocblas/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    # rocblas is supposed to be loaded by dso
    #linkopts = ["-lrocblas", "-lrocroller"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":hipblas",
        ":hipblaslt",
        ":rocm_config",
        ":rocm_hip",
        ":rocm_rpath",
        ":roctracer",
    ],
)

cc_library(
    name = "rocfft",
    data = glob(["%{rocm_root}/lib/librocfft.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lrocfft"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipfft",
    data = glob(["%{rocm_root}/lib/libhipfft.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lhipfft"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":rocfft",
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hiprand",
    hdrs = glob(["%{rocm_root}/include/hiprand/**"]),
    data = glob(["%{rocm_root}/lib/libhiprand.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
        "%{rocm_root}/include/rocrand",
    ],
    linkopts = ["-lhiprand"],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "miopen",
    hdrs = glob(["%{rocm_root}/include/miopen/**"]),
    data = glob([
        "%{rocm_root}/lib/libMIOpen*.so*",
        "%{rocm_root}/share/miopen/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lMIOpen"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm-core",
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rccl",
    hdrs = glob(["%{rocm_root}/include/rccl/**"]),
    data = glob(["%{rocm_root}/lib/librccl.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lrccl"],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
        ":roctracer",
        ":system_libs",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocprim",
    srcs = [
        "%{rocm_root}/include/hipcub/hipcub_version.hpp",
        "%{rocm_root}/include/rocprim/rocprim_version.hpp",
    ],
    hdrs = glob([
        "%{rocm_root}/include/hipcub/**",
        "%{rocm_root}/include/rocprim/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/hipcub",
        "%{rocm_root}/include/rocprim",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_headers",
    ],
)

cc_library(
    name = "hipsparse",
    hdrs = glob(["%{rocm_root}/include/hipsparse/**"]),
    data = glob(["%{rocm_root}/lib/libhipsparse*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    linkopts = [
        "-lhipsparse",
        "-lhipsparselt",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "roctracer",
    hdrs = glob(["%{rocm_root}/include/roctracer/**"]),
    data = glob([
        "%{rocm_root}/lib/libroctracer64.so*",
        "%{rocm_root}/lib/libroctx64.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    linkopts = [
        "-lroctracer64",
        "-lroctx64",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocprofiler-sdk",
    hdrs = glob(["%{rocm_root}/include/rocprofiler-sdk/**"]),
    data = glob([
        "%{rocm_root}/lib/librocprofiler-sdk*.so*",
        "%{rocm_root}/lib/librocprofiler-sdk/**/*.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    linkopts = ["-lrocprofiler-sdk"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocsolver",
    hdrs = glob(["%{rocm_root}/include/rocsolver/**"]),
    data = glob(["%{rocm_root}/lib/librocsolver.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    linkopts = ["-lrocsolver"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocsparse",
    data = glob(["%{rocm_root}/lib/librocsparse*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    linkopts = ["-lrocsparse"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipsolver",
    hdrs = glob(["%{rocm_root}/include/hipsolver/**"]),
    data = glob(["%{rocm_root}/lib/libhipsolver*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    # do not link hipsolver as libcholmod.so.5 is missing in hermetic builds
    #linkopts = ["-lhipsolver"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipblas",
    hdrs = glob(["%{rocm_root}/include/hipblas/**"]),
    data = glob(["%{rocm_root}/lib/libhipblas.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    linkopts = ["-lhipblas"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":hipblas-common",
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipblas-common",
    hdrs = glob(["%{rocm_root}/include/hipblas-common/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocm-core",
    srcs = glob(["%{rocm_root}/lib/librocm-core.so*"]),
    linkopts = ["-lrocm-core"],
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "hipblaslt",
    hdrs = glob(["%{rocm_root}/include/hipblaslt/**"]),
    data = glob([
        "%{rocm_root}/lib/hipblaslt/**",
        "%{rocm_root}/lib/libhipblaslt.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/hipblaslt",
    ],
    linkopts = ["-lhipblaslt"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":hip_runtime",
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocrand",
    srcs = glob(["%{rocm_root}/lib/librocrand*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rocrand/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocprofiler_register",
    srcs = glob([
        "%{rocm_root}/lib/librocprofiler-register.so*",
    ]),
    deps = [":rocm_config"],
)

cc_library(
    name = "amd_comgr_dynamic",
    srcs = ["%{rocm_root}/lib/libamd_comgr_stub.a"],
    hdrs = glob(["%{rocm_root}/include/amd_comgr/**"]),
    data = glob([
        "%{rocm_root}/lib/libamd_comgr_loader.so*",
        "%{rocm_root}/lib/libamd_comgr.so*",
        "%{rocm_root}/lib/llvm/lib/libLLVM.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lamd_comgr_loader"],
    strip_include_prefix = "%{rocm_root}",
    deps = [
        ":rocm_config",
        ":rocm_rpath",
        ":system_libs",
    ],
)

cc_library(
    name = "amd_comgr_static",
    hdrs = glob(["%{rocm_root}/include/amd_comgr/**"]),
    data = glob([
        "%{rocm_root}/lib/libamd_comgr.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lamd_comgr"],
    strip_include_prefix = "%{rocm_root}",
    deps = [
        ":rocm_config",
        ":rocm_rpath",
        ":system_libs",
    ],
)

alias(
    name = "amd_comgr",
    actual = select_threshold(
        above_or_eq = ":amd_comgr_dynamic",
        below = ":amd_comgr_static",
        threshold = 71000,
        value = rocm_version_number(),
    ),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_smi",
    hdrs = glob([
        "%{rocm_root}/include/oam/**",
        "%{rocm_root}/include/rocm_smi/**",
    ]),
    data = glob([
        "%{rocm_root}/lib/librocm_smi64.so*",
        "%{rocm_root}/lib/libroam.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lrocm_smi64"],
    strip_include_prefix = "%{rocm_root}",
    deps = [
        ":rocm_config",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "system_libs",
    srcs = glob([
        "%{rocm_root}/lib/rocm_sysdeps/lib/*.so*",
    ]),
    data = glob([
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
        "%{rocm_root}/lib/librocprofiler-register.so.0*",
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
        "OSFamily": "Linux",
    },
)
