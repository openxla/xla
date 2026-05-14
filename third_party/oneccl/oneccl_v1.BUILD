load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@local_config_sycl//sycl:build_defs.bzl", "sycl_library")

_CMAKE_COMMON_LIST = {
    "#cmakedefine CCL_PRODUCT_STATUS": "#define CCL_PRODUCT_STATUS",
    "#cmakedefine CCL_PRODUCT_BUILD_DATE": "#define CCL_PRODUCT_BUILD_DATE",
    "#cmakedefine CCL_PRODUCT_FULL": "#define CCL_PRODUCT_FULL",
    "@CCL_MAJOR_VERSION@": "2022",
    "@CCL_MINOR_VERSION@": "1",
    "@CCL_UPDATE_VERSION@": "0",
    "@CCL_PRODUCT_STATUS@": "Gold",
    "@CCL_PRODUCT_BUILD_DATE@": "2026-05-13",
    "@CCL_PRODUCT_FULL@": "release",
}

expand_template(
    name = "oneccl_config_h",
    out = "include/oneapi/ccl/config.h",
    template = "include/oneapi/ccl/config.h.in",
    substitutions = _CMAKE_COMMON_LIST,
)

cc_import(
    name = "mpi",
    shared_library = "deps/mpi/lib/libmpi.so.12",
    hdrs = glob(["deps/mpi/include/**/*.h"]),
    visibility = ["//visibility:public"],
)

# Separate C library for PMI C files (without -fsycl)
cc_library(
    name = "pmi_c_files",
    srcs = [
        "src/atl/util/pm/pmi_rt/pmi/simple_pmi.c",
        "src/atl/util/pm/pmi_rt/pmi/simple_pmiutil.c",
    ],
    hdrs = glob([
        "src/atl/util/pm/pmi_rt/pmi/*.h",
    ]),
    includes = [
        "src/atl/util/pm/pmi_rt/pmi",
    ],
    copts = [
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-D_GNU_SOURCE",
        "-Wno-register",
    ],
)

sycl_library(
    name = "oneccl_v1",
    srcs = glob([
        "src/atl/*.cpp",
        "src/atl/mpi/*.cpp",
        "src/atl/ofi/*.cpp",
        "src/atl/util/pm/pmi_resizable_rt/**/*.cpp",
        "src/atl/util/pm/pmi_rt/pmi_simple.cpp",
        "src/occp/inc/*.cpp",
        "src/occp/infra/*.cpp",
        "src/occp/occp/*.cpp",
        "src/occp/protocol/*.cpp",
        "src/coll/**/*.cpp",
        "src/comm/*.cpp",
        "src/common/**/*.cpp",
        "src/comp/**/*.cpp",
        "src/exec/**/*.cpp",
        "src/fusion/fusion.cpp",
        "src/hwloc/hwloc_wrapper.cpp",
        "src/native_device_api/sycl/export.cpp",
        "src/parallelizer/parallelizer.cpp",
        "src/sched/**/*.cpp",
        "src/topology/*.cpp",
        "src/unordered_coll/unordered_coll.cpp",
        "src/*.cpp",
        "src/umf/ipc.cpp",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "src/atl/mpi/atl_mpi.hpp",
    ]) + [":oneccl_config_h"],
    defines = [
    "CCL_C_COMPILER=\\\"Clang\\\"",
    "CCL_CXX_COMPILER=\\\"Clang\\\"",
    "CCL_ENABLE_MPI",
    "CCL_ENABLE_OFI_HMEM",
    "CCL_ENABLE_PMIX",
    "CCL_ENABLE_STUB_BACKEND",
    "CCL_ENABLE_SYCL_INTEROP_EVENT",
    "CCL_ENABLE_UMF",
    "CCL_SYCL_ENABLE_ARCB",
    "CCL_SYCL_VEC_SUPPORT_BF16",
    "CCL_SYCL_VEC_SUPPORT_FP16",
    "_FORTIFY_SOURCE=2",
    "_GNU_SOURCE",
],
    includes = [
        "include",
        "src",
        "src/atl",
        "src/occp/inc",
        "src/occp/infra",
        "src/occp/protocol",
        "deps/level_zero/include",
        "deps/ofi/include",
        "deps/pmix/include",
        "deps/umf/include",
        "deps/mpi/include",
    ],
    copts = [
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fvisibility=internal",
        "-Wno-implicit-fallthrough",
        "-Wformat",
        "-Wformat-security",
        "-fstack-protector",
    ],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@hwloc",
        ":mpi",
        ":pmi_c_files",
    ],
)
