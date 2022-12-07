load("@tsl//platform:rules_cc.bzl", "cc_library")
load("@tsl//:tsl.bzl", "if_nccl")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//third_party/tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//third_party/tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_helpers",
    srcs = ["gpu_helpers.cc"],
    hdrs = ["gpu_helpers.h"],
    visibility = ["//xla/pjrt:friends"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_mem_allocator",
        "//xla/stream_executor:kernel",
        "@com_google_absl//absl/types:span",
        "@tsl//framework:bfc_allocator",
        "@tsl//framework:device_id_impl",
        "@tsl//util:env_var",
    ],
)

cc_library(
    name = "se_gpu_pjrt_client",
    srcs = ["se_gpu_pjrt_client.cc"],
    hdrs = ["se_gpu_pjrt_client.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]),
    visibility = ["//xla/pjrt:friends"],
    deps = [
        ":gpu_helpers",
        "@com_google_absl//absl/strings",
        "//xla/pjrt:pjrt_stream_executor_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla:statusor",
        "//xla/client:client_library",
        "//xla/pjrt/distributed:client",
        "//xla/service:platform_util",
        "//xla:util",
        "@tsl//framework:bfc_allocator",
        "@tsl//framework:device_id",
        "@tsl//framework:device_id_impl",
        "@tsl//platform:env",
        "@tsl//platform:errors",
        "@tsl//util:env_var",
        "//xla/stream_executor:device_mem_allocator",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:tf_allocator_adapter",
    ] + if_cuda([
        ":nccl_id_store_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor/cuda:cuda_activation_header",
        "//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
    ]) + if_rocm([
        ":nccl_id_store_rocm",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

# We actually wish we could write if_cuda(if_nccl(...)) in :gpu_device,
# but Bazel does not allow nested selects. We can work around the problem using
# an intermediate library that has the conditional NCCL pieces that is only
# itself included as a dependency if CUDA is enabled.
cc_library(
    name = "nccl_id_store_cuda",
    srcs = ["nccl_id_store.cc"],
    hdrs = ["nccl_id_store.h"],
    defines = if_nccl(["NCCL_ENABLED=1"]),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "//xla/pjrt/distributed:client",
        "//xla:util",
        "//xla:statusor",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
    ] + if_nccl(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "nccl_id_store_rocm",
    srcs = ["nccl_id_store.cc"],
    hdrs = ["nccl_id_store.h"],
    defines = if_nccl(["NCCL_ENABLED=1"]),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "//xla/pjrt/distributed:client",
        "//xla:util",
        "//xla:statusor",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
    ] + if_nccl(["@local_config_nccl//:nccl"]),
)

tf_cc_test(
    name = "pjrt_client_test_se_gpu",
    srcs = ["pjrt_client_test_se_gpu.cc"],
    tags = [
        "no_oss",
        "notap",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":se_gpu_pjrt_client",
        "//xla/pjrt:pjrt_client_test_common",
        "//xla/service:gpu_plugin",
        "@tsl//platform:test_main",
    ],
)
