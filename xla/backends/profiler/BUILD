load("@tsl//:tsl.bzl", "if_libtpu", "tsl_gpu_library")

# copybara:uncomment package(default_applicable_licenses = ["//third_party/tensorflow:license"])

package_group(
    name = "friends",
    packages = [
        "//xla/backends/profiler/cpu",
        "//xla/backends/profiler/gpu",
        "//xla/backends/profiler/tpu",
    ],
)

tsl_gpu_library(
    name = "profiler_backends",
    cuda_deps = [
        "//xla/backends/profiler/gpu:device_tracer",
    ],
    visibility = ["//third_party/tensorflow:internal"],
    deps = [
        "//xla/backends/profiler/cpu:host_tracer",
        "//xla/backends/profiler/cpu:metadata_collector",
    ] + if_libtpu([
        "//xla/backends/profiler/tpu:tpu_tracer",
    ]),
    alwayslink = True,
)
