# copybara:uncomment package(default_applicable_licenses = ["//third_party/tensorflow:license"])

package_group(
    name = "friends",
    includes = ["//tsl:internal"],
)

package_group(
    name = "internal",
    packages = [
        "//third_party/tensorflow/core/profiler/...",
        "//third_party/tensorflow/python/eager/...",
        "//third_party/tensorflow/python/profiler/...",
        "//third_party/tensorflow/python/tpu/profiler/...",
        "//tsl/profiler/...",
        "//xla/backends/profiler/gpu/...",
    ],
)

package_group(
    name = "xla_profiler_backends",
    packages = ["//xla/backends/profiler/..."],
)

package_group(
    name = "xla_internal",
    packages = [
        "//xla/...",
    ],
)
