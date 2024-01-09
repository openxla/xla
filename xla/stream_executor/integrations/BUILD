load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("@tsl//tsl:tsl.bzl", "set_external_visibility")
load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([":friends"]),
    licenses = ["notice"],
)

# StreamExecutor clients that can depend on targets defined in stream_executor package.
package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

#===--------------------------------------------------------------------------------------------===#
# Exporting headers for Tensorflow
#===--------------------------------------------------------------------------------------------===#

# Tensorflow device memory allocators are aliases for StreamExecutor allocators, we export
# headers for Tensorflow to build shared libraries in OSS.

filegroup(
    name = "device_mem_allocator_headers",
    srcs = [
        "device_host_allocator.h",
        "device_mem_allocator.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor integrations
#===--------------------------------------------------------------------------------------------===#

# Libraries integrating StreamExecutor into framework APIs, e.g. plugging StreamExecutor device
# memory allocator into Tensorflow.

cc_library(
    name = "tf_allocator_adapter",
    srcs = ["tf_allocator_adapter.cc"],
    hdrs = ["tf_allocator_adapter.h"],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "device_mem_allocator",
    hdrs = [
        "device_host_allocator.h",
        "device_mem_allocator.h",
    ],
    deps = [
        "//xla/stream_executor",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:device_id",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)
