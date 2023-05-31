load("//xla:xla.bzl", "xla_cc_test")

xla_cc_test(
    name = "stablehlo_compile_test",
    srcs = ["stablehlo_compile_test.cc"],
    data = ["stablehlo_axpy.mlir"],
    deps = [
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/pjrt:local_device_state",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)
