load("//xla:xla.bzl", "xla_cc_test")
load("//xla/service:xla_compile.bzl", "xla_aot_compile_cpu")

xla_aot_compile_cpu(
    name = "stablehlo_axpy_executable",
    module = "stablehlo_axpy.mlir",
)

xla_cc_test(
    name = "stablehlo_aot_compile_test",
    srcs = ["stablehlo_aot_compile_test.cc"],
    data = [":stablehlo_axpy_executable"],
    deps = [
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:resource_loader",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)
