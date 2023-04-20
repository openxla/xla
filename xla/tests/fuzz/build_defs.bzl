"""Build rules for XLA generated regression testing."""

load("//xla/tests:build_defs.bzl", "xla_test")

def hlo_test(name, hlo_files, **kwargs):
    for hlo in hlo_files:
        without_extension = hlo.split(".")[0]
        xla_test(
            name = without_extension,
            srcs = ["hlo_test_template.cc"],
            env = {"HLO_PATH": "$(location {})".format(hlo)},
            data = [hlo],
            deps = [
                "//xla/tests:hlo_test_base",
                "//xla:error_spec",
                "@tsl//tsl/platform:env",
                "@tsl//tsl/platform:test_main",
            ],
            real_hardware_only = True,
            **kwargs
        )
