load("//xla/tests/fuzz:build_defs.bzl", "hlo_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

filegroup(
    name = "fuzz_corpus",
    srcs = glob(
        include = ["corpus/rand_*.hlo"],
        exclude = [
            "corpus/rand_000001.hlo",  # fails on GPU
            "corpus/rand_000004.hlo",  # times out during coverage
            # These fail on all platforms
            "corpus/rand_000060.hlo",
            "corpus/rand_000067.hlo",
            "corpus/rand_000072.hlo",
        ],
    ),
    visibility = ["//xla/tests:friends"],
)

hlo_test(
    name = "rand",
    hlo_files = glob(
        include = ["corpus/rand_*.hlo"],
        exclude = [
            "corpus/rand_000001.hlo",  # fails on GPU
            "corpus/rand_000004.hlo",  # times out during coverage
            # These fail on all platforms
            "corpus/rand_000060.hlo",
            "corpus/rand_000067.hlo",
            "corpus/rand_000072.hlo",
        ],
    ),
)
