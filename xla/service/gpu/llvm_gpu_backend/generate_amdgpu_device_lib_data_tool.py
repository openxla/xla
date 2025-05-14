import argparse
import subprocess
import itertools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llvm_link_bin", required=True, help="Path to the llvm-link binary"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output filename for the C++ header"
    )
    parser.add_argument(
        "input_files", nargs="+", help="Variable number of input filenames"
    )
    parser.add_argument(
        "--cpp_namespace", default="", help="Namespace to be used when generating data"
    )
    parser.add_argument(
        "--cpp_identifier", required=True, help="Identifier to be used to refer to data"
    )

    args = parser.parse_args()
    llvm_link_bin = args.llvm_link_bin
    output_filename = args.output
    input_filenames = args.input_files
    cpp_namespace = args.cpp_namespace
    cpp_identifier = args.cpp_identifier

    result = subprocess.run(
        [llvm_link_bin, "-f", "-o", "-", "/dev/null"]
        + list(
            itertools.chain.from_iterable(("--override", f) for f in input_filenames)
        ),
        capture_output=True,
        check=True,
    )

    llvm_output = result.stdout
    data_string = ",".join(
        str(byte if byte < 128 else byte - 256) for byte in llvm_output
    )

    with open(output_filename, "w") as output_file:
        output_file.write(f"""\
#pragma once

#include "llvm/ADT/StringRef.h"

namespace {cpp_namespace} {{
  inline const char kRaw_{cpp_identifier}[] = {{{data_string}}};
  constexpr llvm::StringRef {cpp_identifier}{{kRaw_{cpp_identifier}, sizeof(kRaw_{cpp_identifier})}};
}} // namespace {cpp_namespace}
""")


if __name__ == "__main__":
    main()
