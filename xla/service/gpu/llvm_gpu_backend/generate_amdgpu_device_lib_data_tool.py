import argparse
import subprocess
import itertools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llvm_link_bin", required=True, help="Path to the llvm-link binary"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output filename for the C header"
    )
    parser.add_argument(
        "input_files", nargs="+", help="Variable number of input filenames"
    )

    args = parser.parse_args()
    llvm_link_bin = args.llvm_link_bin
    output_filename = args.output
    input_filenames = args.input_files

    result = subprocess.run(
        [llvm_link_bin, "-f", "-o", "-", "/dev/null"]
        + list(
            itertools.chain.from_iterable(("--override", f) for f in input_filenames)
        ),
        capture_output=True,
        check=True,
    )

    llvm_output = result.stdout
    hex_string = ",".join(
        str(byte if byte < 128 else byte - 256) for byte in llvm_output
    )

    with open(output_filename, "w") as output_file:
        output_file.write(hex_string)


if __name__ == "__main__":
    main()
