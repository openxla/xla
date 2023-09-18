"""
Given the path to a .so file, extracts a list of symbols that should be included
in a stub.

Example usage:
$ bazel run -c opt @tsl//third_party/implib_so:get_symbols /usr/local/cuda/lib64/libcudart.so > third_party/tsl/tsl/cuda/cudart.symbols

This code is derived from implib-gen, from the Implib.so project, which is under
the following license.

The MIT License (MIT)

Copyright (c) 2017-2023 Yury Gribov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import importlib

# We can't import implib-gen directly because it has a dash in its name.
implib = importlib.import_module("implib-gen")


def _is_exported(s):
  conditions = [
    s['Bind'] != 'LOCAL',
    s['Type'] != 'NOTYPE',
    s['Ndx'] != 'UND',
    s['Name'] not in ['', '_init', '_fini']]
  return all(conditions)


def main():
  """Driver function"""
  parser = argparse.ArgumentParser(description="Extract a list of symbols from a shared library",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('library',
                      help="Library to be wrapped.")
  args = parser.parse_args()
  syms = list(filter(_is_exported, implib.collect_syms(args.library)))
  orig_funs = filter(lambda s: s['Type'] == 'FUNC', syms)

  warn_versioned = False
  all_funs = set()
  for s in orig_funs:
    if not s['Default']:
      # TODO: support versions
      if not warn_versioned:
        implib.warn(f"library {args.library} contains versioned symbols which are NYI")
        warn_versioned = True
      continue
    all_funs.add(s['Name'])

  for f in sorted(all_funs):
    print(f)

if __name__ == "__main__":
  main()