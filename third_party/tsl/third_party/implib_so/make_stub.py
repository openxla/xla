"""
Given a list of symbols, generates a stub.

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
import configparser
import os
import re
import string

from bazel_tools.tools.python.runfiles import runfiles

r = runfiles.Create()

def main():
  parser = argparse.ArgumentParser(description="Generates stubs for CUDA libraries.",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('symbols',
                      help="File containing a list of symbols.")
  parser.add_argument('--outdir', '-o',
                      help="Path to create wrapper at",
                      default='./')
  parser.add_argument('--target',
                      help="Target platform name, e.g. x86_64, aarch64.",
                      required=True)
  args = parser.parse_args()

  config_path = r.Rlocation(f'implib_so/arch/{args.target}/config.ini')
  table_path = r.Rlocation(f'implib_so/arch/{args.target}/table.S.tpl')
  trampoline_path = r.Rlocation(f'implib_so/arch/{args.target}/trampoline.S.tpl')

  cfg = configparser.ConfigParser(inline_comment_prefixes=';')
  cfg.read(config_path)
  ptr_size = int(cfg['Arch']['PointerSize'])

  with open(args.symbols, "r") as f:
    funs = [s.strip() for s in f.readlines()]

  # Generate assembly code, containing a table for the resolved symbols and the
  # trampolines.
  lib_name, _ = os.path.splitext(os.path.basename(args.symbols))

  with open(os.path.join(args.outdir, f'{lib_name}.tramp.S'), 'w') as f:
    with open(table_path, 'r') as t:
      table_text = string.Template(t.read()).substitute(
        lib_suffix=lib_name,
        table_size=ptr_size * (len(funs) + 1))
    f.write(table_text)

    with open(trampoline_path, 'r') as t:
      tramp_tpl = string.Template(t.read())

    for i, name in enumerate(funs):
      tramp_text = tramp_tpl.substitute(
        lib_suffix=lib_name,
        sym=name,
        offset=i*ptr_size,
        number=i)
      f.write(tramp_text)

  # Generates a list of symbols, formatted as a list of C++ strings.
  with open(os.path.join(args.outdir, f'{lib_name}.inc'), 'w') as f:
    sym_names = ',\n'.join(f'  "{name}"' for name in funs) + ','
    f.write(sym_names)

if __name__ == "__main__":
  main()