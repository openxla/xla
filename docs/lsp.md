# Setting up LSP with clangd
## Background
Editors such as Emacs, Vim, or VS Code support features like code navigation, code completion, inline compiler error messages, and others, through [LSP](https://en.wikipedia.org/wiki/Language_Server_Protocol), the Language Server Protocol. A common language server with LSP support is [clangd](https://clangd.llvm.org), which relies on the presence of `compile_commands.json`, a JSON file with a record of the compile commands for each file in a project.
## How do I generate `compile_commands.json` for XLA source code?
Use the [build_tools/lint/generate_compile_commands.py](https://github.com/openxla/xla/blob/main/build_tools/lint/generate_compile_commands.py) script. At the time of writing, the following invocation from XLA repo root generates a `compile_commands.json` file in place:
```bash
bazel aquery "mnemonic(CppCompile, //xla/...)" --output=jsonproto | \
      python3 build_tools/lint/generate_compile_commands.py
```
Please refer to the script directly for updated usage instructions if you find that the above doesn't work for you or produces an error.
