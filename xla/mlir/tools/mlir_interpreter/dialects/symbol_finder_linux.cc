/* This code defines a POSIX-compatible function, FindSymbolInProcess, 
   which uses dlsym to locate a function or variable symbol (symbol_name) in the current process;
   it returns the symbol's address if found or an error if not */
   
#ifndef _WIN32

#include "xla/mlir/tools/mlir_interpreter/dialects/symbol_finder.h"
#include <dlfcn.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

absl::StatusOr<void*> FindSymbolInProcess(const std::string& symbol_name) {
  void* sym = dlsym(RTLD_DEFAULT, symbol_name.c_str());
  if (sym == nullptr) {
    return absl::NotFoundError("Callee not found");
  }
  return sym;
}
#endif  // !_WIN32
