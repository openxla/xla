#ifndef _WIN32

#include "symbol_finder.h"
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