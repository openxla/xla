#ifdef _WIN32

#include "symbol_finder.h"
#include <windows.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

absl::StatusOr<void*> FindSymbolInProcess(const std::string& symbol_name) {
  HMODULE handle = GetModuleHandle(NULL);
  if (handle) {
    void* sym = GetProcAddress(handle, symbol_name.c_str());
    if (!sym) {
      return absl::NotFoundError("Callee not found");
    }
    return sym;
  } else {
    return absl::InternalError("Failed to get module handle");
  }
}

#endif  // _WIN32