#ifndef SYMBOL_FINDER_H_
#define SYMBOL_FINDER_H_

#include <string>
#include "absl/status/statusor.h"

absl::StatusOr<void*> FindSymbolInProcess(const std::string& symbol_name);

#endif  // SYMBOL_FINDER_H_