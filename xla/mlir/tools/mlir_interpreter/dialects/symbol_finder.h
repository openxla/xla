/* This header file declares the FindSymbolInProcess function, which takes a symbol name as 
input and returns either the symbol's address or an error status, encapsulated in absl::StatusOr<void*> */

#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_DIALECTS_SYMBOL_FINDER_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_DIALECTS_SYMBOL_FINDER_H_

#include <string>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace mlir {
namespace interpreter {

absl::StatusOr<void*> FindSymbolInProcess(const std::string& symbol_name);

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_DIALECTS_SYMBOL_FINDER_H_
