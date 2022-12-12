/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>
#include <utility>

#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/ParseUtilities.h"  // from @llvm-project
#include "xla/mlir_hlo/gml_st/IR/gml_st_ops.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/thlo/IR/thlo_ops.h"
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"
#include "tsl/platform/init_main.h"

struct Options {
  llvm::cl::opt<std::string> input_filename{llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-")};
  llvm::cl::opt<bool> run_all_functions{
      "run-all", llvm::cl::desc("Run all functions in the module"),
      llvm::cl::init(false)};
};

// Copied from from mlir/lib/ExecutionEngine/JitRunner.cpp
static mlir::OwningOpRef<mlir::Operation *> parseMLIRInput(
    llvm::StringRef inputFilename, bool insertImplicitModule,
    mlir::MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  mlir::OwningOpRef<mlir::Operation *> module =
      mlir::parseSourceFileForTool(sourceMgr, context, insertImplicitModule);
  if (!module) return nullptr;
  if (!module.get()->hasTrait<mlir::OpTrait::SymbolTable>()) {
    llvm::errs() << "Error: top-level op must be a symbol table.\n";
    return nullptr;
  }
  return module;
}

mlir::LogicalResult Run(mlir::ModuleOp module, mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::interpreter::InterpreterValue> args;
  absl::SharedBitGen bitgen;
  for (auto arg : function.getBody().getBlocks().front().getArguments()) {
    args.push_back(*mlir::interpreter::MakeRandomInput(bitgen, arg.getType()));
  }

  llvm::outs() << "@" << function.getName().str() << "()\n";

  if (!args.empty()) {
    llvm::outs() << "Arguments:\n";
    for (const auto &arg : args) {
      llvm::outs() << arg.ToString() << "\n";
    }
  }

  mlir::SymbolTable symbol_table{module};
  auto results =
      mlir::interpreter::RunInterpreter(symbol_table, function, args);
  if (!mlir::succeeded(results)) {
    llvm::errs() << "Interpreter failed\n";
    return mlir::failure();
  }

  if (!results->empty()) {
    llvm::outs() << "Results:\n";
    for (const auto &result : *results) {
      llvm::outs() << result.ToString() << "\n";
    }
  }

  if (!args.empty()) {
    llvm::outs() << "Arguments after execution:\n";
    for (const auto &arg : args) {
      llvm::outs() << arg.ToString() << "\n";
    }
  }

  return mlir::success();
}

int main(int argc, char *argv[]) {
  int dummyArgc = 1;
  tsl::port::InitMain(argv[0], &dummyArgc, &argv);
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect, mlir::lmhlo_gpu::LmhloGpuDialect,
                  mlir::gml_st::GmlStDialect, mlir::thlo::THLODialect>();

  mlir::MLIRContext context(registry);
  auto m = parseMLIRInput(options.input_filename, true, &context);
  auto module = llvm::cast<mlir::ModuleOp>(**m);

  if (options.run_all_functions) {
    bool all_succeeded = true;
    module.walk([&](mlir::func::FuncOp function) {
      all_succeeded &= Run(module, function).succeeded();
    });
    if (!all_succeeded) {
      return 1;
    }
  } else {
    auto main = module.lookupSymbol("main");
    if (!main) {
      llvm::errs() << "no main function found.\n";
      return 1;
    }
    if (!Run(module, llvm::cast<mlir::func::FuncOp>(main)).succeeded()) {
      return 1;
    }
  }
  return 0;
}
