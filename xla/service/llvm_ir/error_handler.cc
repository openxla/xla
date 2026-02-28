/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/error_handler.h"

#include <utility>

#include "absl/base/call_once.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ErrorHandling.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

namespace {
using Handler = XlaScopedFatalErrorHandler::Handler;

class LLVMFatalErrorDispatcher {
 public:
  // Registers the master handler with LLVM exactly once.
  static void EnsureInstalled() {
    static absl::once_flag once;
    absl::call_once(once, []() {
      llvm::install_fatal_error_handler(ErrorHandler, nullptr);
    });
  }

  // Sets the handler for the CURRENT thread only.
  // Returns the previous handler for this thread.
  static Handler SetThreadHandler(Handler handler) {
    std::swap(thread_local_handler_, handler);
    return handler;
  }

 private:
  static void ErrorHandler(void* /*user_data*/, const char* reason,
                           bool /*diag*/) {
    if (thread_local_handler_ && *thread_local_handler_) {
      (*thread_local_handler_)(reason);
    } else {
      LOG(ERROR) << "LLVM ERROR: " << reason;
    }
    // We crash here unconditionally.
    // If the handler was to return LLVM will crash anyway.
    LOG(QFATAL) << tsl::CurrentStackTrace();
  }

  static thread_local Handler thread_local_handler_;
};

thread_local Handler LLVMFatalErrorDispatcher::thread_local_handler_;

}  // namespace

XlaScopedFatalErrorHandler::XlaScopedFatalErrorHandler(Handler handler) {
  LLVMFatalErrorDispatcher::EnsureInstalled();
  prev = LLVMFatalErrorDispatcher::SetThreadHandler(std::move(handler));
}

XlaScopedFatalErrorHandler::~XlaScopedFatalErrorHandler() {
  LLVMFatalErrorDispatcher::SetThreadHandler(std::move(prev));
}

}  // namespace xla
