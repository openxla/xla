/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/resource_loader.h"

#include <cstdlib>
#include <string>

#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace tsl {

std::string GetDataDependencyFilepath(const std::string& relative_path) {
  if (kIsOpenSource) {
    using bazel::tools::cpp::runfiles::Runfiles;
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
    if (runfiles == nullptr) {
      LOG(FATAL) << "Could not initialize runfiles: " << error.c_str();
    }

    std::string full_path = runfiles->Rlocation(relative_path);
    if (full_path.empty()) {
      LOG(FATAL) << "Could not find runfile " << relative_path;
    }

    return full_path;
  }

  // TODO(ddunleavy): replace this with `TensorFlowSrcRoot()` from `test.h`.
  const char* srcdir = std::getenv("TEST_SRCDIR");
  if (!srcdir) {
    LOG(FATAL) << "Environment variable TEST_SRCDIR unset!";  // Crash OK
  }

  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (!workspace) {
    LOG(FATAL) << "Environment variable TEST_WORKSPACE unset!";  // Crash OK
  }

  return io::JoinPath(srcdir, workspace, "third_party", relative_path);
}

}  // namespace tsl
