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

#include "xla/tsl/platform/scoped_temp_dir.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/env.h"

namespace tsl {
namespace testing {

absl::StatusOr<std::string> ScopedTempDir::CreateDirectory() {
  std::string path;
  if (!Env::Default()->LocalTempFilename(&path)) {
    return absl::InternalError("Failed to create temporary directory.");
  }
  ABSL_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(path));
  return path;
}

absl::StatusOr<ScopedTempDir> ScopedTempDir::Create() {
  ABSL_ASSIGN_OR_RETURN(std::string path, CreateDirectory());
  return ScopedTempDir(std::move(path));
}

ScopedTempDir::ScopedTempDir() {
  absl::StatusOr<std::string> temp_dir = CreateDirectory();
  CHECK_OK(temp_dir.status());
  path_ = *std::move(temp_dir);
}

ScopedTempDir::ScopedTempDir(ScopedTempDir&& other) noexcept
    : path_(std::move(other.path_)) {
  other.path_.clear();
}

ScopedTempDir& ScopedTempDir::operator=(ScopedTempDir&& other) noexcept {
  if (this != &other) {
    Cleanup();
    path_ = std::move(other.path_);
    other.path_.clear();
  }
  return *this;
}

ScopedTempDir::~ScopedTempDir() { Cleanup(); }

void ScopedTempDir::Cleanup() {
  if (path_.empty()) {
    return;
  }
  int64_t undeleted_files = 0;
  int64_t undeleted_dirs = 0;
  absl::Status status = Env::Default()->DeleteRecursively(
      path_, &undeleted_files, &undeleted_dirs);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to delete temporary directory " << path_ << ": "
                 << status;
  }
  path_.clear();
}

}  // namespace testing
}  // namespace tsl
