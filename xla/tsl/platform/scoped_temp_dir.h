/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_SCOPED_TEMP_DIR_H_
#define XLA_TSL_PLATFORM_SCOPED_TEMP_DIR_H_

#include <string>

#include "absl/status/statusor.h"

namespace tsl {
namespace testing {

// Creates a uniquely named, empty temporary directory on construction and
// recursively deletes it on destruction.
//
// Typical usage in tests:
//
//   ScopedTempDir temp_dir;
//   std::string path = tsl::io::JoinPath(temp_dir.path(), "file.txt");
//
// The directory is created under the process's local temporary directory
// (see Env::LocalTempFilename), which honors TEST_TMPDIR when set.
//
// Instances are movable but not copyable.
// A moved-from instance owns nothing and its
// destructor is a no-op.
class ScopedTempDir {
 public:
  // Creates the directory. Check-fails if it cannot be created.
  ScopedTempDir();

  // Creates the directory.
  static absl::StatusOr<ScopedTempDir> Create();

  ScopedTempDir(ScopedTempDir&& other) noexcept;
  ScopedTempDir& operator=(ScopedTempDir&& other) noexcept;
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  // Recursively deletes the directory.
  ~ScopedTempDir();

  // The absolute path of the directory. Empty for a moved from instance.
  const std::string& path() const { return path_; }

 private:
  explicit ScopedTempDir(std::string path) : path_(std::move(path)) {}

  // Creates a uniquely named directory and returns its path. This is the only
  // place that touches the filesystem on creation; both the constructor and
  // Create() delegate to it.
  static absl::StatusOr<std::string> CreateDirectory();

  // Deletes the directory if owned and resets `path_`.
  void Cleanup();

  std::string path_;
};

}  // namespace testing
}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_SCOPED_TEMP_DIR_H_
