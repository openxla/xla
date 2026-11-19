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

#include "xla/tsl/platform/scoped_temp_dir.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "absl/status/status_matchers.h"
#include "tsl/platform/path.h"
#include "xla/tsl/platform/env.h"

namespace tsl {
namespace testing {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;

TEST(ScopedTempDirTest, CreatesEmptyDirectory) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  EXPECT_THAT(Env::Default()->IsDirectory(dir.path()), IsOk());
}

TEST(ScopedTempDirTest, PathsAreUnique) {
  ScopedTempDir a;
  ScopedTempDir b;
  EXPECT_NE(a.path(), b.path());
}

TEST(ScopedTempDirTest, DeletesContentsOnDestruction) {
  std::string path;
  {
    ScopedTempDir dir;
    path = dir.path();
    std::string nested = io::JoinPath(path, "sub", "file.txt");
    ASSERT_THAT(
        Env::Default()->RecursivelyCreateDir(std::string(io::Dirname(nested))),
        IsOk());
    ASSERT_THAT(WriteStringToFile(Env::Default(), nested, "x"), IsOk());
  }
  EXPECT_THAT(Env::Default()->FileExists(path), Not(IsOk()));
}

TEST(ScopedTempDirTest, MoveConstructorTransfersOwnership) {
  std::string path;
  {
    ScopedTempDir dir;
    path = dir.path();
    ScopedTempDir moved_to(std::move(dir));
    EXPECT_TRUE(dir.path().empty());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(moved_to.path(), path);
    EXPECT_THAT(Env::Default()->FileExists(path), IsOk());
  }
  EXPECT_THAT(Env::Default()->FileExists(path), Not(IsOk()));
}

TEST(ScopedTempDirTest, MoveAssignmentTransfersOwnership) {
  std::string path;
  {
    ScopedTempDir moved_to;
    {
      ScopedTempDir dir;
      path = dir.path();
      moved_to = std::move(dir);
      EXPECT_TRUE(dir.path().empty());  // NOLINT(bugprone-use-after-move)
    }
    // `dir` is gone, but `moved_to` still owns the directory.
    EXPECT_THAT(Env::Default()->FileExists(path), IsOk());
  }
  EXPECT_THAT(Env::Default()->FileExists(path), Not(IsOk()));
}

TEST(ScopedTempDirTest, MoveAssignmentDeletesPreviousDirectory) {
  ScopedTempDir dir;
  std::string old_path = dir.path();
  dir = ScopedTempDir();
  EXPECT_NE(dir.path(), old_path);
  EXPECT_THAT(Env::Default()->FileExists(old_path), Not(IsOk()));
  EXPECT_THAT(Env::Default()->IsDirectory(dir.path()), IsOk());
}

TEST(ScopedTempDirTest, CreateReturnsOwnedDirectory) {
  std::string path;
  {
    ASSERT_OK_AND_ASSIGN(ScopedTempDir dir, ScopedTempDir::Create());
    path = dir.path();
    EXPECT_THAT(Env::Default()->IsDirectory(path), IsOk());
  }
  EXPECT_THAT(Env::Default()->FileExists(path), Not(IsOk()));
}

}  // namespace
}  // namespace testing
}  // namespace tsl
