/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/error_functions.h"

#include <gtest/gtest.h>

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

TEST(ErrorFunctionsTest, FormattedErrors) {
  absl::Status s1 = InvalidArgument("Value %d is invalid: %s", 42, "foo");
  EXPECT_TRUE(absl::IsInvalidArgument(s1));
  EXPECT_EQ(s1.message(), "Value 42 is invalid: foo");

  absl::Status s2 = Internal("Internal failure: %s", "disk full");
  EXPECT_TRUE(absl::IsInternal(s2));
  EXPECT_EQ(s2.message(), "Internal failure: disk full");

  absl::Status s3 = NotFound("Key %s not found", "bar");
  EXPECT_TRUE(absl::IsNotFound(s3));
  EXPECT_EQ(s3.message(), "Key bar not found");

  absl::Status s4 = Unimplemented("Feature %s is unimplemented", "xyz");
  EXPECT_TRUE(absl::IsUnimplemented(s4));
  EXPECT_EQ(s4.message(), "Feature xyz is unimplemented");

  absl::Status s5 =
      ResourceExhausted("Out of memory: %d bytes requested", 1024);
  EXPECT_TRUE(absl::IsResourceExhausted(s5));
  EXPECT_EQ(s5.message(), "Out of memory: 1024 bytes requested");

  absl::Status s6 = FailedPrecondition("State is %s", "uninitialized");
  EXPECT_TRUE(absl::IsFailedPrecondition(s6));
  EXPECT_EQ(s6.message(), "State is uninitialized");

  absl::Status s7 = Unavailable("Service %s unavailable", "auth");
  EXPECT_TRUE(absl::IsUnavailable(s7));
  EXPECT_EQ(s7.message(), "Service auth unavailable");

  absl::Status s8 = Aborted("Operation %s aborted", "op1");
  EXPECT_TRUE(absl::IsAborted(s8));
  EXPECT_EQ(s8.message(), "Operation op1 aborted");

  absl::Status s9 = Cancelled("Request %d cancelled", 123);
  EXPECT_TRUE(absl::IsCancelled(s9));
  EXPECT_EQ(s9.message(), "Request 123 cancelled");

  absl::Status s10 = DeadlineExceeded("Timeout after %d ms", 5000);
  EXPECT_TRUE(absl::IsDeadlineExceeded(s10));
  EXPECT_EQ(s10.message(), "Timeout after 5000 ms");

  absl::Status s11 = Unknown("Unknown code %d", -1);
  EXPECT_TRUE(absl::IsUnknown(s11));
  EXPECT_EQ(s11.message(), "Unknown code -1");

  absl::Status s12 = InvalidArgument("No format args");
  EXPECT_TRUE(absl::IsInvalidArgument(s12));
  EXPECT_EQ(s12.message(), "No format args");
}

template <typename... Args>
absl::Status ForwardFormatSpec(const absl::FormatSpec<Args...>& format,
                               const Args&... args) {
  return InvalidArgument(format, args...);
}

TEST(ErrorFunctionsTest, ForwardedFormatSpec) {
  absl::Status s = ForwardFormatSpec("Forwarded %s %d", "msg", 7);
  EXPECT_TRUE(absl::IsInvalidArgument(s));
  EXPECT_EQ(s.message(), "Forwarded msg 7");
}

TEST(ErrorFunctionsTest, StrCatErrors) {
  absl::Status s1 = InvalidArgumentStrCat("Value ", 42, " is invalid: ", "foo");
  EXPECT_TRUE(absl::IsInvalidArgument(s1));
  EXPECT_EQ(s1.message(), "Value 42 is invalid: foo");

  absl::Status s2 = InternalStrCat("Internal failure: ", "disk full");
  EXPECT_TRUE(absl::IsInternal(s2));
  EXPECT_EQ(s2.message(), "Internal failure: disk full");

  absl::Status s3 = UnimplementedStrCat("Feature ", "xyz", " is unimplemented");
  EXPECT_TRUE(absl::IsUnimplemented(s3));
  EXPECT_EQ(s3.message(), "Feature xyz is unimplemented");

  absl::Status s4 =
      ResourceExhaustedStrCat("Out of memory: ", 1024, " bytes requested");
  EXPECT_TRUE(absl::IsResourceExhausted(s4));
  EXPECT_EQ(s4.message(), "Out of memory: 1024 bytes requested");

  // Also test other StrCat errors now supported uniformly
  absl::Status s5 = NotFoundStrCat("Item ", 99, " not found");
  EXPECT_TRUE(absl::IsNotFound(s5));
  EXPECT_EQ(s5.message(), "Item 99 not found");

  absl::Status s6 = FailedPreconditionStrCat("state=", 1);
  EXPECT_TRUE(absl::IsFailedPrecondition(s6));
  EXPECT_EQ(s6.message(), "state=1");

  absl::Status s7 = InvalidArgumentStrCat();
  EXPECT_TRUE(absl::IsInvalidArgument(s7));
  EXPECT_EQ(s7.message(), "");
}

TEST(ErrorFunctionsTest, AddStatusAndAppendStatus) {
  absl::Status prior = absl::InvalidArgumentError("invalid shape");

  absl::Status with_context = AddStatus(prior, "failed to compile");
  EXPECT_TRUE(absl::IsInvalidArgument(with_context));
  EXPECT_EQ(with_context.message(), "failed to compile: invalid shape");

  absl::Status appended = AppendStatus(prior, "additional details");
  EXPECT_TRUE(absl::IsInvalidArgument(appended));
  EXPECT_EQ(appended.message(), "invalid shape: additional details");
}

TEST(ErrorFunctionsTest, SourceLocationCaptured) {
  int line = __LINE__ + 1;
  absl::Status status = InvalidArgument("test error %d", 1);
  auto locs = status.GetSourceLocations();
  ASSERT_FALSE(locs.empty());
  EXPECT_EQ(locs[0].line(), line);

  line = __LINE__ + 1;
  absl::Status cat_status = InvalidArgumentStrCat("cat error ", 2);
  auto cat_locs = cat_status.GetSourceLocations();
  ASSERT_FALSE(cat_locs.empty());
  EXPECT_EQ(cat_locs[0].line(), line);
}

}  // namespace
}  // namespace xla
