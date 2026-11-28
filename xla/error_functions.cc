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

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {

absl::Status AddStatus(absl::Status prior, absl::string_view context) {
  CHECK(!prior.ok());
  return absl::Status{prior.code(),
                      absl::StrCat(context, ": ", prior.message())};
}

absl::Status AppendStatus(absl::Status prior, absl::string_view context) {
  CHECK(!prior.ok());
  return absl::Status{prior.code(),
                      absl::StrCat(prior.message(), ": ", context)};
}

}  // namespace xla
