/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_LOGGING_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_LOGGING_H_

// IWYU pragma: private, include "tsl/platform/logging.h"
// IWYU pragma: friend third_party/tensorflow/tsl/platform/logging.h

#include <cstdint>
#include <sstream>

namespace tsl {

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the maximum log level for VLOG statements.
  // E.g., if MaxVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64_t MaxVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable TF_CPP_VMODULE contains foo=3 and fname is
  // foo.cc and lvl is <= 3, this will return true. It will also return true if
  // the level is lower or equal to TF_CPP_MAX_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char* fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

}  // namespace internal

#ifdef IS_MOBILE_PLATFORM
// Turn VLOG off when under mobile devices for considerations of binary size.
#define VLOG_IS_ON(lvl) ((lvl) <= 0)
#else
// Otherwise, set TF_CPP_MAX_VLOG_LEVEL environment to update minimum log level
// of VLOG, or TF_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define VLOG_IS_ON(lvl)                                              \
  (([](int level, const char* fname) {                               \
    static const bool vmodule_activated =                            \
        ::tsl::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                        \
  })(lvl, __FILE__))
#endif

#define VLOG(level) LOG_IF(INFO, VLOG_IS_ON(level)).WithVerbosity(level)

// Change verbose level of pre-defined files if environment
// variable `env_var` is defined. This is currently a no op.
void UpdateLogVerbosityIfDefined(const char* env_var);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_LOGGING_H_
