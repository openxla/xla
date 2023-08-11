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

#ifndef XLA_PYTHON_IDEMPOTENT_SERIALIZER_H_
#define XLA_PYTHON_IDEMPOTENT_SERIALIZER_H_

#include <functional>
#include <string>

#include "tsl/platform/protobuf.h"

namespace xla {

// IdempotentSerializer: given a proto, serialize it in an idempotent manner,
// Idempotent behavior is in the context of the binary that the serializer is
// part of, i.e., may not be preserved across releases. The ordering of repeated
// fields and map fields in the proto will not matter.
class IdempotentSerializer {
 public:
  using SerializerFn =
      std::function<std::string(const tsl::protobuf::Message&)>;

  // Register a serializer function (a serializer function takes a proto and
  // returns the serialized representation). Only one serializer function is
  // permitted, so Register can be called at most once. Register returns true if
  // registration succeeded, false otherwise.
  //
  // It is expected that Register will be called during program initialization.
  // It does not offer thread safety.
  static bool Register(SerializerFn serializer_fn) {
    if (is_registered()) return false;
    serializer_fn_ = new SerializerFn(serializer_fn);
    return true;
  }

  // Return true if Register was successfully called to register a serializer,
  // else return false.
  static bool is_registered() { return serializer_fn_ != nullptr; }

  // Call the registered serializer function. If no function has been
  // registered, return empty string. There is no protection against races
  // due to concurrent calls to Register and Serialize.
  static std::string Serialize(const tsl::protobuf::Message& message) {
    return is_registered() ? (*serializer_fn_)(message) : "";
  }

 private:
  static SerializerFn* serializer_fn_;
};

}  // namespace xla

#endif  // XLA_PYTHON_IDEMPOTENT_SERIALIZER_H_
