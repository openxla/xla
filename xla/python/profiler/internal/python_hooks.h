/* Copyright 2020 The OpenXLA Authors.

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
#ifndef XLA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
#define XLA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_

#include <Python.h>
#if PY_VERSION_HEX < 0x030b0000
#include <frameobject.h>
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/platform/macros.h"

#ifdef Py_GIL_DISABLED
#include "absl/synchronization/mutex.h"
#endif  // Py_GIL_DISABLED

namespace xla {
namespace profiler {

struct PythonHooksOptions {
  bool enable_trace_python_function = false;
  bool enable_python_traceme = true;
  bool end_to_end_mode = false;
  // Incomplete events are defined as those python calls which we only see
  // either start or end, but not both. If we want to include them in the final
  // result, profiler start, end time are used respectively to the absent
  // timestamps.
  bool include_incomplete_events = true;
};

struct TraceEventInfo {
  std::string name;
  uint64_t start_time_ns;
  uint64_t end_time_ns;
};

struct PerThreadConsumeData {
  int64_t thread_id;
  std::vector<TraceEventInfo> events;
};

struct PythonTraceEntry {
  PythonTraceEntry(uint64_t start, uint64_t end, const std::string* name)
      : start_time_ns(start), end_time_ns(end), event_name(name) {}

  absl::string_view Name() const { return *event_name; }

  uint64_t start_time_ns;
  uint64_t end_time_ns;
  // Interned name owned by the PythonHookContext that recorded this entry.
  const std::string* event_name;
};

struct PerThreadEvents {
  PerThreadEvents() = default;
  PerThreadEvents(PerThreadEvents&&) = default;
  PerThreadEvents& operator=(PerThreadEvents&&) = default;
  PerThreadEvents(const PerThreadEvents&) = delete;
  PerThreadEvents& operator=(const PerThreadEvents&) = delete;

  std::deque<PythonTraceEntry> completed;
  std::stack<PythonTraceEntry> active;
  // Track C Functions call in its own stack.
  std::stack<PythonTraceEntry> active_c;
};

class PythonHooks;

class PythonHookContext {
 public:
  ~PythonHookContext() = default;
  void Finalize(tensorflow::profiler::XSpace* space);
  std::vector<PerThreadConsumeData> Consume();

  friend class ::xla::profiler::PythonHooks;

 private:
  void Start(const PythonHooksOptions& option);
  void Stop();
  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg);
  void CollectData(tensorflow::profiler::XPlane* raw_plane);

  static bool SetProfilerInAllThreads();
  static bool ClearProfilerInAllThreads();
  static bool RegisterAtexitCallback();
  static PyObject* PyProfileCallback(PyObject* self, PyObject* const* args,
                                     Py_ssize_t nargs);
  static PyObject* PyAtexitCallback(PyObject* self, PyObject* args);

  void operator=(const PythonHookContext&) = delete;
  void operator=(PythonHookContext&&) = delete;

  // The thread id to entries map, Note: by convention the thread id is
  // int64_t to be consistent with cpu tracer when serialize to Xspace.
  struct EntryShard {
    // If the GIL is enabled, this data structure is protected by the GIL.
    // Otherwise, it is protected by mu.
#ifdef Py_GIL_DISABLED
    absl::Mutex mu;
#endif  // Py_GIL_DISABLED
    absl::flat_hash_map<int64_t, PerThreadEvents> entries;

    // Event names are formatted once per distinct function and interned here,
    // so the profile callback only does a lookup. The tables keep their code
    // objects and C function modules alive until Stop(), so that a freed
    // object's address cannot be reused by a different function while it is
    // a key. Unlike function objects, these don't reference runtime values
    // such as device buffers. The strings live until the context is
    // destroyed; std::deque keeps pointers to them stable.
    absl::flat_hash_map<PyCodeObject*, const std::string*> code_names;
    absl::flat_hash_map<std::pair<PyMethodDef*, PyObject*>, const std::string*>
        c_function_names;
    std::deque<std::string> names;
  };

  // Returns the interned event name for the function, formatting it the first
  // time the function is seen. Must be called with the shard's lock (the GIL,
  // or `shard.mu` in free-threaded builds). The returned pointer stays valid
  // until the context is destroyed.
  static const std::string* InternName(EntryShard& shard,
                                       PyCodeObject* py_code_object);
  static const std::string* InternName(EntryShard& shard,
                                       PyCFunctionObject* py_c_function);
  // Drops the references held by the interning tables. Requires the GIL.
  void ReleaseInternedObjects();

#ifdef Py_GIL_DISABLED
  static constexpr size_t kNumEntryShards = 16;
#else   // Py_GIL_DISABLED
  static constexpr size_t kNumEntryShards = 1;
#endif  // Py_GIL_DISABLED
  std::array<EntryShard, kNumEntryShards> entry_shards_;
  uint64_t start_timestamp_ns_;
  PythonHooksOptions options_;
  std::atomic<bool> stopped_ = false;
  // In end to end mode, Python get uninitialized before Stop()/Finalize(), we
  // need to buffer the result.
  std::optional<tensorflow::profiler::XPlane> end_to_end_xplane_;
};

// Singleton for tracing python function calls.
class PythonHooks {
 public:
  static PythonHooks* GetSingleton();

  void Start(const PythonHooksOptions& option) {
    if (active_context_) {
      return;
    }
    active_context_ = std::make_unique<PythonHookContext>();
    active_context_->Start(option);
  }

  std::unique_ptr<PythonHookContext> Stop() {
    if (e2e_context_) {
      auto* e2e_context = e2e_context_;
      e2e_context_ = nullptr;
      return absl::WrapUnique(e2e_context);
    }

    if (!active_context_) {
      return nullptr;
    }
    active_context_->Stop();
    std::unique_ptr<PythonHookContext> output = std::move(active_context_);
    active_context_.reset();
    return output;
  }

  std::vector<PerThreadConsumeData> Consume() {
    if (!active_context_) {
      return {};
    }
    return active_context_->Consume();
  }

  friend class ::xla::profiler::PythonHookContext;

 private:
  void ProfileSlow(PyFrameObject* frame, absl::string_view event,
                   PyObject* arg);

  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg) {
    if (TF_PREDICT_TRUE(active_context_)) {
      active_context_->ProfileFast(frame, what, arg);
    }
  }

  static void set_e2e_context(PythonHookContext* e2e_context) {
    e2e_context_ = e2e_context;
  }

  static int ProfileFunction(PyObject* obj, PyFrameObject* frame, int what,
                             PyObject* arg);

  // active_context_ are accessed when GIL is held, therefore no race
  // conditions.
  std::unique_ptr<PythonHookContext> active_context_;
  static PythonHookContext* e2e_context_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
