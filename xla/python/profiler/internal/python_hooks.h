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

// Trivially copyable and trivially destructible trace record.
// References acquired on construction are explicitly released in batches by
// PythonHookContext::Consume(), CollectData(), or ~PythonHookContext().
struct PythonTraceEntry {
  // Capture the source/line information for a PyCodeObject object.
  // In eager mode, keeping a reference to PyCodeObject leaks device memory.
  PythonTraceEntry(uint64_t start, uint64_t end, PyCodeObject* py_code_object)
      : start_time_ns(start),
        end_time_ns(end),
        co_filename(py_code_object != nullptr ? py_code_object->co_filename
                                              : nullptr),
        co_name(py_code_object != nullptr ? py_code_object->co_name : nullptr),
        method_name(nullptr),
        co_firstlineno(
            py_code_object != nullptr ? py_code_object->co_firstlineno : 0),
        is_python_call(true) {
    Py_XINCREF(co_filename);
    Py_XINCREF(co_name);
  }

  // Capture the source/line information for a PyCFunctionObject object.
  // In eager mode, keeping a reference to PyCFunctionObject leaks device
  // memory. If m_self is a PyCapsule (e.g. pybind11::cpp_function), retain a
  // reference to the capsule so heap-allocated PyMethodDef::ml_name stays
  // valid until references are released.
  PythonTraceEntry(uint64_t start, uint64_t end,
                   PyCFunctionObject* py_c_function)
      : start_time_ns(start),
        end_time_ns(end),
        m_module(py_c_function != nullptr ? py_c_function->m_module : nullptr),
        c_capsule((py_c_function != nullptr &&
                   py_c_function->m_self != nullptr &&
                   PyCapsule_CheckExact(py_c_function->m_self))
                      ? py_c_function->m_self
                      : nullptr),
        method_name((py_c_function != nullptr && py_c_function->m_ml != nullptr)
                        ? py_c_function->m_ml->ml_name
                        : nullptr),
        co_firstlineno(0),
        is_python_call(false) {
    Py_XINCREF(m_module);
    if (c_capsule != nullptr) {
      Py_INCREF(c_capsule);
    }
  }

  bool IsPythonCall() const { return is_python_call; }

  std::string Name() const;

  uint64_t start_time_ns;
  uint64_t end_time_ns;
  union {
    PyObject* co_filename;  // Active when is_python_call is true.
    PyObject* m_module;     // Active when is_python_call is false.
  };
  union {
    PyObject* co_name;    // Active when is_python_call is true.
    PyObject* c_capsule;  // Active when is_python_call is false.
  };
  const char* method_name = nullptr;
  int co_firstlineno = 0;
  bool is_python_call = false;
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
  ~PythonHookContext();
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
  };

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
