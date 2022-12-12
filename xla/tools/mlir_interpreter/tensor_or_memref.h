/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_TENSOR_OR_MEMREF_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_TENSOR_OR_MEMREF_H_

#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace interpreter {

// Represents a view into a physical buffer.
struct BufferView {
  int64_t offset;
  llvm::SmallVector<int64_t> sizes;    // [10, 11, 12]
  llvm::SmallVector<int64_t> strides;  // [132, 12, 1]

  int64_t rank() const { return sizes.size(); }

  int64_t num_elements() const {
    size_t n = 1;
    for (auto size : sizes) n *= size;
    return n;
  }

  struct LogicalIndexView {
    struct Iterator {
      using iterator_category = std::forward_iterator_tag;
      using value_type = llvm::SmallVector<int64_t>;
      using difference_type = std::ptrdiff_t;
      using pointer = llvm::SmallVector<int64_t>*;
      using reference = llvm::SmallVector<int64_t>&;

      const llvm::SmallVector<int64_t>& operator*() const {
        return view_indices;
      }
      const llvm::SmallVector<int64_t>* operator->() const {
        return &view_indices;
      }

      Iterator& operator++() {
        auto index_it = view_indices.rbegin();
        auto size_it = view->sizes.rbegin();

        for (auto e = view_indices.rend(); index_it != e;
             ++index_it, ++size_it) {
          ++*index_it;
          if (*index_it < *size_it) {
            return *this;
          }
          *index_it = 0;
        }

        view_indices.clear();
        view_indices.push_back(-1);
        return *this;
      }

      Iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }

      bool operator==(const Iterator& other) const {
        return view_indices == other.view_indices;
      }

      bool operator!=(const Iterator& other) const { return !(*this == other); }

     private:
      friend class LogicalIndexView;

      Iterator(const BufferView* view, llvm::SmallVector<int64_t> indices)
          : view(view), view_indices(std::move(indices)) {}

      const BufferView* view;
      llvm::SmallVector<int64_t> view_indices;
    };

    Iterator begin() const {
      if (view->num_elements() == 0) return end();
      return {view, llvm::SmallVector<int64_t>(view->rank())};
    }
    Iterator end() const { return {view, {-1}}; }

   private:
    friend class BufferView;

    explicit LogicalIndexView(const BufferView* view) : view(view) {}

    const BufferView* view;
  };

  struct PhysicalIndexView {
    struct Iterator {
      using iterator_category = std::forward_iterator_tag;
      using value_type = int64_t;
      using difference_type = std::ptrdiff_t;
      using pointer = int64_t*;
      using reference = int64_t&;

      const int64_t& operator*() const {
        buffer_index = view->physical_index(*iter);
        return buffer_index;
      }

      const int64_t* operator->() const { return &(**this); }

      Iterator& operator++() {
        ++iter;
        return *this;
      }

      Iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }

      bool operator==(const Iterator& other) const {
        return iter == other.iter;
      }

      bool operator!=(const Iterator& other) const { return !(*this == other); }

     private:
      friend class PhysicalIndexView;

      Iterator(const BufferView* view, LogicalIndexView::Iterator iter)
          : view(view), buffer_index(0), iter(iter) {}

      const BufferView* view;
      mutable int64_t buffer_index;
      LogicalIndexView::Iterator iter;
    };

    Iterator begin() const { return {view, view->indices().begin()}; }
    Iterator end() const { return {view, view->indices().end()}; }

   private:
    friend class BufferView;

    explicit PhysicalIndexView(const BufferView* view) : view(view) {}
    const BufferView* view;
  };

  int64_t physical_index(llvm::ArrayRef<int64_t> view_indices) const;
  LogicalIndexView indices() const { return LogicalIndexView{this}; }
  PhysicalIndexView physical_indices() const { return PhysicalIndexView{this}; }

  bool InBounds(llvm::ArrayRef<int64_t> view_indices) const;
  static SmallVector<int64_t> DefaultStrides(ArrayRef<int64_t> sizes);
};

template <typename T>
struct Buffer {
  llvm::SmallVector<T> storage;

  explicit Buffer(size_t size) : storage(size) {}
};

template <typename T>
struct TensorOrMemref {
  static TensorOrMemref<T> Empty(ArrayRef<int64_t> sizes) {
    BufferView dummy{0, SmallVector<int64_t>(sizes), {}};
    return EmptyLike(dummy);
  }

  static TensorOrMemref<T> EmptyLike(const BufferView& view) {
    return {std::make_shared<Buffer<T>>(view.num_elements()),
            {0, view.sizes, BufferView::DefaultStrides(view.sizes)}};
  }

  TensorOrMemref<T> Clone() const {
    auto out = EmptyLike(view);
    for (auto [src_index, dst] :
         llvm::zip(view.physical_indices(), out.buffer->storage)) {
      dst = buffer->storage[src_index];
    }
    return out;
  }

  const T& at(ArrayRef<int64_t> indices) const {
    return buffer->storage[view.physical_index(indices)];
  }

  T& at(ArrayRef<int64_t> indices) {
    return buffer->storage[view.physical_index(indices)];
  }

  bool operator==(const TensorOrMemref& other) const {
    if (other.view.sizes != view.sizes) return false;
    for (const auto& indices : view.indices()) {
      if (at(indices) != other.at(indices)) return false;
    }
    return true;
  }

  std::shared_ptr<Buffer<T>> buffer;
  BufferView view;
};

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_TENSOR_OR_MEMREF_H_
