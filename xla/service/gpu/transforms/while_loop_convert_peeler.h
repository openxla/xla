/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_WHILE_LOOP_CONVERT_PEEPER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_WHILE_LOOP_CONVERT_PEEPER_H_

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/value_range.h"
namespace xla::gpu {

/*

While loop convert peeler optimization

This pass peels off the convert instructions from the while loop body.

We look for the following pattern in the while body:

```
data = f32[8, 1024] parameter(0)
iter = 0
while (iter < 8):
  slice = f32[1, 1024] data[iter:iter+1, :]
  convert = bf16[1, 1024] convert(slice)
  ...
  ROOT data
```

That is, the data is accessed in the while loop, one slice at a time. These
slices cover the entire data buffer, and they are immediately converted to bf16
inside the loop body. There are no other uses of the data buffer or the sliced
operation, except the convert and the root instruction. In this case, we can
peel the convert outside the while loop:

```
data = f32[8, 1024] parameter(0)
converted_data = bf16[8, 1024] convert(data)
iter = 0
while (iter < 8):
  slice = bf16[1, 1024] converted_data[iter:iter+1, :]
  ...
  ROOT data
```

This helps reduce the cost of slicing operations in the while loop.

The conditions for peeling are:

 1. The dynamic-slice operation must have exactly one variable index (lets call
that index k).

 2. The dynamic-slice operation must be a contiguous slice (i.e. for all indices
i>k, we should have that dimension_of_buffer[i]=slice_size[i] and offset[i]=0.)

 3. The dynamic-slice operation must cover the entire buffer. This means that

   3a. The variable index k must have the monotonic range [0, dimension[k]) and
each of those values must be taken once (step=1).

   3b. For all indices i < k, we should have that dimension[i]=slice_size[i]=1
and offset[i]=0.

 4. Obviously, the pattern: dynamic-slice followed by a convert operation
(allowing no-op reshapes).

 5. The buffer should be from the parameter, unmodified, and the buffer should
go to the results, unmodified.

 6. There should be exactly two uses of the buffer inside the while loop:
convert operation and the root instruction.

If these conditions are met, then we can peel the convert outside the while
loop:

 1. We change the while loop body/condition signature to accept the input of
destination type (from the convert operation).

 2. We add the convert operation outside the while loop.

 3. Any user of the previous results of while operation are adjusted.

*/

class WhileLoopConvertPeeler : public HloModulePass {
 private:
  // Store metadata about a buffer, for which the convert has to be peeled off.
  struct BufferInfo {
    // The dynamic-slice instruction in the while body. Must be non-null.
    HloDynamicIndexInstruction* dynamic_index_instruction = nullptr;
    // The buffer of dynamic-slice instruction in the while body. Must be
    // non-null.
    HloInstruction* body_buffer = nullptr;
    // The convert instruction inside while body whose operand is the result of
    // dynamic-slice instruction. Must be non-null.
    HloInstruction* body_convert = nullptr;
    // The root instruction of the while body. Must be non-null.
    HloInstruction* body_root = nullptr;
    // The index of the buffer in the parameter tuple of the while instruction.
    int64_t while_tuple_buffer_idx = -1;
    // The get-tuple-element instruction that uses the buffer after the while
    // loop. This could be nullptr, if the buffer is not used after the while
    // loop.
    HloInstruction* while_gte_user_after_loop = nullptr;

    // For VLOG and debugging.
    std::string ToString(int indent = 0) const;
  };

  // This struct stores the convert peeling information for a while operation.
  // Each while operation can have multiple buffers that are eligible for
  // convert peeling, hence it has a vector of `BufferInfos`.
  struct ConvertInfo {
    // List of buffer infos to peel the convert of.
    absl::InlinedVector<BufferInfo, 4> buffer_infos;

    // The new shape of the while operation after peeling the convert. This will
    // be invalid shape, until the new shape is deduced.
    Shape new_while_shape;

    // For VLOG and debugging.
    std::string ToString(int indent = 0) const;
  };

  // Map of while operation to its convert peeling information.
  using ConvertInfoMap = absl::flat_hash_map<HloInstruction*, ConvertInfo>;

  // When the while instruction is a root instruction to the computation, just
  // peeling the convert will change the return datatype of the computation. To
  // prevent this, we should set the root instruction to a tuple of
  // get-tuple-elements over the results of while operation.
  absl::Status CreateTupleGetTupleElementForRoot(HloInstruction* while_op,
                                                 ConvertInfo& convert_info);

  // Deduce the new shape of the while operation, based on convert info. This
  // populates the field `new_while_shape`.
  void DeduceNewWhileShape(HloInstruction* while_op, ConvertInfo& convert_info);

  // Apply fixes to the while body, condition and init computation based on the
  // convert infos.
  void FixWhileBody(HloInstruction* while_op, ConvertInfo& convert_info);
  void FixWhileCond(HloInstruction* while_op, ConvertInfo& convert_info);
  absl::Status FixWhileInit(HloInstruction* while_op,
                            ConvertInfo& convert_info);

  // Collect the ConvertInfos. This is a analysis-only function, that does not
  // modify the HLO.
  ConvertInfoMap CollectConvertInfos(
      HloModule* module,
      absl::flat_hash_map<const HloInstruction*, Range>& known_ranges,
      absl::flat_hash_set<HloInstruction*>& while_ops);

 public:
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
  absl::string_view name() const override {
    return "while-loop-convert-peeler";
  }
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_WHILE_LOOP_CONVERT_PEEPER_H_
