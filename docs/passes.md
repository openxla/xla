# XLA Passes

The following describes existing XLA passes

## XLA Service passes

### algsimp

See also
[`xla::AlgebraicSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/algebraic_simplifier.h).

A pass which performs algebraic simplifications.

### all-gather-bcast-reorder

See also
[`xla::AllGatherBroadcastReorder`](https://github.com/openxla/xla/tree/main/xla/service/all_gather_broadcast_reorder.h).

A pass that reorders `all-gather(broadcast(x)) -> broadcast(all-gather(x))`. The intent is to reduce the size of `all-gather` when possible by doing an `all-gather` on the (smaller) pre-broadcasted data and then applying the broadcast.

### all-gather-combiner

See also
[`xla::AllGatherCombiner`](https://github.com/openxla/xla/tree/main/xla/service/all_gather_combiner.h).

Combines small non-dependent AllGather ops into larger combined AllGather ops. A typical AllGather implementation has a minimum latency-induced time for a AllGather op so a single combined op can be more efficient than many small ones.

### all-reduce-combiner

See also
[`xla::AllReduceCombiner`](https://github.com/openxla/xla/tree/main/xla/service/all_reduce_combiner.h).

Combines small non-dependent AllReduce ops into larger combined AllReduce ops. A typical AllReduce implementation has a minimum latency-induced time for a AllReduce op so a ingle combined op can be more efficient than many small ones.

### all-reduce-contiguous

See also
[`xla::AllReduceContiguous`](https://github.com/openxla/xla/tree/main/xla/service/all_reduce_contiguous.h).

Concatenates `all-reduce` operands together, so the `all-reduce` is performed over a single, contiguous buffer.

### all-reduce-folder

See also
[`xla::AllReduceFolder`](https://github.com/openxla/xla/tree/main/xla/service/all_reduce_folder.h).

A pass that folds an `all-reduce` feeding into another `all-reduce` by expanding the replica groups.

As an example:
```cpp
ar0 = all-reduce(x) replica_groups={{0,1},{2,3},{4,5},{6,7}}
ar1 = all-reduce(all-reduce0) replica_groups={{0,2},{1,3},{4,6},{5,7}}
```

Can be combined into a single `all-reduce`:
```cpp
ar1 = all-reduce(x) replica_groups={{0,1,2,3},{4,5,6,7}}
```

### all-reduce-promotion

See also
[`xla::AllReducePromotion`](https://github.com/openxla/xla/tree/main/xla/service/all_reduce_promotion.h).

Promotes `<from_type>` all-reduce and reduce-scatter to `<to_type>` types.

Arguments           | Type             | Semantics
------------------- | ---------------- | ------------------------------------
`from_to_types`     | `absl::Span<std::pair<PrimitiveType, PrimitiveType> const> ` | Span of pairs of `PrimitiveType`, e.g. `{{U16, U32}, {S16, S32}}`

### all-reduce-reassociate

See also
[`xla::AllReduceReassociate`](https://github.com/openxla/xla/tree/main/xla/service/all_reduce_reassociate.h).

A pass that reassociates `all-reduce` feeding into compatible elementwise operations. As an example: `add(all-reduce(x), all-reduce(y))` will be replaced with `all-reduce(add(x,y))`. Mathematically, this is replacing

```cpp
add(x0, x1, ... xk) + add(y0, y1, ... yk) with
add((x0+y0), (x1+y), ... (xk+yk)
```

i.e., reassociating the reduction operation.

### all-reduce-simp

See also
[`xla::AllReduceSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/all_reduce_simplifier.h).

A pass that detects all-reduces whose inputs are already the same across replicas using the replication analysis, then replaces those all-reduces with local computations. E.g., a sum all-reduce on replicated input will be replaced by a multiply with the replica count.

### all_gather_decomposer

See also
[`xla::AllGatherDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/all_gather_decomposer.h).

AllGatherDecomposer is a pass which converts unsupported all-gathers into dynamic-update-slices and all-reduces.

### ar-crs-combiner

See also
[`xla::ArCrsCombiner`](https://github.com/openxla/xla/tree/main/xla/service/ar_crs_combiner.h).

When the HLO graph contains a cross-module AllReduce (N separate AllReduce ops that share the same channel_id for MPMD partitioning, or 1 AllReduce op for SPMD partitioning), followed by some simple linear operations, followed by a cross-replica AllReduce (also known as cross-replica sum, or CRS), we can combine the CMAR and the CRAR, to use an efficient AllReduce implementation that fully utilizes the interconnect bandwidth.

Such sequences appear in spatially partitioned models (either MPMD or SPMD). This pass must run right after spatial partitioning, when the code is still in a single HLO module.

The steps are:
1. Find CMARs followed by simple ops followed by CRARs.
2. Group CMARs by `channel_id`. They must all be rewritten. For SPMD partitioning, there will only be a single CMAR for each `channel_id`.
3. Prove that the CMAR patterns in each core produce the same result.
4. Eliminate the CMAR, and if it feeds an addition/subtraction, divide the other operand by the number of spatial partitions.
5. Turn the CRAR into an all-core AllReduce.

The pass also handles the case where multiple CMARs lead to the same CRAR, and eliminates all CMARs. This graph:

```cpp
        Y
        |
  X   CMAR_2   Z
  |      \    /
 CMAR_1     +
    \     /
       +
       |
     CRAR
```

gets rewritten to:

```cpp

           Z   num_partitions
            \  /
       Y    div
        \   /
    X     +
     \   /
       +
       |
  all-core AR
```

### async-collective-creator

See also
[`xla::AsyncCollectiveCreator`](https://github.com/openxla/xla/tree/main/xla/service/async_collective_creator.h).

TODO: Pass Description

### batch-dot-simplification

See also
[`xla::BatchDotSimplification`](https://github.com/openxla/xla/tree/main/xla/service/batch_dot_simplification.h).

TODO: Pass Description

### batchnorm_expander

See also
[`xla::BatchNormExpander`](https://github.com/openxla/xla/tree/main/xla/service/batchnorm_expander.h).

TODO: Pass Description

### bf16-mixed-precision-removal

See also
[`xla::FloatNormalization`](https://github.com/openxla/xla/tree/main/xla/service/float_normalization.h).

TODO: Pass Description

### bf16-mixed-precision-removal

See also
[`xla::BFloat16MixedPrecisionRemoval`](https://github.com/openxla/xla/tree/main/xla/service/float_normalization.h).

TODO: Pass Description

### bfloat16-fold

See also
[`xla::BFloat16ConversionFolding`](https://github.com/openxla/xla/tree/main/xla/service/bfloat16_conversion_folding.h).

TODO: Pass Description

### bfloat16-propagation

See also
[`xla::BFloat16Propagation`](https://github.com/openxla/xla/tree/main/xla/service/bfloat16_propagation.h).

TODO: Pass Description

### broadcast_canonicalizer

See also
[`xla::BroadcastCanonicalizer`](https://github.com/openxla/xla/tree/main/xla/service/broadcast_canonicalizer.h).

TODO: Pass Description

### CallInliner

See also
[`xla::CallInliner`](https://github.com/openxla/xla/tree/main/xla/service/call_inliner.h).

TODO: Pass Description

### change-op-data-type

See also
[`xla::ChangeOpDataType`](https://github.com/openxla/xla/tree/main/xla/service/change_op_data_type.h).

TODO: Pass Description

### collective-permute-decomposer

See also
[`xla::CollectivePermuteDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/collective_permute_decomposer.h).

TODO: Pass Description

### collective-pipeliner-forward / collective-pipeliner-backward / collective-pipeliner-forwardsink

See also
[`xla::CollectivePipeliner`](https://github.com/openxla/xla/tree/main/xla/service/collective_pipeliner.h).

TODO: Pass Description

### collective-transformation-reorderer

See also
[`xla::CollectiveTransformationReorder`](https://github.com/openxla/xla/tree/main/xla/service/collective_transformation_reorderer.h).

TODO: Pass Description

### collectives-schedule-linearizer

See also
[`xla::CollectivesScheduleLinearizer`](https://github.com/openxla/xla/tree/main/xla/service/collectives_schedule_linearizer.h).

TODO: Pass Description

### computation-deduplicator

See also
[`xla::HloComputationDeduplicator`](https://github.com/openxla/xla/tree/main/xla/service/hlo_computation_deduplicator.h).

TODO: Pass Description

### conditional-canonicalizer

See also
[`xla::ConditionalCanonicalizer`](https://github.com/openxla/xla/tree/main/xla/service/conditional_canonicalizer.h).

TODO: Pass Description

### conditional-code-motion

See also
[`xla::ConditionalCodeMotion`](https://github.com/openxla/xla/tree/main/xla/service/conditional_code_motion.h).

TODO: Pass Description

### conditional-to-select

See also
[`xla::ConditionalToSelect`](https://github.com/openxla/xla/tree/main/xla/service/conditional_to_select.h).

TODO: Pass Description

### constant_folding

See also
[`xla::HloConstantFolding`](https://github.com/openxla/xla/tree/main/xla/service/hlo_constant_folding.h).

TODO: Pass Description

### control-dep-remover

See also
[`xla::Despecializer`](https://github.com/openxla/xla/tree/main/xla/service/despecializer.h).

TODO: Pass Description

### control-dep-remover

See also
[`xla::DeconstructReduceWindowToReduceBroadcast`](https://github.com/openxla/xla/tree/main/xla/service/despecializer.h).

TODO: Pass Description

### control-dep-remover

See also
[`xla::ControlDepRemover`](https://github.com/openxla/xla/tree/main/xla/service/despecializer.h).

TODO: Pass Description

### convert-async-collectives-to-sync

See also
[`xla::ConvertAsyncCollectivesToSync`](https://github.com/openxla/xla/tree/main/xla/service/convert_async_collectives_to_sync.h).

TODO: Pass Description

### convert-memory-placement-to-internal-annotations

See also
[`xla::ConvertMemoryPlacementToInternalAnnotations`](https://github.com/openxla/xla/tree/main/xla/service/convert_memory_placement_to_internal_annotations.h).

TODO: Pass Description

### convert-mover

See also
[`xla::ConvertMover`](https://github.com/openxla/xla/tree/main/xla/service/convert_mover.h).

TODO: Pass Description

### convolution-group-converter

See also
[`xla::ConvolutionGroupConverter`](https://github.com/openxla/xla/tree/main/xla/service/convolution_group_converter.h).

TODO: Pass Description

### copy-insertion

See also
[`xla::CopyInsertion`](https://github.com/openxla/xla/tree/main/xla/service/copy_insertion.h).

TODO: Pass Description

### cse

See also
[`xla::HloCSE`](https://github.com/openxla/xla/tree/main/xla/service/hlo_cse.h).

TODO: Pass Description

### cse_barrier_expander

See also
[`xla::OptimizationBarrierExpander`](https://github.com/openxla/xla/tree/main/xla/service/optimization_barrier_expander.h).

TODO: Pass Description

### dce

See also
[`xla::HloDCE`](https://github.com/openxla/xla/tree/main/xla/service/hlo_dce.h).

TODO: Pass Description

### defuser

See also
[`xla::Defuser`](https://github.com/openxla/xla/tree/main/xla/service/defuser.h).

TODO: Pass Description

### domain_isolator

See also
[`xla::HloDomainIsolator`](https://github.com/openxla/xla/tree/main/xla/service/hlo_domain_isolator.h).

TODO: Pass Description

### domain_remover

See also
[`xla::HloDomainRemover`](https://github.com/openxla/xla/tree/main/xla/service/hlo_domain_remover.h).

TODO: Pass Description

### domain_verifier

See also
[`xla::HloDomainVerifier`](https://github.com/openxla/xla/tree/main/xla/service/hlo_domain_verifier.h).

TODO: Pass Description

### dot-merger

See also
[`xla::DotMerger`](https://github.com/openxla/xla/tree/main/xla/service/dot_merger.h).

TODO: Pass Description

### dot_decomposer

See also
[`xla::DotDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/dot_decomposer.h).

TODO: Pass Description

### dot_dimension_merger

See also
[`xla::DotDimensionMerger`](https://github.com/openxla/xla/tree/main/xla/service/dot_dimension_merger.h).

TODO: Pass Description

### dynamic-dimension-simplifier

See also
[`xla::DynamicDimensionSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/dynamic_dimension_simplifier.h).

TODO: Pass Description

### dynamic-index-splitter

See also
[`xla::DynamicIndexSplitter`](https://github.com/openxla/xla/tree/main/xla/service/dynamic_index_splitter.h).

TODO: Pass Description

### dynamic_padder

See also
[`xla::DynamicPadder`](https://github.com/openxla/xla/tree/main/xla/service/dynamic_padder.h).

TODO: Pass Description

### element_type_converter

See also
[`xla::HloElementTypeConverter`](https://github.com/openxla/xla/tree/main/xla/service/hlo_element_type_converter.h).

TODO: Pass Description

### flatten-call-graph

See also
[`xla::FlattenCallGraph`](https://github.com/openxla/xla/tree/main/xla/service/flatten_call_graph.h).

TODO: Pass Description

### fusion

See also
[`xla::InstructionFusion`](https://github.com/openxla/xla/tree/main/xla/service/instruction_fusion.h).

TODO: Pass Description

### fusion_constant_sinking

See also
[`xla::FusionConstantSinking`](https://github.com/openxla/xla/tree/main/xla/service/fusion_constant_sinking.h).

TODO: Pass Description

### hlo-descheduler

See also
[`xla::HloMemoryScheduler`](https://github.com/openxla/xla/tree/main/xla/service/hlo_memory_scheduler.h).

TODO: Pass Description

### hlo-descheduler

See also
[`xla::HloTrivialScheduler`](https://github.com/openxla/xla/tree/main/xla/service/hlo_memory_scheduler.h).

TODO: Pass Description

### hlo-descheduler

See also
[`xla::HloDescheduler`](https://github.com/openxla/xla/tree/main/xla/service/hlo_memory_scheduler.h).

TODO: Pass Description

### hlo-module-dce

See also
[`xla::HloModuleDCE`](https://github.com/openxla/xla/tree/main/xla/service/hlo_module_dce.h).

TODO: Pass Description

### hlo-verifier

See also
[`xla::HloVerifier`](https://github.com/openxla/xla/tree/main/xla/service/hlo_verifier.h).

TODO: Pass Description

### host-memory-transfer-asyncifier

See also
[`xla::HostMemoryTransferAsyncifier`](https://github.com/openxla/xla/tree/main/xla/service/host_memory_transfer_asyncifier.h).

TODO: Pass Description

### host-offload-legalize

See also
[`xla::HostOffloadLegalize`](https://github.com/openxla/xla/tree/main/xla/service/host_offload_legalize.h).

TODO: Pass Description

### host-offloader

See also
[`xla::HostOffloader`](https://github.com/openxla/xla/tree/main/xla/service/host_offloader.h).

TODO: Pass Description

### indexed-array-analysis-printer-pass

See also
[`xla::IndexedArrayAnalysisPrinterPass`](https://github.com/openxla/xla/tree/main/xla/service/indexed_array_analysis.h).

TODO: Pass Description

### instruction-hoister

See also
[`xla::InstructionHoister`](https://github.com/openxla/xla/tree/main/xla/service/instruction_hoister.h).

TODO: Pass Description

### int4-size-removal / int4-size-setter

See also
[`xla::SubByteNormalization`](https://github.com/openxla/xla/tree/main/xla/service/sub_byte_normalization.h).

TODO: Pass Description

### latency-hiding-scheduler

See also
[`xla::LatencyHidingScheduler`](https://github.com/openxla/xla/tree/main/xla/service/latency_hiding_scheduler.h).

TODO: Pass Description

### latency-hiding-scheduler-preparation

See also
[`xla::P2PSchedulePreparation`](https://github.com/openxla/xla/tree/main/xla/service/p2p_schedule_preparation.h).

TODO: Pass Description

### layout-assignment

See also
[`xla::LayoutAssignment`](https://github.com/openxla/xla/tree/main/xla/service/layout_assignment.h).

TODO: Pass Description

### layout_normalization

See also
[`xla::LayoutNormalization`](https://github.com/openxla/xla/tree/main/xla/service/layout_normalization.h).

TODO: Pass Description

### loop-schedule-linearizer

See also
[`xla::LoopScheduleLinearizer`](https://github.com/openxla/xla/tree/main/xla/service/loop_schedule_linearizer.h).

TODO: Pass Description

### map-inline

See also
[`xla::MapInliner`](https://github.com/openxla/xla/tree/main/xla/service/map_inliner.h).

TODO: Pass Description

### memory-space-propagation

See also
[`xla::MemorySpacePropagation`](https://github.com/openxla/xla/tree/main/xla/service/memory_space_propagation.h).

TODO: Pass Description

### multi_output_fusion

See also
[`xla::MultiOutputFusion`](https://github.com/openxla/xla/tree/main/xla/service/multi_output_fusion.h).

TODO: Pass Description

### my-new-pass

See also
[`xla::MyNewPass`](https://github.com/openxla/xla/tree/main/xla/service/hlo_pass_interface.h).

TODO: Pass Description

### optimize_input_output_buffer_alias

See also
[`xla::OptimizeInputOutputBufferAlias`](https://github.com/openxla/xla/tree/main/xla/service/optimize_input_output_buffer_alias.h).

TODO: Pass Description

### reduce-decomposer

See also
[`xla::ReduceDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/reduce_decomposer.h).

TODO: Pass Description

### reduce-scatter-combiner

See also
[`xla::ReduceScatterCombiner`](https://github.com/openxla/xla/tree/main/xla/service/reduce_scatter_combiner.h).

TODO: Pass Description

### reduce-scatter-decomposer

See also
[`xla::ReduceScatterDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/reduce_scatter_decomposer.h).

TODO: Pass Description

### reduce-scatter-reassociate

See also
[`xla::ReduceScatterReassociate`](https://github.com/openxla/xla/tree/main/xla/service/reduce_scatter_reassociate.h).

TODO: Pass Description

### rematerialization

See also
[`xla::HloRematerialization`](https://github.com/openxla/xla/tree/main/xla/service/hlo_rematerialization.h).

TODO: Pass Description

### reshape-decomposer

See also
[`xla::ReshapeDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/reshape_decomposer.h).

TODO: Pass Description

### reshape-mover

See also
[`xla::ReshapeMover`](https://github.com/openxla/xla/tree/main/xla/service/reshape_mover.h).

TODO: Pass Description

### root-instruction-sinker

See also
[`xla::RootInstructionSinker`](https://github.com/openxla/xla/tree/main/xla/service/root_instruction_sinker.h).

TODO: Pass Description

### sharding-format-picker

See also
[`xla::ShardingFormatPicker`](https://github.com/openxla/xla/tree/main/xla/service/sharding_format_picker.h).

TODO: Pass Description

### sharding-propagation

See also
[`xla::ShardingPropagation`](https://github.com/openxla/xla/tree/main/xla/service/sharding_propagation.h).

TODO: Pass Description

### sharding-remover

See also
[`xla::ShardingRemover`](https://github.com/openxla/xla/tree/main/xla/service/sharding_remover.h).

TODO: Pass Description

### simplify-conditional

See also
[`xla::ConditionalSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/conditional_simplifier.h).

TODO: Pass Description

### simplify-fp-conversions

See also
[`xla::SimplifyFPConversions`](https://github.com/openxla/xla/tree/main/xla/service/simplify_fp_conversions.h).

TODO: Pass Description

### simplify-sorts

See also
[`xla::SortSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/sort_simplifier.h).

TODO: Pass Description

### simplify-while-loops

See also
[`xla::WhileLoopSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_simplifier.h).

TODO: Pass Description

### slice-sinker

See also
[`xla::SliceSinker`](https://github.com/openxla/xla/tree/main/xla/service/slice_sinker.h).

TODO: Pass Description

### space-to-batch-converter

See also
[`xla::SpaceToBatchConverter`](https://github.com/openxla/xla/tree/main/xla/service/space_to_batch_converter.h).

TODO: Pass Description

### stochastic_convert_decomposer

See also
[`xla::StochasticConvertDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/stochastic_convert_decomposer.h).

TODO: Pass Description

### topk-decomposer

See also
[`xla::TopkRewriter`](https://github.com/openxla/xla/tree/main/xla/service/topk_rewriter.h).

TODO: Pass Description

### topk-decomposer

See also
[`xla::TopkDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/topk_rewriter.h).

TODO: Pass Description

### transpose-folding

See also
[`xla::TransposeFolding`](https://github.com/openxla/xla/tree/main/xla/service/transpose_folding.h).

TODO: Pass Description

### tree_reduction_rewriter

See also
[`xla::TreeReductionRewriter`](https://github.com/openxla/xla/tree/main/xla/service/tree_reduction_rewriter.h).

TODO: Pass Description

### tuple-simplifier

See also
[`xla::TupleSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/tuple_simplifier.h).

TODO: Pass Description

### while-loop-all-reduce-code-motion

See also
[`xla::WhileLoopAllReduceCodeMotion`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_all_reduce_code_motion.h).

TODO: Pass Description

### while-loop-concat-code-motion

See also
[`xla::WhileLoopConcatCodeMotion`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_concat_code_motion.h).

TODO: Pass Description

### while-loop-constant-sinking

See also
[`xla::WhileLoopConstantSinking`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_constant_sinking.h).

TODO: Pass Description

### while-loop-expensive-invariant-code-motion

See also
[`xla::WhileLoopExpensiveInvariantCodeMotion`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_expensive_invariant_code_motion.h).

TODO: Pass Description

### while-loop-fusible-sinking

See also
[`xla::WhileLoopFusibleSinking`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_fusible_sinking.h).

TODO: Pass Description

### while-loop-invariant-code-motion

See also
[`xla::WhileLoopInvariantCodeMotion`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_invariant_code_motion.h).

TODO: Pass Description

### while-loop-trip-count-annotator

See also
[`xla::WhileLoopTripCountAnnotator`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_trip_count_annotator.h).

TODO: Pass Description

### while_loop_unroller

See also
[`xla::WhileLoopUnroller`](https://github.com/openxla/xla/tree/main/xla/service/while_loop_unroller.h).

TODO: Pass Description

### zero_sized_hlo_elimination

See also
[`xla::ZeroSizedHloElimination`](https://github.com/openxla/xla/tree/main/xla/service/zero_sized_hlo_elimination.h).

TODO: Pass Description

## XLA Service CPU passes

### convolution-canonicalization

See also
[`xla::ConvCanonicalization`](https://github.com/openxla/xla/tree/main/xla/service/cpu/conv_canonicalization.h).

TODO: Pass Description

### cpu-parallel-task-assigner

See also
[`xla::ParallelTaskAssigner`](https://github.com/openxla/xla/tree/main/xla/service/cpu/parallel_task_assignment.h).

TODO: Pass Description

### onednn-matmul-rewriter

See also
[`xla::OneDnnMatMulRewriter`](https://github.com/openxla/xla/tree/main/xla/service/cpu/onednn_matmul_rewriter.h).

TODO: Pass Description

### onednn-ops-rewriter

See also
[`xla::OneDnnOpsRewriter`](https://github.com/openxla/xla/tree/main/xla/service/cpu/onednn_ops_rewriter.h).

TODO: Pass Description

### onednn-rewriter

See also
[`xla::OneDnnRewriter`](https://github.com/openxla/xla/tree/main/xla/service/cpu/onednn_rewriter.h).

TODO: Pass Description

## XLA Service GPU passes

### address-computation-fusion-rewriter

See also
[`xla::AddressComputationFusionRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/address_computation_fusion_rewriter.h).

TODO: Pass Description

### algorithm-checker

See also
[`xla::AlgorithmChecker`](https://github.com/openxla/xla/tree/main/xla/service/gpu/algorithm_checker.h).

TODO: Pass Description

### alias_passthrough_params

See also
[`xla::AliasPassthroughParams`](https://github.com/openxla/xla/tree/main/xla/service/gpu/alias_passthrough_params.h).

TODO: Pass Description

### all-gather-optimizer

See also
[`xla::AllGatherOptimizer`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_all_gather_optimizer.h).

TODO: Pass Description

### all-reduce-blueconnect

See also
[`xla::AllReduceBlueConnect`](https://github.com/openxla/xla/tree/main/xla/service/gpu/all_reduce_blueconnect.h).

TODO: Pass Description

### async-stream-attribute-wrapper

See also
[`xla::StreamAttributeAsyncWrapper`](https://github.com/openxla/xla/tree/main/xla/service/gpu/stream_attribute_async_wrapper.h).

TODO: Pass Description

### collective-permute-cycle-decomposer

See also
[`xla::CollectivePermuteCycleDecomposer`](https://github.com/openxla/xla/tree/main/xla/service/gpu/collective_permute_cycle_decomposer.h).

TODO: Pass Description

### command-buffer-scheduling

See also
[`xla::CommandBufferScheduling`](https://github.com/openxla/xla/tree/main/xla/service/gpu/command_buffer_scheduling.h).

TODO: Pass Description

### copy_fusion

See also
[`xla::CopyFusion`](https://github.com/openxla/xla/tree/main/xla/service/gpu/copy_fusion.h).

TODO: Pass Description

### cublas-gemm-broadcast-folding-rewriter

See also
[`xla::GemmBroadcastFoldingRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gemm_broadcast_folding_rewriter.h).

TODO: Pass Description

### cublas-gemm-rewriter

See also
[`xla::GemmRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gemm_rewriter.h).

TODO: Pass Description

### cublas-pad-for-gemms

See also
[`xla::CublasPadForGemms`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cublas_pad_for_gemms.h).

TODO: Pass Description

### cudnn-fused-convolution-rewriter

See also
[`xla::CudnnFusedConvRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_fused_conv_rewriter.h).

TODO: Pass Description

### cudnn-fused-multi-headed-attention-rewriter

See also
[`xla::CudnnFusedMHARewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_fused_mha_rewriter.h).

TODO: Pass Description

### cudnn-fused-multi-headed-attention-transpose-fusion

See also
[`xla::CudnnFusedMHATransposeFusion`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_fused_mha_transpose_fusion.h).

TODO: Pass Description

### cudnn-fusion-compiler

See also
[`xla::CuDnnFusionCompiler`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_fusion_compiler.h).

TODO: Pass Description

### cudnn-workspace-rewriter

See also
[`xla::CuDnnWorkspaceRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_workspace_rewriter.h).

TODO: Pass Description

### cudnn_pad_for_convolutions

See also
[`xla::CudnnPadForConvolutions`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_pad_for_convolutions.h).

TODO: Pass Description

### cudnn_simplify_padding

See also
[`xla::CudnnSimplifyPadding`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_simplify_padding.h).

TODO: Pass Description

### cudnn_vectorize_convolutions

See also
[`xla::CudnnVectorizeConvolutions`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_vectorize_convolutions.h).

TODO: Pass Description

### custom-kernel-fusion-rewriter

See also
[`xla::CustomKernelFusionRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/custom_kernel_fusion_rewriter.h).

TODO: Pass Description

### dot_dimension_sorter

See also
[`xla::DotDimensionSorter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/dot_dimension_sorter.h).

TODO: Pass Description

### fusion-wrapper

See also
[`xla::FusionWrapper`](https://github.com/openxla/xla/tree/main/xla/service/gpu/fusion_wrapper.h).

TODO: Pass Description

### fusion_merger

See also
[`xla::FusionMerger`](https://github.com/openxla/xla/tree/main/xla/service/gpu/fusion_merger.h).

TODO: Pass Description

### gemm-algorithm-picker

See also
[`xla::GemmAlgorithmPicker`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gemm_algorithm_picker.h).

TODO: Pass Description

### gemv-rewriter

See also
[`xla::GemvRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gemv_rewriter.h).

TODO: Pass Description

### gpu-async-collective-annotator

See also
[`xla::GpuAsyncCollectiveAnnotator`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_async_collective_annotator.h).

TODO: Pass Description

### gpu-conv-algorithm-picker

See also
[`xla::GpuConvAlgorithmPicker`](https://github.com/openxla/xla/tree/main/xla/service/gpu/conv_algorithm_picker.h).

TODO: Pass Description

### gpu-conv-padding-legalization

See also
[`xla::GpuConvPaddingLegalization`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_conv_padding_legalization.h).

TODO: Pass Description

### gpu-conv-rewriter

See also
[`xla::GpuConvRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_conv_rewriter.h).

TODO: Pass Description

### gpu-schedule-postprocessing

See also
[`xla::GpuSchedulePostprocessing`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_schedule_postprocessing.h).

TODO: Pass Description

### gpu-sort-rewriter

See also
[`xla::GpuSortRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_sort_rewriter.h).

TODO: Pass Description

### gpu-tree-reduction-rewriter

See also
[`xla::GpuTreeReductionRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/tree_reduction_rewriter.h).

TODO: Pass Description

### gpu-windowed-einsum-handler

See also
[`xla::GpuWindowedEinsumHandler`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_windowed_einsum_handler.h).

TODO: Pass Description

### gpu_cost_model_stats_collection

See also
[`xla::GpuCostModelStatsCollection`](https://github.com/openxla/xla/tree/main/xla/service/gpu/model/gpu_cost_model_stats_collection.h).

TODO: Pass Description

### gpu_horizontal_input_fusion

See also
[`xla::GpuHorizontalInputFusion`](https://github.com/openxla/xla/tree/main/xla/service/gpu/horizontal_input_fusion.h).

TODO: Pass Description

### gpu_horizontal_loop_fusion

See also
[`xla::GpuHorizontalLoopFusion`](https://github.com/openxla/xla/tree/main/xla/service/gpu/horizontal_loop_fusion.h).

TODO: Pass Description

### gpusolver-rewriter

See also
[`xla::GpusolverRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cusolver_rewriter.h).

TODO: Pass Description

### loop-double-buffer-transformer

See also
[`xla::LoopDoubleBufferTransformer`](https://github.com/openxla/xla/tree/main/xla/service/gpu/loop_double_buffer_transformer.h).

TODO: Pass Description

### move_copy_to_users

See also
[`xla::MoveCopyToUsers`](https://github.com/openxla/xla/tree/main/xla/service/gpu/move_copy_to_users.h).

TODO: Pass Description

### multi_output_fusion

See also
[`xla::GpuMultiOutputFusion`](https://github.com/openxla/xla/tree/main/xla/service/gpu/multi_output_fusion.h).

TODO: Pass Description

### norm-rewriter

See also
[`xla::CudnnNormRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/cudnn_norm_rewriter.h).

TODO: Pass Description

### reduce-scatter-creator

See also
[`xla::ReduceScatterCreator`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_reduce_scatter_creator.h).

TODO: Pass Description

### reduction-degenerate-dim-remover

See also
[`xla::ReductionDegenerateDimRemover`](https://github.com/openxla/xla/tree/main/xla/service/gpu/reduction_degenerate_dim_remover.h).

TODO: Pass Description

### reduction-dimension-grouper

See also
[`xla::ReductionDimensionGrouper`](https://github.com/openxla/xla/tree/main/xla/service/gpu/reduction_dimension_grouper.h).

TODO: Pass Description

### reduction-layout-normalizer

See also
[`xla::ReductionLayoutNormalizer`](https://github.com/openxla/xla/tree/main/xla/service/gpu/reduction_layout_normalizer.h).

TODO: Pass Description

### reduction-splitter

See also
[`xla::ReductionSplitter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/reduction_splitter.h).

TODO: Pass Description

### rename_fusions

See also
[`xla::RenameFusions`](https://github.com/openxla/xla/tree/main/xla/service/gpu/rename_fusions.h).

TODO: Pass Description

### sanitize-constant-names

See also
[`xla::GpuSanitizeConstantNames`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gpu_sanitize_constant_names.h).

TODO: Pass Description

### scatter-slice-simplifier

See also
[`xla::ScatterSliceSimplifier`](https://github.com/openxla/xla/tree/main/xla/service/gpu/scatter_slice_simplifier.h).

TODO: Pass Description

### stream-attribute-annotator

See also
[`xla::StreamAttributeAnnotator`](https://github.com/openxla/xla/tree/main/xla/service/gpu/stream_attribute_annotator.h).

TODO: Pass Description

### topk-specializer

See also
[`xla::TopkSpecializer`](https://github.com/openxla/xla/tree/main/xla/service/gpu/topk_specializer.h).

TODO: Pass Description

### topk-splitter

See also
[`xla::TopKSplitter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/topk_splitter.h).

TODO: Pass Description

### triangular-solve-rewriter

See also
[`xla::TriangularSolveRewriter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/triangular_solve_rewriter.h).

TODO: Pass Description

### triton-autotuner

See also
[`xla::GemmFusionAutotuner`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gemm_fusion_autotuner.h).

TODO: Pass Description

### triton-gemm-rewriter

See also
[`xla::GemmFusion`](https://github.com/openxla/xla/tree/main/xla/service/gpu/gemm_fusion.h).

TODO: Pass Description

### triton-softmax-rewriter

See also
[`xla::SoftmaxRewriterTriton`](https://github.com/openxla/xla/tree/main/xla/service/gpu/softmax_rewriter_triton.h).

TODO: Pass Description

### variadic-op-splitter

See also
[`xla::VariadicOpSplitter`](https://github.com/openxla/xla/tree/main/xla/service/gpu/variadic_op_splitter.h).

TODO: Pass Description

## XLA Service SPMD passes

### canon-all-gather-for-cse

See also
[`xla::CanonicalizeAllGatherForCSE`](https://github.com/openxla/xla/tree/main/xla/service/spmd/canonicalize_all_gather_for_cse.h).

TODO: Pass Description

### collective-permute-motion

See also
[`xla::CollectivePermuteMotion`](https://github.com/openxla/xla/tree/main/xla/service/spmd/collective_permute_motion.h).

TODO: Pass Description

### Noop

See also
[`xla::PartitionAssignment`](https://github.com/openxla/xla/tree/main/xla/service/spmd/partition_assignment.h).

TODO: Pass Description

### schedule-aware-collective-cse

See also
[`xla::ScheduleAwareCollectiveOpsCSE`](https://github.com/openxla/xla/tree/main/xla/service/spmd/schedule_aware_collective_ops_cse.h).

TODO: Pass Description

### spmd-partitioning

See also
[`xla::SpmdPartitioner`](https://github.com/openxla/xla/tree/main/xla/service/spmd/spmd_partitioner.h).

TODO: Pass Description

### spmd-prepare

See also
[`xla::SpmdPrepare`](https://github.com/openxla/xla/tree/main/xla/service/spmd/spmd_prepare.h).

TODO: Pass Description

### whole-graph-manual-pass

See also
[`xla::WholeGraphManualPass`](https://github.com/openxla/xla/tree/main/xla/service/spmd/whole_graph_manual_pass.h).

TODO: Pass Description

## XLA HLO Transforms passes

### hlo-constant-splitter

See also
[`xla::HloConstantSplitter`](https://github.com/openxla/xla/tree/main/xla/hlo/transforms/hlo_constant_splitter.h).

Splits the constant instructions such that they have a single user.
This is typically used before domain placement, to make sure a shared constant does not short-circuit domains. It is also used before sharding propagation to prevent unintended propagation of sharding due to shared used of constants.

CSE passes after domain placements will ensure that all the sharable constants within the same domain, will be rejoined back.

This pass may generate dead instructions. Thus, HloDCE is recommended after this pass.

## XLA HLO Experimental passes

### auto_sharding

See also
[`xla::DummyAutoSharding`](https://github.com/openxla/xla/tree/main/xla/hlo/experimental/auto_sharding/auto_sharding.h).

TODO: Pass Description

### auto_sharding

See also
[`xla::AutoSharding`](https://github.com/openxla/xla/tree/main/xla/hlo/experimental/auto_sharding/auto_sharding.h).

TODO: Pass Description

## XLA Tools passes

### control-flow-flattening

See also
[`xla::HloControlFlowFlattening`](https://github.com/openxla/xla/tree/main/xla/tools/hlo_control_flow_flattening.h).

TODO: Pass Description
