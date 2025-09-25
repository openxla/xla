# HLO Passes 

[XLA (Accelerated Linear Algebra)](https://openxla.org/xla) is an open source compiler for machine learning that is part of the [OpenXLA](http://openxla.org) ecosystem. 

The XLA compiler takes models from popular frameworks such as PyTorch, TensorFlow, and JAX, and optimizes the models for high-performance execution across different hardware platforms including GPUs, CPUs, and Google’s custom accelerator, the TPU. Among other uses, XLA serves as a high-performance backend to Tensorflow and is the only method of running code on the TPU. 

Programs running on XLA are described in a high-level language called HLO (High Level Optimizer) which also serves as the machine-independent intermediate representation (IR) in the compiler. XLA performs numerous critical optimizations at the HLO-level before lowering the program into a machine-instruction level IR for code generation. This document outlines an overview of the HLO parts of the XLA compiler including summaries of important optimization and analysis passes.

[StableHLO](https://openxla.org/stablehlo), which is also part of the OpenXLA ecosystem, is not covered in the scope of this document. For information on StableHLO passes see [OpenXLA \- StableHLO \- StableHLO Passes](https://openxla.org/stablehlo/generated/stablehlo_passes).

## Features and Overview 

In HLO, programs are represented as dataflow graphs where values in the graph are arrays similar to Tensorflow’s program representation. Individual HLO instructions describe high-level operations performed on arrays such as convolution or element-wise operations such as add and subtract. A comprehensive description of HLO Operations can be found at [OpenXLA \- XLA \- Operation Semantics](https://openxla.org/xla/operation_semantics), which represents isolated functional computations given to the compiler.

XLA includes a large number of optimizations and transformations which operate on HLO. At the time of writing there are several hundred HLO optimization and analysis passes identified. These passes take advantage of the high abstraction level of HLO to perform optimizations and other transformations which are more difficult to perform on a program representation closer to the abstraction level of machine instructions.

An HLO operation generally lowers to a loop nest which computes the array output of the operation. However, HLO does not represent these loop nests explicitly. Instead, only the shapes of arrays are explicitly specified. These shapes determine the bounds of each loop and the depth of the loop nest after lowering. Focusing only on the shape and semantics of the operation makes many operations easier. 

For example:

* The [algebraic simplifier](https://github.com/openxla/xla/blob/c37fc6a383b870f43cef82280418fcefcc90b0f8/xla/hlo/transforms/simplifiers/algebraic_simplifier.h#L417) pass performs a number of mostly arithmetic simplifications and optimizations. A simple example is replacing division by a constant with multiplication by the inversion of the constant. More complicated examples include transformations involving shape-changing operations such as broadcast and reshape. The level of abstraction of HLO makes these optimizations on arrays as simple as scalar optimizations in a conventional compiler. 
* [Rematerialization](https://github.com/openxla/xla/tree/main/xla/hlo/transforms/simplifiers/hlo_rematerialization.h) recomputes selected expressions in the computation to reduce memory pressure caused by long live ranges of array-shaped values. In HLO, the array is the fundamental data type which makes the analysis and graph rewriting performed by rematerialization simpler. Rematerialization would be more complicated if the IR instead operated at the level of array elements. For similar reasons, the HLO abstraction level is a good match for scheduling to minimize memory pressure.

Relatedly, HLO also does not represent any pattern of reuse in the operation. For example, a matrix multiply looks similar to an element-wise add in HLO with the exception of their operand shape constraints. The data reuse inherent in the matrix multiply is not expressed. 

## Tooling and Testing

XLA comes with multiple command line tools, including the hlo-opt tool. This tool allows execution of an individual pass independent of the given platform compilation stages. For more information see [XLA \- Tooling](https://openxla.org/xla/tools#hlo-opt_hlo_pass_development_and_debugging).

For informationing on writing unit tests for HLO Passes see [Testing HLO Passes](https://openxla.org/xla/test_hlo_passes).

## Base Class

The base class for HLO passes can be found at [hlo\_pass\_interface](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h%20). HLO pass should not extend this class directly but instead should extend [HloModulePass](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h#L142) or [HloModuleGroupPass](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h#L172).

## Hardware-independent HLO Pass Examples

This section describes a few examples of passes shared across XLA backends. Some passes may be specialized for specific backends, but the high-level functionality is similar.

For a full list of hardware-independent HLO see [OptProvider::RegisterAllHardwareIndependentPasses](https://github.com/openxla/xla/blob/c37fc6a383b870f43cef82280418fcefcc90b0f8/xla/hlo/tools/hlo_opt/opt_lib.cc#L226).

### Rematerialization 

See also [HloRematerialization](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_rematerialization.h).

Selectively recomputes expressions within the HLO graph to reduce memory usage. Trades off higher compute for lower memory usage. Can reduce memory usage by tens of percent and is required to run many large models.

### Algebraic Simplifier 

See also [AlgebraicSimplifier](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.h).

Simplification A grab bag of simplifications, optimizations, and canonicalizations. Analogous to LLVM’s instcombine pass.

### Constant Folding 

See also [HloConstantFolding](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_constant_folding.h).

Replaces expressions which can be evaluated at compile time with their constant equivalent. 

### Dead Code Elimination 

See also [HloDCE](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/hlo_dce.h) .

Removes operations with unused results (fast implementation). 

### Call Graph Flattening 

See also [FlattenCallGraph](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/flatten_call_graph.h).

A legalization pass which converts the HLO call graph into a tree by cloning computations. Required because memory is statically assigned to HLO operations and not based on dynamic call context.

### Reshape Mover 

See also [ReshapeMover](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/reshape_mover.h).

Reshapes and transposes can be expensive, especially on TPU. This pass moves and reshapes and transposes across elementwise operations enabling the operations to be merged or eliminated.

### Zero-sized HLO Elimination

See also [ZeroSizedHloElimination](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h).

HLO supports arrays of zero size (one or more dimensions has a bound of zero). This pass simplifies the graph by replacing zero-sized operations with zero-sized constants.

## TPU-specific HLO Pass Examples

Passes specific to the TPU backend.

### Model parallelism 

The partitioning of an XLA program across multiple cores is performed at the HLO level and the TPU HLO pipeline includes a number of passes for supporting multi-core execution.

#### Spatial partitioning 

See also [ShardingPropagation](https://github.com/openxla/xla/blob/main/xla/service/sharding_propagation.h).

Pass to support dividing operations across devices along non-batch dimensions.

### Handling of bfloat16

See also [BFloat16ConversionFolding](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/bfloat16_conversion_folding.h), [BFloat16MixedPrecisionRemoval](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/float_normalization.h), and [BFloat16Propagation](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/bfloat16_propagation.h).

TPUs support bfloat16 as a lower-precision, more compact floating-point representation than 32-bit floats. Using bfloat16 reduces memory footprint and memory bandwidth. The TPU HLO pipeline includes various passes for replacing floats with bfloat16 into the program and propagating the precision through the graph.

### Legalization passes

See also [GatherExpander](https://github.com/openxla/xla/blob/main/xla/service/gather_expander.h), and [BatchNormExpander](https://github.com/openxla/xla/blob/main/xla/service/batchnorm_expander.h).

Passes which transform unsupported HLO into a form which the backend can emit or for which the backend produces a more efficient lowering.

## GPU-specific HLO Pass Example

Passes can be specific to the GPU backend. These passes can be identified as classes defined in `namespace gpu`.

### cuDNN Rewriter 

See also [CudnnFusedConvRewriter](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h) and [CudnnNormRewriter](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/cudnn_norm_rewriter.h).

Rewrites fused convolution and norm operations into their respective library calls in cuDNN.

## CPU-specific HLO Pass Examples

Passes can be specific to the CPU backend. These passes can be identified as classes defined in `namespace cpu`.

### Convolution Canonicalization

See also [ConvCanonicalization](https://github.com/openxla/xla/blob/main/xla/service/cpu/conv_canonicalization.h).

Canonicalizes convolutions so that they can be lowered to a fast implementation in Eigen.

### Operation Parallelization

See also [ParallelTaskAssigner](https://github.com/openxla/xla/blob/main/xla/service/cpu/parallel_task_assignment.h).

 Partitions HLOs into tasks to run on separate threads.

## Analysis passes

Analysis passes are not considered "HLO passes" since they do not transform HLO and may not extend HloModulePass or HloModuleGroupPass.

### Analysis Pass Examples

#### Dataflow Analysis

See also [HloDataflowAnalysis](https://github.com/openxla/xla/tree/main/xla/hlo/analysis/hlo_dataflow_analysis.h).

Analysis which identifies all HLO values in the graph and their uses.

#### Alias Analysis

See also [HloAliasAnalysis](https://github.com/openxla/xla/tree/main/xla/hlo/analysis/hlo_alias_analysis.h).

Identifies must-alias relationships between values in the program.

#### Computation Cost Analysis

See also [HloCostAnalysis](https://github.com/openxla/xla/tree/main/xla/service/hlo_cost_analysis.h).

Computes FLOP count and memory usage for all operations in the program.

#### HLO Verification

See also [Hloverifier](https://github.com/openxla/xla/tree/main/xla/service/hlo_verifier.h).

Verifies various invariants of the HLO graph.

