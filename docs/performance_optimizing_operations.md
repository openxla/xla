# Performance Optimizing Operations 

This is a high level landing page that describes what operations or variants of operations can help make the computation faster, cheaper, or more memory efficient. Detailed information on all operations can be found on the [operation semantics](https://openxla.org/xla/operation_semantics) page.

Operations on this page are organized by the inherent optimization of the operation.

## Parallel and Collective Ops

The following ops distribute work and/or aggregates results across replicas or devices. 

* [`AllGather`](https://openxla.org/xla/operation_semantics#allgather) \- Concatenates data across replicas.  
* [`AllReduce`](https://openxla.org/xla/operation_semantics#allreduce) \- Performs a combined reduction across replicas.  
* [`AllToAll`](https://openxla.org/xla/operation_semantics#alltoall) \- A collective operation that splits and exchanges data among replicas.  
* [`CollectiveBroadcast`](https://openxla.org/xla/operation_semantics#collectivebroadcast) \- Broadcasts data across replicas.   
* [`CollectivePermute`](https://openxla.org/xla/operation_semantics#collectivepermute) \- A collective operation that sends and receives data across replicas.  
* [`CrossReplicaSum`](https://openxla.org/xla/operation_semantics#crossreplicasum) \- Performs [`AllReduce`](https://openxla.org/xla/operation_semantics#allreduce) with a summation computation.  
* [`ReduceScatter`](https://openxla.org/xla/operation_semantics#reducescatter) \- A collective operation that effectively does an [`AllReduce`](https://openxla.org/xla/operation_semantics#allreduce) and then scatters the result by splitting it into shard blocks and replica groups.

## Reduced Precision

Ops performing truncating low-order mantissa bits and/or exponent bits to emulate lower precision.

* [ReducePrecision](https://openxla.org/xla/operation_semantics#reduceprecision) \- Models the effect of converting floating-point values to a lower-precision format (such as IEEE-FP16) and back to the original format.

## Matrix Multiplication / Dot Products

* [`Dot`](https://openxla.org/xla/operation_semantics#dot) \- Performs a standard dot product or matrix multiplication between two tensors.   
* [`DotGeneral`](https://openxla.org/xla/operation_semantics#dotgeneral)  \- Similar to [`Dot`](https://openxla.org/xla/operation_semantics#dot), but allows contracting and batch dimension numbers to be specified for both the lhs and rhs.  
* [`ScaledDot`](https://openxla.org/xla/operation_semantics#scaleddot) \- Creates a scaled dot op with operands 'lhs', 'lhs\_scale', 'rhs', and 'rhs\_scale', with contracting and batch dimensions specified in 'dimension\_numbers'.  
* [`RaggedDot`](https://openxla.org/xla/operation_semantics#raggeddot) \- Computes a matmul over a single ragged dimension. 

## Convolution

* [`Conv`](https://openxla.org/xla/operation_semantics#conv_convolution) \- Computes a convolution.   
* [ConvWithGeneralPadding](https://openxla.org/xla/operation_semantics#ConvWithGeneralPadding)  \- Same as [Conv](https://openxla.org/xla/operation_semantics#conv_convolution) where padding configuration is explicit.  
* [ConvWithGeneralDimensions](https://openxla.org/xla/operation_semantics#ConvWithGeneralDimensions) \- Same as [Conv](https://openxla.org/xla/operation_semantics#conv_convolution) where dimension numbers are explicit.  
* [ConvGeneral](https://openxla.org/xla/operation_semantics#ConvGeneral) \- Same as [Conv](https://openxla.org/xla/operation_semantics#conv_convolution) where dimension numbers and padding configuration is explicit. 
* [ConvGeneralDilated](https://openxla.org/xla/operation_semantics#convgeneraldilated) \- Same as [Conv](https://openxla.org/xla/operation_semantics#conv_convolution) where padding configuration, dilation factors, and dimension numbers are explicit.

## Data Layout and Memory Ops

The following are ops that change how data is stored, accessed, or moved. 

* [`BitcastConvertType`](https://openxla.org/xla/operation_semantics#bitcastconverttype) \- Performs an element-wise bitcast operation from a data shape to a target shape.
* [`Broadcast`](https://openxla.org/xla/operation_semantics#broadcast) \- Adds dimensions to an array by duplicating the data in the array.  
* [`BroadcastInDim`](https://openxla.org/xla/operation_semantics#broadcastindim) \- Expands the size and number of dimensions of an array by duplicating the data in the array.  
* [`Collapse`](https://openxla.org/xla/operation_semantics#collapse) \- Collapses dimensions of an array into one dimension.  
* [`ConcatInDim`](https://openxla.org/xla/operation_semantics#concatindim_concatenate) \- Composes an array from multiple array operands.  
* [`DynamicReshape`](https://openxla.org/xla/operation_semantics#dynamicreshape) \- This operation is functionally identical to reshape, but the result shape is specified dynamically via output\_shape.  
* [`DynamicSlice`](https://openxla.org/xla/operation_semantics#dynamicslice) \- Extracts a sub-array from the input array at dynamic start indices.  
* [`Gather`](https://openxla.org/xla/operation_semantics#gather) \- Stitches together several elements of an input array based on indexing.   
* [`Iota`](https://openxla.org/xla/operation_semantics#iota) \- Builds a constant literal on device rather than a potentially large host transfer.  
* [`Pad`](https://openxla.org/xla/operation_semantics#pad) \- Expands the given operand array by padding around the array as well as between the elements of the array with the given padding specifies the amount of edge padding and the interior padding for each dimension.  
* [`Reshape`](https://openxla.org/xla/operation_semantics#reshape) \- Reshapes the dimensions of an array into a new configuration.  
* [`Reverse`](https://openxla.org/xla/operation_semantics#rev_reverse) \- Reverses the order of elements in the operand array along the specified dimensions, generating an output array of the same shape.  
* [`Scatter`](https://openxla.org/xla/operation_semantics#scatter) \- Write or update elements in a tensor according to indices.  
* [`Slice`](https://openxla.org/xla/operation_semantics#slice) \- Extracts a sub-array from the input array.  
* [`Transpose`](https://openxla.org/xla/operation_semantics#transpose) \- Permuted the operand dimension.

## Other Domain Specific Operations

* [`Cholesky`](https://openxla.org/xla/operation_semantics#cholesky) \- Computes the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) of a batch of symmetric (Hermitian) positive definite matrices.  
* [`Fft`](https://openxla.org/xla/operation_semantics#fft) \-  Implements the forward and inverse Fourier Transforms for real and complex inputs/outputs. Multidimensional FFTs on up to 3 axes are supported.  
* [`TriangularSolve`](https://openxla.org/xla/operation_semantics#triangularsolve) \- Solves systems of linear equations with lower or upper triangular coefficient matrices by forward- or back-substitution.

