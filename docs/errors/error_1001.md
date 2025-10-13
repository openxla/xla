# Error code: 1001

**Category:** HBM and Vmem OOM

**Type:** Compile Time

## Error log example

```
XlaRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=134217728) would exceed memory (size=67108864) :: #allocation25 [shape = 'u8[134217728]{0}', space=vmem, size = 0x8000000, tag = 'scoped memory for ragged_latency_optimized_all_gather_lhs_contracting_gated_matmul_kernel.2'] :: ragged_latency_optimized_all_gather_lhs_contracting_gated_matmul_kernel.2
```
