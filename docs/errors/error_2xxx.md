# Error code: 2xxx

**Category:** Mosaic Compilation Failures

**Type:** Compile Time

## Error log example

```
"INTERNAL: Mosaic failed to compile TPU kernel: Unsupported matmul RHS type on target: 'vector<128x128xbf16>'

at location: loc(""dot_general:""(""dot_general""(callsite(""_flash_attention_kernel_single_bat

INTERNAL: Mosaic failed to compile TPU kernel: cannot statically prove that index in dimension 0 is a multiple of 1024

at location: loc(""masked_load:""(""masked_load""(callsite(""fused_attention_kernel""(

INVALID_ARGUMENT: Mosaic failed to compile TPU kernel: unregistered operation 'tpu.wait_dma' found in dialect ('tpu') that does not allow unknown operations

INVALID_ARGUMENT: Mosaic failed to compile TPU kernel: Static device assignment is undefined

INTERNAL: Mosaic failed to compile TPU kernel: failed to legalize operation 'arith.mulf'

INTERNAL: Mosaic failed to compile TPU kernel: Slice shape along dimension 1 must be aligned to tiling (2), but is 1.at location: loc(""dma_start:""(""a2a-dma-send/dma_start""(callsite(""run_a2a.<locals>"
```
