# Error code: 1000

**Category:** HBM and Vmem OOM

**Type:** Compile Time

## Error log example

```
"XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G. Total hbm usage >= 49.34G: reserved 3.12M program unknown size arguments 49.34G

JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem while allocating on stack for %ragged_latency_optimized_all_gather_lhs_contracting_gated_matmul_kernel.18 = bf16[2048,4096]{1,0:T(8,128)(2,1)} custom-call(%get-tuple-element.18273, %get-tuple-element.18274, %get-tuple-element.18275, %get-tuple-element.18276, %get-tuple-element.18277, /*index=5*/%bitcast.8695, %get-tuple-element.19201, %get-tuple-element.19202, %get-tuple-element.19203, %get-tuple-element.19204), custom_call_target="""
```
