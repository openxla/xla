# Error code: 03xx

**Category:** Core Halt

**Type:** Runtime

## Error log example

```
"XlaRuntimeError: INTERNAL: Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x175 (from TensorCoreSequencer:1:0x21f): scheckne: 
***************
An unexpected peer shows up in the launch group with a different launch id than the current group leader. If using single controller backends, this signals a bug in the runtime scheduling system or above. If using multi-controller backends"
```
