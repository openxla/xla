The specs in this folder are obtained by calling 
`StreamExecutor::CreateDeviceDescription`, then turned into
`GpuDeviceInfoProto`. They are useful when compiling with the flag
`--xla_gpu_target_config_filename`. Since a hardware generation may have several
SKUs, a spec may not be identical to what we would get on a particular machine,
but it will be "close enough".
