// RUN: xla-opt --split-input-file --verify-diagnostics %s

func.func @extract_0d(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{cannot extract a 0-d tensor}}
  %0 = triton_xla.extract from %arg0 as memref<bf16, #triton_xla.layout<[]>> [][][] : tensor<bf16>
  return
}

// -----

func.func @insert_0d(%arg0: tensor<bf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{cannot insert a 0-d tensor}}
  triton_xla.insert %arg0 into %arg1 as memref<bf16, #triton_xla.layout<[]>> [][][] : tensor<bf16>
  return
}

// -----

func.func @extract_wrong_rank(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{shape attribute has a wrong size}}
  %0 = triton_xla.extract from %arg0 as memref<bf16, #triton_xla.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @extract_wrong_shape(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{shape size must match operand size}}
  %0 = triton_xla.extract from %arg0 as memref<16xbf16, #triton_xla.layout<[0]>> [0][16][1] : tensor<8xbf16>
  return
}

// -----

func.func @extract_wrong_layout(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{layout has 0 dimensions, but shape has 1}}
  %0 = triton_xla.extract from %arg0 as memref<8xbf16, #triton_xla.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @insert_wrong_rank(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{shape attribute has a wrong size}}
  triton_xla.insert %arg0 into %arg1 as memref<bf16, #triton_xla.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @insert_wrong_shape(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{shape size must match operand size}}
  triton_xla.insert %arg0 into %arg1 as memref<16xbf16, #triton_xla.layout<[0]>> [0][16][1] : tensor<8xbf16>
  return
}

// -----

func.func @insert_wrong_layout(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{layout has 0 dimensions, but shape has 1}}
  triton_xla.insert %arg0 into %arg1 as memref<8xbf16, #triton_xla.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}
