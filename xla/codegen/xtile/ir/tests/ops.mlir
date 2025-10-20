// RUN: emitters_opt %s --split-input-file --verify-roundtrip -verify-diagnostics

xtile.entry_func @happy_path(%input: memref<1024x4xf32>, %output: memref<128x1024xf32>, %tile_id: index) {
  %tile = xtile.extract %input[%tile_id, %tile_id][10, 1][1, 1] : memref<1024x4xf32> -> tensor<10xf32>
  xtile.insert %tile into %output[%tile_id, %tile_id][10, 1][1, 1] : tensor<10xf32> -> memref<128x1024xf32>
  xtile.return
}

// -----

xtile.entry_func @with_attributes(
  %input: memref<1024xf32> {xla.some_attr = 1},
  %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:10, tiles_per_workgroup:5>} {
  xtile.return
}

// -----

// expected-error@+1 {{entry function arguments should be of the form (arg: memref..., tile_id: index)}}
xtile.entry_func @tile_id_at_start(%tile_id: index, %input: memref<1024xf32>, %output: memref<1024xf32>) {
  xtile.return
}

// -----

// expected-error@+1 {{entry function arguments should be of the form (arg: memref..., tile_id: index)}}
xtile.entry_func @too_many_tile_ids(%input: memref<1024xf32>, %id0: index, %id1: index) {
  xtile.return
}

// -----

func.func @incorrect_full_shape_extract(%arg: memref<1024xf32>) -> tensor<10xf32> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape size: 2 does not match rank of source: 1}}
  %tile = xtile.extract %arg[%offset][10, 1][1] : memref<1024xf32> -> tensor<10xf32>
  return %tile : tensor<10xf32>
}

// -----

func.func @incorrect_offset_count_extract(%arg: memref<1024xf32>) -> tensor<10xf32> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{expected 1 offset operands, got 2}}
  %tile = xtile.extract %arg[%offset, %offset][10][1] : memref<1024xf32> -> tensor<10xf32>
  return %tile : tensor<10xf32>
}

// -----

func.func @incorrect_rank_reduction_extract(%arg: memref<16x1024xf32>) -> tensor<10xf32> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape: [16, 10] does not reduce to result shape: [10]}}
  %tile = xtile.extract %arg[%offset, %offset][16, 10][1, 1] : memref<16x1024xf32> -> tensor<10xf32>
  return %tile : tensor<10xf32>
}

// -----

func.func @type_mismatch_extract(%arg: memref<1024xf32>) -> tensor<10xf64> {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{result element type: 'f64' does not match element type of source: 'f32'}}
  %tile = xtile.extract %arg[%offset][10][1] : memref<1024xf32> -> tensor<10xf64>
  return %tile : tensor<10xf64>
}

// -----

func.func @incorrect_full_shape_insert(%src: tensor<24xf32>, %dst: memref<1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape size: 2 does not match rank of destination: 1}}
  xtile.insert %src into %dst[%offset][24, 1][1] : tensor<24xf32> -> memref<1024xf32>
  return
}

// -----

func.func @incorrect_offset_count_insert(%src: tensor<24xf32>, %dst: memref<1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{expected 1 offset operands, got 2}}
  xtile.insert %src into %dst[%offset, %offset][24][1] : tensor<24xf32> -> memref<1024xf32>
  return
}

// -----

func.func @incorrect_rank_reduction_insert(%src: tensor<24xf32>, %dst: memref<16x1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{full tile shape: [16, 24] does not reduce to source shape: [24]}}
  xtile.insert %src into %dst[%offset, %offset][16, 24][1, 1] : tensor<24xf32> -> memref<16x1024xf32>
  return
}

// -----

func.func @type_mismatch_insert(%src: tensor<24xf64>, %dst: memref<1024xf32>) {
  %offset = arith.constant 0 : index
  // expected-error@+1 {{destination element type: 'f32' does not match element type of source: 'f64'}}
  xtile.insert %src into %dst[%offset][24][1] : tensor<24xf64> -> memref<1024xf32>
  return
}
