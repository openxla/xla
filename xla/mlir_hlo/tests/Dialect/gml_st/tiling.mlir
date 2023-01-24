// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-2d" \
// RUN: --gml-tiling="tile-sizes=1,1 distribute=false op-label=tile-2d-point" \
// RUN: --gml-tiling="tile-sizes=1 distribute=false op-label=tile-1d-point" \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-3d" \
// RUN: --gml-tiling="tile-sizes=10 distribute=false op-label=tile-1d" \
// RUN: --gml-tiling="tile-sizes=2,4 distribute=false op-label=tile-pad" \
// RUN: --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-FOR

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=true op-label=tile-2d" \
// RUN: --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-PARALLEL

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_static(%lhs: tensor<1024x1024xf32>, %rhs: tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<1024x1024xf32>
  %add = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "tile-2d"}
      ins(%lhs, %rhs : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%init : tensor<1024x1024xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<1024x1024xf32>
  func.return %add : tensor<1024x1024xf32>
}

// CHECK-FOR-LABEL: @add_static
// CHECK-FOR-SAME:  %[[ARG0:.*]]: tensor<1024x1024xf32>, %[[ARG1:.*]]: tensor<1024x1024xf32>

// CHECK-FOR-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[C256:.*]] = arith.constant 256
// CHECK-FOR-DAG:   %[[C512:.*]] = arith.constant 512
// CHECK-FOR-DAG:   %[[C1024:.*]] = arith.constant 1024
// CHECK-FOR:       %[[INIT:.*]] = tensor.empty()
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[C1024]], %[[C1024]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[ARG4:.*]] = %[[INIT]]: tensor<1024x1024xf32>)
// CHECK-FOR:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[I]], %[[J]]] [256, 512] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG1]][%[[I]], %[[J]]] [256, 512] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[ARG4]][%[[I]], %[[J]]] [256, 512] [1, 1]
// CHECK-FOR:         %[[GENERIC:.*]] = linalg.generic
// CHECK-FOR-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<256x512xf32>, tensor<256x512xf32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_1]] : tensor<256x512xf32>)
// CHECK-FOR-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-FOR:         ^bb0(%[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32, %[[ARG7:.*]]: f32):
// CHECK-FOR:           %[[ADDF:.*]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK-FOR:           linalg.yield %[[ADDF]]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]]] [256, 512] [1, 1]
// CHECK-FOR:         gml_st.set_yield %[[GENERIC]] into %[[ARG4]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %add = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "tile-2d"}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<?x?xf32>
  func.return %add : tensor<?x?xf32>
}


// CHECK-FOR-LABEL: @add(
// CHECK-FOR-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>

// CHECK-FOR:       %[[C0:.*]] = arith.constant 0
// CHECK-FOR:       %[[C1:.*]] = arith.constant 1
// CHECK-FOR:       %[[C256:.*]] = arith.constant 256
// CHECK-FOR:       %[[C512:.*]] = arith.constant 512
// CHECK-FOR:       %[[LHS_DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-FOR:       %[[LHS_DIM_1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-FOR:       %[[INIT:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<?x?xf32>)
// CHECK-FOR:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[ARG2]])[%[[LHS_DIM_0]]]
// CHECK-FOR:         %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[ARG3]])[%[[LHS_DIM_1]]]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[OUT]][%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[GENERIC:.*]] = linalg.generic
// CHECK-FOR-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
// CHECK-FOR-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-FOR:         ^bb0(%[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32, %[[ARG7:.*]]: f32):
// CHECK-FOR:           %[[ADDF:.*]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK-FOR:           linalg.yield %[[ADDF]]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         gml_st.set_yield %[[GENERIC]] into %[[OUT]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]


// CHECK-PARALLEL-LABEL: @add(
// CHECK-PARALLEL-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>

// CHECK-PARALLEL:       %[[C0:.*]] = arith.constant 0
// CHECK-PARALLEL:       %[[C1:.*]] = arith.constant 1
// CHECK-PARALLEL:       %[[C256:.*]] = arith.constant 256
// CHECK-PARALLEL:       %[[C512:.*]] = arith.constant 512
// CHECK-PARALLEL:       %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]]
// CHECK-PARALLEL:       %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-PARALLEL:       %[[INIT:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-PARALLEL:       %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-PARALLEL-SAME:      to (%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-PARALLEL-SAME:      step (%[[C256]], %[[C512]])
// CHECK-PARALLEL:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[ARG2]])[%[[LHS_DIM_0]]]
// CHECK-PARALLEL:         %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[ARG3]])[%[[LHS_DIM_1]]]
// CHECK-PARALLEL:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[LHS]]
// CHECK-PARALLEL-SAME:      [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[RHS]]
// CHECK-PARALLEL-SAME:      [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[INIT]]
// CHECK-PARALLEL-SAME:      [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[GENERIC:.*]] = linalg.generic
// CHECK-PARALLEL-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-PARALLEL-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-PARALLEL-SAME:        outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
// CHECK-PARALLEL-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-PARALLEL:         ^bb0(%[[OUT:.*]]: f32, %[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32):
// CHECK-PARALLEL:           %[[ADDF:.*]] = arith.addf %[[OUT]], %[[ARG5]]
// CHECK-PARALLEL:           linalg.yield %[[ADDF]]
// CHECK-PARALLEL:         %[[TILE:.*]] = gml_st.tile [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         gml_st.set_yield %[[GENERIC]] into %[[INIT]][%[[TILE]]]
// CHECK-PARALLEL:       return %[[PARALLEL]]

// -----

func.func @reduce_row(%lhs: tensor<?x?xf32>,
                      %rhs: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>

  %init = tensor.empty(%0) : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32)
                      outs(%init : tensor<?xf32>) -> tensor<?xf32>
  %sum_of_prod = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"],
    op_label = "tile-2d"}
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %prod = arith.mulf %l, %r : f32
    %add = arith.addf %prod, %o : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  func.return %sum_of_prod : tensor<?xf32>
}


// CHECK-FOR-LABEL: @reduce_row
// CHECK-FOR-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>

// CHECK-FOR-DAG:   %[[C0_0:.*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[C1_0:.*]] = arith.constant 1
// CHECK-FOR-DAG:   %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0_0]]
// CHECK-FOR-DAG:   %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1_0]]
// CHECK-FOR-DAG:   %[[C256_0:.*]] = arith.constant 256
// CHECK-FOR-DAG:   %[[C512_0:.*]] = arith.constant 512
// CHECK-FOR-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00
// CHECK-FOR-DAG:   %[[INIT_0:.*]] = tensor.empty(%[[LHS_DIM_0]])
// CHECK-FOR-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT_0]] : tensor<?xf32>)
// CHECK-FOR:       %[[FOR_0:.*]] = gml_st.for (%[[ARG2_0:.*]], %[[ARG3_0:.*]]) = (%[[C0_0]], %[[C0_0]])
// CHECK-FOR-SAME:      to (%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-FOR-SAME:      step (%[[C256_0]], %[[C512_0]])
// CHECK-FOR-SAME:      outs (%[[OUT_0:.*]] = %[[FILL]]: tensor<?xf32>)
// CHECK-FOR:         %[[MIN_1:.*]] = affine.min #map{{[0-9]*}}(%[[ARG2_0]])[%[[LHS_DIM_0]]]
// CHECK-FOR:         %[[MIN_2:.*]] = affine.min #map{{[0-9]*}}(%[[ARG3_0]])[%[[LHS_DIM_1]]]
// CHECK-FOR:         %[[MATERIALIZE_2:.*]] = tensor.extract_slice %[[LHS]]
// CHECK-FOR-SAME:      [%[[ARG2_0]], %[[ARG3_0]]] [%[[MIN_1]], %[[MIN_2]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_3:.*]] = tensor.extract_slice %[[RHS]]
// CHECK-FOR-SAME:      [%[[ARG2_0]], %[[ARG3_0]]] [%[[MIN_1]], %[[MIN_2]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_4:.*]] = tensor.extract_slice %[[OUT_0]]
// CHECK-FOR-SAME:      [%[[ARG2_0]]] [%[[MIN_1]]] [1]
// CHECK-FOR:         %[[GENERIC_0:.*]] = linalg.generic
// CHECK-FOR-SAME:        iterator_types = ["parallel", "reduction"]}
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE_2]], %[[MATERIALIZE_3]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_4]] : tensor<?xf32>)
// CHECK-FOR-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-FOR:         ^bb0(%[[ARG5_0:.*]]: f32, %[[ARG6_0:.*]]: f32, %[[ARG7_0:.*]]: f32):
// CHECK-FOR:           %[[MULF:.*]] = arith.mulf %[[ARG5_0]], %[[ARG6_0]]
// CHECK-FOR:           %[[ADDF_0:.*]] = arith.addf %[[MULF]], %[[ARG7_0]]
// CHECK-FOR:           linalg.yield %[[ADDF_0]]
// CHECK-FOR:         %[[TILE_4:.*]] = gml_st.tile [%[[ARG2_0]]] [%[[MIN_1]]] [1]
// CHECK-FOR:         gml_st.set_yield %[[GENERIC_0]] into %[[OUT_0]][%[[TILE_4]]]
// CHECK-FOR:       return %[[FOR_0]]
