module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<2.0> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense<2.0> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense<2.0> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense<2.0> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
      %2 = arith.mulf %1#1, %cst_0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      } {mgmt.mgmt = #mgmt.mgmt<level = 0>}
      %6 = arith.mulf %5#1, %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}