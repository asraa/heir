Loading: 
Loading: 
Loading: 0 packages loaded
Analyzing: target //tools:heir-opt (0 packages loaded, 0 targets configured)
INFO: Analyzed target //tools:heir-opt (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
[0 / 1] [Prepa] BazelWorkspaceStatusAction stable-status.txt
Target //tools:heir-opt up-to-date:
  bazel-bin/tools/heir-opt
INFO: Elapsed time: 0.243s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/tools/heir-opt '--mlir-to-openfhe-ckks=entry-function=mlp' /usr/local/google/home/asraa/git/heir/tests/Examples/benchmark/mlp/mlp_inline.mlir --mlir-print-ir-after-all '--mlir-elide-elementsattrs-if-larger=128'
// -----// IR Dump After WrapGeneric (wrap-generic) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %cst = arith.constant dense<8.82341385> : tensor<1x1024xf32>
      %cst_0 = arith.constant dense<-8.664150e+01> : tensor<1x1024xf32>
      %cst_1 = arith.constant dense<388.964722> : tensor<1x1024xf32>
      %cst_2 = arith.constant dense<-797.090148> : tensor<1x1024xf32>
      %cst_3 = arith.constant dense<746.781677> : tensor<1x1024xf32>
      %cst_4 = arith.constant dense<-260.038666> : tensor<1x1024xf32>
      %cst_5 = arith.constant dense<5.000000e-01> : tensor<1x1024xf32>
      %cst_6 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
      %cst_7 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
      %1 = linalg.matmul ins(%input0, %cst_6 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_7 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %cst_8 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
      %cst_9 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst_8 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_9 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ConvertSecretExtractToStaticExtract (convert-secret-extract-to-static-extract) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ConvertSecretInsertToStaticInsert (convert-secret-insert-to-static-insert) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ConvertSecretWhileToStaticFor (convert-secret-while-to-static-for) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ConvertSecretForToStaticFor (convert-secret-for-to-static-for) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ConvertIfToSelect (convert-if-to-select) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After LinalgCanonicalizations (linalg-canonicalizations) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = linalg.matmul ins(%input0, %cst_1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %2 = func.call @external_relu(%1) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %3 = linalg.matmul ins(%2, %cst : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%cst_0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After LinalgToTensorExt (linalg-to-tensor-ext) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_1, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice_3 = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice_3 : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %extracted_slice = tensor.extract_slice %cst_0[1023, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
      %2 = arith.mulf %1#1, %extracted_slice : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_1, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice_3 = tensor.extract_slice %cst[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice_3 : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %extracted_slice_2 = tensor.extract_slice %cst[1023, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
      %6 = arith.mulf %5#1, %extracted_slice_2 : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ApplyFolders (apply-folders) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After InsertRotate (insert-rotate) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CollapseInsertionChains (collapse-insertion-chains) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After SCCP (sccp) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_2 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst_3 : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_2 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst_3 : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_2 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst_3 : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After RotateAndReduce (rotate-and-reduce) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_0[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_2 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst_3 : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After SCCP (sccp) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After ApplyFolders (apply-folders) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After OperationBalancer (operation-balancer) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After CSE (cse) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %input0) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_2[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %2 = arith.mulf %1#1, %cst_0 : tensor<1x1024xf32>
      %3 = arith.addf %1#0, %2 : tensor<1x1024xf32>
      %4 = func.call @external_relu(%3) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %5:2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3, %arg3 = %4) -> (tensor<1x1024xf32>, tensor<1x1024xf32>) {
        %extracted_slice = tensor.extract_slice %cst_1[%arg1, 0] [1, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<1x1024xf32>
        %8 = arith.mulf %arg3, %extracted_slice : tensor<1x1024xf32>
        %9 = arith.addf %arg2, %8 : tensor<1x1024xf32>
        %10 = tensor_ext.rotate %arg3, %c1 : tensor<1x1024xf32>, index
        affine.yield %9, %10 : tensor<1x1024xf32>, tensor<1x1024xf32>
      }
      %6 = arith.mulf %5#1, %cst : tensor<1x1024xf32>
      %7 = arith.addf %5#0, %6 : tensor<1x1024xf32>
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}


// -----// IR Dump After AnnotateMgmt (annotate-mgmt) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
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


// -----// IR Dump After SecretInsertMgmtCKKS (secret-insert-mgmt-ckks) //----- //
module {
  func.func private @external_relu(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @mlp(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1024xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1024x1024xf32>
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


/usr/local/google/home/asraa/git/heir/tests/Examples/benchmark/mlp/mlp_inline.mlir:14:10: error: 'affine.for' op 0-th region iter_arg and 0-th yielded value have different type: 'tensor<1x1024xf32>' != '!secret.secret<tensor<1x1024xf32>>'
    %0 = linalg.matmul ins(%arg0, %weight0 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%buffer0 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
         ^
/usr/local/google/home/asraa/git/heir/tests/Examples/benchmark/mlp/mlp_inline.mlir:14:10: note: see current operation: 
%6:2 = "affine.for"(%5, %arg0) <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = affine_map<() -> (1023)>}> ({
^bb0(%arg14: index, %arg15: tensor<1x1024xf32>, %arg16: !secret.secret<tensor<1x1024xf32>>):
  %25 = "tensor.extract_slice"(%4, %arg14) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>}> : (tensor<1024x1024xf32>, index) -> tensor<1x1024xf32>
  %26 = "secret.generic"(%arg16) ({
  ^bb0(%arg19: tensor<1x1024xf32>):
    %31 = "arith.mulf"(%arg19, %25) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    "secret.yield"(%31) : (tensor<1x1024xf32>) -> ()
  }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
  %27 = "secret.generic"(%26) ({
  ^bb0(%arg18: tensor<1x1024xf32>):
    %30 = "arith.addf"(%arg15, %arg18) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    "secret.yield"(%30) : (tensor<1x1024xf32>) -> ()
  }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
  %28 = "secret.generic"(%arg16) ({
  ^bb0(%arg17: tensor<1x1024xf32>):
    %29 = "tensor_ext.rotate"(%arg17, %3) : (tensor<1x1024xf32>, index) -> tensor<1x1024xf32>
    "secret.yield"(%29) : (tensor<1x1024xf32>) -> ()
  }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
  "affine.yield"(%27, %28) : (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) -> ()
}) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (tensor<1x1024xf32>, !secret.secret<tensor<1x1024xf32>>) -> (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>)
// -----// IR Dump After SecretDistributeGeneric Failed (secret-distribute-generic) //----- //
#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (1023)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x1024xf32>) -> tensor<1x1024xf32>, sym_name = "external_relu", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}], function_type = (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>, sym_name = "mlp"}> ({
  ^bb0(%arg0: !secret.secret<tensor<1x1024xf32>>):
    %0 = "arith.constant"() <{value = dense_resource<__elided__> : tensor<1x1024xf32>}> : () -> tensor<1x1024xf32>
    %1 = "arith.constant"() <{value = dense_resource<__elided__> : tensor<1x1024xf32>}> : () -> tensor<1x1024xf32>
    %2 = "arith.constant"() <{value = dense_resource<__elided__> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %3 = "arith.constant"() <{value = 1 : index}> : () -> index
    %4 = "arith.constant"() <{value = dense_resource<__elided__> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>
    %5 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<1x1024xf32>}> : () -> tensor<1x1024xf32>
    %6:2 = "affine.for"(%5, %arg0) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg14: index, %arg15: tensor<1x1024xf32>, %arg16: !secret.secret<tensor<1x1024xf32>>):
      %25 = "tensor.extract_slice"(%4, %arg14) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>}> : (tensor<1024x1024xf32>, index) -> tensor<1x1024xf32>
      %26 = "secret.generic"(%arg16) ({
      ^bb0(%arg19: tensor<1x1024xf32>):
        %31 = "arith.mulf"(%arg19, %25) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
        "secret.yield"(%31) : (tensor<1x1024xf32>) -> ()
      }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
      %27 = "secret.generic"(%26) ({
      ^bb0(%arg18: tensor<1x1024xf32>):
        %30 = "arith.addf"(%arg15, %arg18) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
        "secret.yield"(%30) : (tensor<1x1024xf32>) -> ()
      }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
      %28 = "secret.generic"(%arg16) ({
      ^bb0(%arg17: tensor<1x1024xf32>):
        %29 = "tensor_ext.rotate"(%arg17, %3) : (tensor<1x1024xf32>, index) -> tensor<1x1024xf32>
        "secret.yield"(%29) : (tensor<1x1024xf32>) -> ()
      }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
      "affine.yield"(%27, %28) : (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (tensor<1x1024xf32>, !secret.secret<tensor<1x1024xf32>>) -> (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>)
    %7 = "secret.generic"(%6#1) ({
    ^bb0(%arg13: tensor<1x1024xf32>):
      %24 = "arith.mulf"(%arg13, %1) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
      "secret.yield"(%24) : (tensor<1x1024xf32>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
    %8 = "secret.generic"(%6#0, %7) ({
    ^bb0(%arg11: tensor<1x1024xf32>, %arg12: tensor<1x1024xf32>):
      %23 = "arith.addf"(%arg11, %arg12) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
      "secret.yield"(%23) : (tensor<1x1024xf32>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
    %9 = "secret.generic"(%8) ({
    ^bb0(%arg10: tensor<1x1024xf32>):
      %22 = "func.call"(%arg10) <{callee = @external_relu}> : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      "secret.yield"(%22) : (tensor<1x1024xf32>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
    %10:2 = "affine.for"(%5, %9) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg4: index, %arg5: tensor<1x1024xf32>, %arg6: !secret.secret<tensor<1x1024xf32>>):
      %15 = "tensor.extract_slice"(%2, %arg4) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>}> : (tensor<1024x1024xf32>, index) -> tensor<1x1024xf32>
      %16 = "secret.generic"(%arg6) ({
      ^bb0(%arg9: tensor<1x1024xf32>):
        %21 = "arith.mulf"(%arg9, %15) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
        "secret.yield"(%21) : (tensor<1x1024xf32>) -> ()
      }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
      %17 = "secret.generic"(%16) ({
      ^bb0(%arg8: tensor<1x1024xf32>):
        %20 = "arith.addf"(%arg5, %arg8) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
        "secret.yield"(%20) : (tensor<1x1024xf32>) -> ()
      }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
      %18 = "secret.generic"(%arg6) ({
      ^bb0(%arg7: tensor<1x1024xf32>):
        %19 = "tensor_ext.rotate"(%arg7, %3) : (tensor<1x1024xf32>, index) -> tensor<1x1024xf32>
        "secret.yield"(%19) : (tensor<1x1024xf32>) -> ()
      }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
      "affine.yield"(%17, %18) : (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (tensor<1x1024xf32>, !secret.secret<tensor<1x1024xf32>>) -> (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>)
    %11 = "secret.generic"(%10#1) ({
    ^bb0(%arg3: tensor<1x1024xf32>):
      %14 = "arith.mulf"(%arg3, %0) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
      "secret.yield"(%14) : (tensor<1x1024xf32>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
    %12 = "secret.generic"(%10#0, %11) ({
    ^bb0(%arg1: tensor<1x1024xf32>, %arg2: tensor<1x1024xf32>):
      %13 = "arith.addf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
      "secret.yield"(%13) : (tensor<1x1024xf32>) -> ()
    }) {mgmt.mgmt = #mgmt.mgmt<level = 0>} : (!secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
    "func.return"(%12) : (!secret.secret<tensor<1x1024xf32>>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
}) : () -> ()


