// RUN: heir-opt --comb-to-cggi -cse %s | FileCheck %s

// This test was produced by running
//   heir-opt --yosys-optimizer tests/yosys_optimizer/add_one.mlir

module {
  func.func @add_one(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    %c1_i8 = arith.constant 1 : i8
    %0 = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<tensor<8xi1>>
    %c0_i8 = arith.constant 0 : i8
    %1 = arith.andi %c1_i8, %c1_i8 : i8
    %2 = arith.shrsi %1, %c0_i8 : i8
    %3 = arith.trunci %2 : i8 to i1
    %c2_i8 = arith.constant 2 : i8
    %4 = arith.andi %c1_i8, %c2_i8 : i8
    %5 = arith.shrsi %4, %c1_i8 : i8
    %6 = arith.trunci %5 : i8 to i1
    %c4_i8 = arith.constant 4 : i8
    %7 = arith.andi %c1_i8, %c4_i8 : i8
    %8 = arith.shrsi %7, %c2_i8 : i8
    %9 = arith.trunci %8 : i8 to i1
    %c3_i8 = arith.constant 3 : i8
    %c8_i8 = arith.constant 8 : i8
    %10 = arith.andi %c1_i8, %c8_i8 : i8
    %11 = arith.shrsi %10, %c3_i8 : i8
    %12 = arith.trunci %11 : i8 to i1
    %c16_i8 = arith.constant 16 : i8
    %13 = arith.andi %c1_i8, %c16_i8 : i8
    %14 = arith.shrsi %13, %c4_i8 : i8
    %15 = arith.trunci %14 : i8 to i1
    %c5_i8 = arith.constant 5 : i8
    %c32_i8 = arith.constant 32 : i8
    %16 = arith.andi %c1_i8, %c32_i8 : i8
    %17 = arith.shrsi %16, %c5_i8 : i8
    %18 = arith.trunci %17 : i8 to i1
    %c6_i8 = arith.constant 6 : i8
    %c64_i8 = arith.constant 64 : i8
    %19 = arith.andi %c1_i8, %c64_i8 : i8
    %20 = arith.shrsi %19, %c6_i8 : i8
    %21 = arith.trunci %20 : i8 to i1
    %c7_i8 = arith.constant 7 : i8
    %c-128_i8 = arith.constant -128 : i8
    %22 = arith.andi %c1_i8, %c-128_i8 : i8
    %23 = arith.shrsi %22, %c7_i8 : i8
    %24 = arith.trunci %23 : i8 to i1
    %from_elements = tensor.from_elements %3, %6, %9, %12, %15, %18, %21, %24 : tensor<8xi1>
    %25 = secret.generic ins(%0, %from_elements : !secret.secret<tensor<8xi1>>, tensor<8xi1>) {
    ^bb0(%arg1: tensor<8xi1>, %arg2: tensor<8xi1>):
      %c0 = arith.constant 0 : index
      %extracted = tensor.extract %arg1[%c0] : tensor<8xi1>
      %extracted_0 = tensor.extract %arg2[%c0] : tensor<8xi1>
      %false = arith.constant false
      %27 = comb.truth_table %extracted, %extracted_0, %false -> 8 : ui8
      %c1 = arith.constant 1 : index
      %extracted_1 = tensor.extract %arg1[%c1] : tensor<8xi1>
      %extracted_2 = tensor.extract %arg2[%c1] : tensor<8xi1>
      %28 = comb.truth_table %27, %extracted_1, %extracted_2 -> 150 : ui8
      %29 = comb.truth_table %27, %extracted_1, %extracted_2 -> 23 : ui8
      %c2 = arith.constant 2 : index
      %extracted_3 = tensor.extract %arg1[%c2] : tensor<8xi1>
      %extracted_4 = tensor.extract %arg2[%c2] : tensor<8xi1>
      %30 = comb.truth_table %29, %extracted_3, %extracted_4 -> 43 : ui8
      %c3 = arith.constant 3 : index
      %extracted_5 = tensor.extract %arg1[%c3] : tensor<8xi1>
      %extracted_6 = tensor.extract %arg2[%c3] : tensor<8xi1>
      %31 = comb.truth_table %30, %extracted_5, %extracted_6 -> 43 : ui8
      %c4 = arith.constant 4 : index
      %extracted_7 = tensor.extract %arg1[%c4] : tensor<8xi1>
      %extracted_8 = tensor.extract %arg2[%c4] : tensor<8xi1>
      %32 = comb.truth_table %31, %extracted_7, %extracted_8 -> 43 : ui8
      %c5 = arith.constant 5 : index
      %extracted_9 = tensor.extract %arg1[%c5] : tensor<8xi1>
      %extracted_10 = tensor.extract %arg2[%c5] : tensor<8xi1>
      %33 = comb.truth_table %32, %extracted_9, %extracted_10 -> 43 : ui8
      %c6 = arith.constant 6 : index
      %extracted_11 = tensor.extract %arg1[%c6] : tensor<8xi1>
      %extracted_12 = tensor.extract %arg2[%c6] : tensor<8xi1>
      %34 = comb.truth_table %33, %extracted_11, %extracted_12 -> 105 : ui8
      %35 = comb.truth_table %33, %extracted_11, %extracted_12 -> 43 : ui8
      %c7 = arith.constant 7 : index
      %extracted_13 = tensor.extract %arg1[%c7] : tensor<8xi1>
      %extracted_14 = tensor.extract %arg2[%c7] : tensor<8xi1>
      %36 = comb.truth_table %35, %extracted_13, %extracted_14 -> 105 : ui8
      %37 = comb.truth_table %extracted, %extracted_0, %false -> 6 : ui8
      %38 = comb.truth_table %29, %extracted_3, %extracted_4 -> 105 : ui8
      %39 = comb.truth_table %30, %extracted_5, %extracted_6 -> 105 : ui8
      %40 = comb.truth_table %31, %extracted_7, %extracted_8 -> 105 : ui8
      %41 = comb.truth_table %32, %extracted_9, %extracted_10 -> 105 : ui8
      %from_elements_15 = tensor.from_elements %36, %34, %41, %40, %39, %38, %28, %37 : tensor<8xi1>
      secret.yield %from_elements_15 : tensor<8xi1>
    } -> !secret.secret<tensor<8xi1>>
    %26 = secret.cast %25 : !secret.secret<tensor<8xi1>> to !secret.secret<i8>
    return %26 : !secret.secret<i8>
  }
}
