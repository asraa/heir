// CHECK: module
module {
  func.func @test_add(%in: i4) -> (i4) {
    // FIXME: Change to Comb
    %0 = arith.addi %in, %in : i4
    %1 = arith.addi %0, %in : i4
    return %1 : i4
  }
}
