#ifndef HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_COMBOPTIMIZER_TD_
#define HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_COMBOPTIMIZER_TD_

include "mlir/Pass/PassBase.td"

def ABCPass : Pass<"abc-optimize", "mlir::ModuleOp"> {
  let summary = "ABCPass uses Yosys and ABC to optimize a circuit";

  let description = [{
    This pass can be used to optimize combinational circuit.
  }];

  let constructor = "createABCPass(\"\")";

  let dependentDialects = [
    "::circt::comb::CombDialect",
  ];
}

#endif  // HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_COMBOPTIMIZER_TD_
