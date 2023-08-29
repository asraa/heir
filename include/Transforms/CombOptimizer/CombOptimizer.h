#ifndef HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_COMBOPTIMIZER_H_
#define HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_COMBOPTIMIZER_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

namespace heir {

std::unique_ptr<Pass> createABCPass(std::string_view runfiles);

#define GEN_PASS_DECL
#include "include/Transforms/CombOptimizer/CombOptimizer.h.inc"

#define GEN_PASS_REGISTRATION
#include "include/Transforms/CombOptimizer/CombOptimizer.h.inc"

}  // namespace heir

}  // namespace mlir

#endif  // HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_COMBOPTIMIZER_H_
