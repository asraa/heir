#ifndef HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_RTLILFRONTEND_H_
#define HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_RTLILFRONTEND_H_

#include <utility>

#include "kernel/rtlil.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

// Translates the given RTLIL design to an MLIR module.
// We use RTLIL
mlir::FailureOr<mlir::ModuleOp> convertRtlilToMlir(Yosys::RTLIL::Design *design,
                                                   mlir::MLIRContext *context);

class RTLILImporter {
 public:
  RTLILImporter(llvm::SmallVector<std::string, 10> cellOrdering)
      : cellOrdering(cellOrdering) {}

  mlir::LogicalResult convert(Yosys::RTLIL::Design *design, mlir::Block *block,
                              mlir::OpBuilder &builder,
                              mlir::MLIRContext *context);

 private:
  llvm::StringMap<Value> wireNameToValue;
  llvm::SmallVector<std::string, 10> cellOrdering;

  Value getOrCreateValue(Yosys::RTLIL::Wire *wire);

  void addWireValue(Yosys::RTLIL::Wire *wire, mlir::Value value);
};

}  // namespace heir
}  // namespace mlir

#endif  // HEIR_INCLUDE_TRANSFORMS_COMBOPTIMIZER_RTLILFRONTEND_H_
