#include "include/Transforms/YosysOptimizer/YosysOptimizer.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/Secret/IR/SecretOps.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "lib/Transforms/YosysOptimizer/LUTImporter.h"
#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "yosysoptimizer"

namespace mlir {
namespace heir {
using std::string;

#define GEN_PASS_DEF_YOSYSOPTIMIZER
#include "include/Transforms/YosysOptimizer/YosysOptimizer.h.inc"

// $0: verilog filename
// $1: function name
// $2: yosys runfiles
// $3: abc path
// $4: abc fast option -fast
constexpr std::string_view kYosysTemplate = R"(
read_verilog {0};
hierarchy -check -top \{1};
proc; memory;
techmap -map {2}/techmap.v; opt;
abc -exe {3} -lut 3 {4};
opt_clean -purge;
rename -hide */c:*; rename -enumerate */c:*;
techmap -map {2}/map_lut_to_lut3.v; opt_clean -purge;
hierarchy -generate * o:Y i:*; opt; opt_clean -purge;
clean;
)";

struct YosysOptimizer : public impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;

  YosysOptimizer(std::string yosysFilesPath, std::string abcPath, bool abcFast)
      : yosysFilesPath(yosysFilesPath), abcPath(abcPath), abcFast(abcFast) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override;

 private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
};

tensor::FromElementsOp convertIntegerValue(Value value, Type convertedType,
                                           OpBuilder &b, Location loc) {
  IntegerType argType = value.getType().cast<IntegerType>();
  int width = argType.getWidth();
  SmallVector<Value> extractedBits;
  extractedBits.reserve(width);

  for (int i = 0; i < width; i++) {
    // These arith ops correspond to extracting the i-th bit
    // from the input
    auto shiftAmount =
        b.create<arith::ConstantOp>(loc, argType, b.getIntegerAttr(argType, i));
    auto bitMask = b.create<arith::ConstantOp>(
        loc, argType, b.getIntegerAttr(argType, 1 << i));
    auto andOp = b.create<arith::AndIOp>(loc, value, bitMask);
    auto shifted = b.create<arith::ShRSIOp>(loc, andOp, shiftAmount);
    extractedBits.push_back(
        b.create<arith::TruncIOp>(loc, b.getI1Type(), shifted));
  }

  return b.create<tensor::FromElementsOp>(loc, convertedType,
                                          ValueRange{extractedBits});
}

/// Convert a secret.generic's operands secret.secret<i3>
/// to secret.secret<tensor<3xi1>>.
LogicalResult convertOpOperands(secret::GenericOp op, func::FuncOp func,
                                SmallVector<Value> &typeConvertedArgs) {
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type convertedType =
        func.getFunctionType()
            .getInputs()[opOperand.getOperandNumber()];  // tensor<i8>

    if (!secret::SecretType::isSecretType(opOperand.get().getType())) {
      // The type is not secret, but still must be booleanized
      OpBuilder builder(op);
      auto fromElementsOp = convertIntegerValue(opOperand.get(), convertedType,
                                                builder, op.getLoc());
      typeConvertedArgs.push_back(fromElementsOp.getResult());
      continue;
    }

    Type originalType =
       secret::SecretType::castFromSecretType(opOperand.get().getType());  // secret<i8>
    if (!originalType.isa<IntegerType>()) {
      op.emitError() << "Unsupported input type to secret.generic: "
                     << originalType;
      return failure();
    }

    // Insert a conversion from the original type to the converted type
    OpBuilder builder(op);
    typeConvertedArgs.push_back(builder.create<secret::CastOp>(
        op.getLoc(), secret::SecretType::castToSecretType(convertedType),
        opOperand.get()));
  }

  return success();
}

/// Convert a secret.generic's results from tensor<3xsecret.secret<i1>>
/// to secret.secret<i3>.
LogicalResult convertOpResults(secret::GenericOp op,
                               DenseSet<Operation *> &castOps,
                               SmallVector<Value> &typeConvertedResults) {
  for (Value opResult : op.getResults()) {
    // The secret.yield verifier ensures generic can only return secret types.
    assert(secret::SecretType::isSecretType(opResult.getType()));

    secret::SecretType convertedType = opResult.getType()
                                           .cast<RankedTensorType>()
                                           .getElementType()
                                           .cast<secret::SecretType>();
    if (!convertedType.getValueType().isa<IntegerType>()) {
      op.emitError() << "While booleanizing secret.generic, found converted "
                        "type that cannot be reassembled: "
                     << convertedType;
      return failure();
    }

    IntegerType elementType = convertedType.getValueType().cast<IntegerType>();
    if (elementType.getWidth() != 1) {
      op.emitError() << "Converted element type must be i1";
      return failure();
    }

    IntegerType reassembledType = IntegerType::get(
        op.getContext(),
        elementType.getWidth() *
            opResult.getType().cast<RankedTensorType>().getNumElements());

    // Insert a reassembly of the original integer type from its booleanized
    // tensor version.
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto castOp = builder.create<secret::CastOp>(
        op.getLoc(), secret::SecretType::castToSecretType(reassembledType),
        opResult);
    castOps.insert(castOp);
    typeConvertedResults.push_back(castOp.getOutput());
  }

  return success();
}

LogicalResult runOnGenericOp(MLIRContext *context, secret::GenericOp op,
                             const std::string &yosysFilesPath,
                             const std::string &abcPath, bool abcFast) {
  std::string moduleName = "generic_body";

  // Translate function to Verilog. Translation will fail if the func contains
  // unsupported operations.
  // TODO(https://github.com/google/heir/issues/111): Directly convert MLIR to
  // Yosys' AST instead of using Verilog.
  //
  // After that is done, it might make sense to rewrite this as a
  // RewritePattern, which only runs if the body does not contain any comb ops,
  // and generalize this to support converting a secret.generic as well as a
  // func.func. It's necessary to wait for the migration because the Yosys API
  // used here maintains global state that apparently does not play nicely with
  // the instantiation of multiple rewrite patterns.
  char *filename = std::tmpnam(nullptr);
  std::error_code ec;
  llvm::raw_fd_ostream of(filename, ec);
  if (failed(translateToVerilog(op, of, moduleName,
                                /*allowSecretOps=*/true)) ||
      ec) {
    op.emitError() << "Failed to translate to verilog";
    of.close();
    return failure();
  }
  of.close();

  // Invoke Yosys to translate to a combinational circuit and optimize.
  Yosys::yosys_setup();
  Yosys::log_error_stderr = true;
  LLVM_DEBUG(Yosys::log_streams.push_back(&std::cout));
  Yosys::run_pass(llvm::formatv(kYosysTemplate.data(), filename, moduleName,
                                yosysFilesPath, abcPath,
                                abcFast ? "-fast" : ""));

  // Translate Yosys result back to MLIR and insert into the func
  LLVM_DEBUG(Yosys::run_pass("dump;"));
  std::stringstream cellOrder;
  Yosys::log_streams.push_back(&cellOrder);
  Yosys::run_pass("torder -stop * P*;");
  Yosys::log_streams.clear();
  auto topologicalOrder = getTopologicalOrder(cellOrder);
  LUTImporter lutImporter = LUTImporter(context);
  Yosys::RTLIL::Design *design = Yosys::yosys_get_design();
  func::FuncOp func =
      lutImporter.importModule(design->top_module(), topologicalOrder);
  Yosys::yosys_shutdown();

  // The pass changes the yielded value types, e.g., from an i8 to a
  // tensor<8xi1>. So the containing secret.generic needs to be updated and
  // conversions implemented on either side to convert the ints to tensors
  // and back again.
  //
  // convertOpOperands goes from i8 -> tensor.tensor<8xi1>
  // converOpResults from tensor.tensor<8xi1> -> i8
  SmallVector<Value> typeConvertedArgs;
  typeConvertedArgs.reserve(op->getNumOperands());
  if (failed(convertOpOperands(op, func, typeConvertedArgs))) {
    return failure();
  }

  int resultIndex = 0;
  for (Type ty : func.getFunctionType().getResults())
    op->getResult(resultIndex++)
        .setType(secret::SecretType::castToSecretType(ty));

  // Replace the func.return with a secret.yield
  op.getRegion().takeBody(func.getBody());
  op.getOperation()->setOperands(typeConvertedArgs);

  Block &block = op.getRegion().getBlocks().front();
  func::ReturnOp returnOp = cast<func::ReturnOp>(block.getTerminator());
  OpBuilder bodyBuilder(&block, block.end());
  bodyBuilder.create<secret::YieldOp>(returnOp.getLoc(),
                                      returnOp.getOperands());
  returnOp.erase();
  func.erase();

  DenseSet<Operation *> castOps;
  SmallVector<Value> typeConvertedResults;
  castOps.reserve(op->getNumResults());
  typeConvertedResults.reserve(op->getNumResults());
  if (failed(convertOpResults(op, castOps, typeConvertedResults))) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Generic results: " << typeConvertedResults.size()
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Original results: " << op.getResults().size()
                          << "\n");

  op.getResults().replaceUsesWithIf(
      typeConvertedResults, [&](OpOperand &operand) {
        return !castOps.contains(operand.getOwner());
      });
  return success();
}

// Optimize the body of a secret.generic op.
// FIXME: consider utilizing
// https://mlir.llvm.org/docs/PassManagement/#dynamic-pass-pipelines
void YosysOptimizer::runOnOperation() {
  auto result = getOperation()->walk([&](secret::GenericOp op) {
    if (failed(runOnGenericOp(&getContext(), op, yosysFilesPath, abcPath,
                              abcFast))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createYosysOptimizer(
    const std::string &yosysFilesPath, const std::string &abcPath,
    bool abcFast) {
  return std::make_unique<YosysOptimizer>(yosysFilesPath, abcPath, abcFast);
}

void registerYosysOptimizerPipeline(const std::string &yosysFilesPath,
                                    const std::string &abcPath) {
  PassPipelineRegistration<YosysOptimizerPipelineOptions>(
      "yosys-optimizer", "The yosys optimizer pipeline.",
      [yosysFilesPath, abcPath](OpPassManager &pm,
                                const YosysOptimizerPipelineOptions &options) {
        pm.addPass(
            createYosysOptimizer(yosysFilesPath, abcPath, options.abcFast));
        pm.addPass(mlir::createCSEPass());
      });
}

}  // namespace heir
}  // namespace mlir
