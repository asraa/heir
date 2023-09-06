#include "include/Transforms/CombOptimizer/CombOptimizer.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string_view>

#include "frontends/ast/ast.h"
#include "frontends/verilog/verilog_frontend.h"
#include "include/Target/Verilog/VerilogEmitter.h"
#include "include/circt/Dialect/Comb/CombDialect.h"  // from @circt
#include "include/circt/Dialect/Comb/CombOps.h"      // from @circt
#include "kernel/rtlil.h"
#include "kernel/yosys.h"
#include "lib/Transforms/CombOptimizer/ImportRTLIL.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineValueMap.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {
// Returns a list of cell names that are topologically ordered using the Yosys
// toder output. This is extracted from the lines containing cells in the
// output:
// -- Running command `torder -stop * P*;' --

// 14. Executing TORDER pass (print cells in topological order).
// module test_add
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$168
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$170
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$169
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$171
llvm::SmallVector<std::string, 10> getTopologicalOrder(
    std::stringstream &torderOutput) {
  llvm::SmallVector<std::string, 10> cells;
  std::string line;
  while (std::getline(torderOutput, line)) {
    auto lineCell = line.find("cell $");
    if (lineCell != std::string::npos) {
      cells.push_back(line.substr(lineCell + 5, std::string::npos));
    }
  }
  return cells;
}

}  // namespace

// in-memory representation of combinational circuit from the IR

// ABCPass
struct YosysABCPass
    : public mlir::PassWrapper<YosysABCPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  YosysABCPass(std::string_view runfiles) : runfiles(runfiles) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect, ::circt::comb::CombDialect>();
  }

  // FIXME: Works on a module: I'll need to support
  //   (1) Local optimizations (e.g. within for loops). So this would require
  //   constructing an in-memory "funcOp" for the for loop body (see
  //   ExtractLoopBody), and then passing to Yosys, and then retrieving the
  //   result back. Need to do "input/output" analysis for the local areas.
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    mlir::RewritePatternSet patterns(&getContext());

    // DONT Logs Yosys to stdout
    // Yosys::log_streams.push_back(&std::cout);
    Yosys::yosys_setup();
    Yosys::RTLIL::Design *design = Yosys::yosys_get_design();

    // TODO: It's probably easier to build the AST and use the AST conversion
    getOperation().walk([&](mlir::func::FuncOp op) {
      std::string result;
      llvm::raw_string_ostream stream(result);
      if (failed(translateToVerilog(op, stream))) {
        return WalkResult::interrupt();
      }
      char *filename = tmpnam(NULL);
      std::ofstream of(filename);
      of << result;
      of.close();
      std::vector<std::string> args = {"read_verilog", filename};
      Yosys::Pass::call(design, args);
      Yosys::run_pass(
          llvm::formatv("hierarchy -check -top \\{0}", op.getSymName()));
      return WalkResult::advance();
    });

    Yosys::run_pass("proc; memory; exec -- pwd;");
    Yosys::run_pass(
        llvm::formatv("techmap -map {0}/techlibs/techmap.v; opt;", runfiles));
    Yosys::run_pass(llvm::formatv(
        "abc -exe {0}/../edu_berkeley_abc/abc -lut 3; opt_clean -purge; clean",
        runfiles));
    Yosys::run_pass(llvm::formatv(
        "techmap -map "
        "{0}/../heir/lib/Transforms/CombOptimizer/map_lut_to_lut3.v; clean;",
        runfiles));
    Yosys::run_pass(
        "hierarchy -generate * o:Y o:Q i:*; opt; opt_clean -purge; clean");
    std::stringstream cellOrder;
    Yosys::log_streams.push_back(&cellOrder);
    Yosys::run_pass("torder -stop * P*;");
    Yosys::log_streams.clear();

    Yosys::log_streams.push_back(&std::cout);
    Yosys::run_pass("dump");

    auto importer = RTLILImporter(getTopologicalOrder(cellOrder));
    mlir::ModuleOp module = getOperation();
    for (auto func : module.getOps<func::FuncOp>()) {
      func->getBlock()->erase();
      auto block = func.addEntryBlock();
      auto builder = mlir::OpBuilder(func.getBody());
      auto combModuleOr = importer.convert(Yosys::yosys_get_design(), block,
                                           builder, &getContext());
      if (failed(combModuleOr)) {
        return;
      }
    }
    Yosys::yosys_shutdown();
    module.dump();
    // module.push_back(*combModuleOr.value().getOps<func::FuncOp>().begin());

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }

  mlir::StringRef getArgument() const final { return "abc-optimizer"; }

 private:
  int64_t valueCount;
  llvm::DenseMap<Value, Yosys::AST::AstNode *> valueToNode;
  std::string_view runfiles;

  Yosys::AST::AstNode *makeRange(int msb = 31, int lsb = 0,
                                 bool isSigned = true) {
    auto *range = new Yosys::AST::AstNode(Yosys::AST::AST_RANGE);
    range->children.push_back(Yosys::AST::AstNode::mkconst_int(msb, true));
    range->children.push_back(Yosys::AST::AstNode::mkconst_int(lsb, true));
    range->is_signed = isSigned;
    return range;
  }

  Yosys::AST::AstNode *makeOrGetWire(Value value, std::string_view prefix,
                                     bool isInput, bool isOutput,
                                     Yosys::AST::AstNode *currentASTMod) {
    if (!valueToNode.contains(value)) {
      auto *node = new Yosys::AST::AstNode(Yosys::AST::AST_WIRE);
      node->str = llvm::formatv("\\{0}{1}", prefix, ++valueCount);
      node->is_input = isInput;
      node->is_output = isOutput;
      node->is_signed = value.getType().isSignedInteger();
      node->children.push_back(
          makeRange(value.getType().getIntOrFloatBitWidth() - 1, 0, false));
      currentASTMod->children.push_back(node);
      valueToNode.insert(std::make_pair(value, node));
    }
    return valueToNode.at(value);
  }

  Yosys::AST::AstNode *makeOrGetWire(BlockArgument arg, bool isInput,
                                     bool isOutput,
                                     Yosys::AST::AstNode *currentASTMod) {
    return makeOrGetWire(arg, "arg", isInput, isOutput, currentASTMod);
  }

  Yosys::AST::AstNode *makeOrGetWire(Value value, bool isInput, bool isOutput,
                                     Yosys::AST::AstNode *currentASTMod) {
    return makeOrGetWire(value, "v", isInput, isOutput, currentASTMod);
  }
};

std::unique_ptr<Pass> createABCPass(std::string_view runfiles) {
  return std::make_unique<YosysABCPass>(runfiles);
}

}  // namespace heir
}  // namespace mlir
