#include "lib/Transforms/CombOptimizer/ImportRTLIL.h"

#include "include/circt/Dialect/Comb/CombDialect.h"  // from @circt
#include "include/circt/Dialect/Comb/CombOps.h"      // from @circt
#include "kernel/rtlil.h"
#include "kernel/yosys.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

void RTLILImporter::addWireValue(Yosys::RTLIL::Wire *wire, mlir::Value value) {
  wireNameToValue.insert(std::make_pair(wire->name.str(), value));
}

Value RTLILImporter::getOrCreateValue(Yosys::RTLIL::Wire *wire) {
  auto wireName = wire->name.str();
  if (!wireNameToValue.contains(wireName)) {
    std::cout << "VALUE NOT FOUND " << std::endl;
    // Create a new Value?
    // wireNameToValue.insert(std::make_pair());
  }
  return wireNameToValue.at(wireName);
}

mlir::LogicalResult RTLILImporter::convert(Yosys::RTLIL::Design *design,
                                           mlir::Block *block,
                                           mlir::OpBuilder &builder,
                                           mlir::MLIRContext *context) {
  // mlir::ModuleOp module =
  //     mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

  for (auto *rModule : design->modules()) {
    if (rModule->get_blackbox_attribute()) {
      continue;
    }
    // Add a function for each module in the design.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    llvm::SmallVector<Yosys::RTLIL::Wire *, 4> args;
    llvm::SmallVector<mlir::Type, 4> retTypes;

    // Hold onto the mlir::Values that comprise a result wire, at the end we
    // have to concat these and return them.
    // FIXME: Make multiple, maybe I need to pass the output wire names to
    // preserve ordering.
    llvm::SmallVector<Yosys::RTLIL::Wire *, 4> ret;
    llvm::MapVector<Yosys::RTLIL::Wire *, llvm::SmallVector<mlir::Value>>
        retBitValues;

    for (auto *wire : rModule->wires()) {
      // Note: assert that there are no input+output wires.
      if (wire->port_input) {
        argTypes.push_back(builder.getIntegerType(wire->width));
        args.push_back(wire);
      } else if (wire->port_output) {
        retTypes.push_back(builder.getIntegerType(wire->width));
        retBitValues[wire].resize(wire->width);
        ret.push_back(wire);
      }
    }
    mlir::FunctionType funcType = builder.getFunctionType(argTypes, retTypes);
    auto funcName = rModule->name.str();
    auto function = mlir::func::FuncOp::create(
        mlir::UnknownLoc::get(context), funcName, funcType,
        llvm::ArrayRef<mlir::NamedAttribute>{});

    // Seed the builder with an initial block.
    builder.setInsertionPointToStart(block);
    // Map the wires to the block arguments' mlir::Values.
    for (auto i = 0; i < args.size(); i++) {
      addWireValue(args[i], block->getArgument(i));
    }

    // Convert cells into statements.
    for (const auto &cellName : cellOrdering) {
      auto *cell = rModule->cells_[cellName];
      std::cout << cell->name.str() << std::endl;
      llvm::SmallVector<bool> lutValues(8 /* FIXME LUT SIZE */, false);
      for (const auto &attr : cell->attributes) {
        // Create truth table out of the cell attribute.
        if (llvm::StringRef(attr.first.str()).contains("LUT")) {
          auto lutStr = attr.second.as_string();
          for (auto i = 0; i < lutStr.size(); i++) {
            lutValues[i] = (lutStr[i] == '1');
          }
        }
      }
      // Gather a list of the inputs and outputs.
      llvm::SmallVector<mlir::Value, 4> inputValues;

      // Gather inputs.
      for (const auto &conn : std::vector<std::string>{"\\A", "\\B", "\\C"}) {
        auto sigSpec = cell->connections().at(Yosys::RTLIL::IdString(conn));
        if (sigSpec.is_wire()) {
          std::cout << "is wire" << std::endl;
          std::cout << sigSpec.as_wire()->name.str() << std::endl;
          if (!wireNameToValue.contains(sigSpec.as_wire()->name.str())) {
            // Maybe create a placeholder value, and then replaceAllUsesWith if
            // it's the output wire.
            // builder.
            std::cout << "map does not contain wire "
                      << sigSpec.as_wire()->name.str();
          }
          inputValues.push_back(getOrCreateValue(sigSpec.as_wire()));
          std::cout << "pushed back input value " << std::endl;
        } else if (sigSpec.is_bit()) {
          if (sigSpec.as_bit().is_wire()) {
            std::cout << "is bit and wire" << std::endl;
            std::cout << sigSpec.as_bit().wire->name.str() << " "
                      << sigSpec.as_bit().offset << std::endl;
            auto argA = getOrCreateValue(sigSpec.as_bit().wire);
            auto extractOp = builder.create<circt::comb::ExtractOp>(
                function.getLoc(),
                builder.getIntegerType(argA.getType().getIntOrFloatBitWidth() -
                                       1),
                argA, sigSpec.as_bit().offset);
            inputValues.push_back(extractOp->getResult(0));
          }
        } else if (sigSpec.is_fully_const()) {
          std::cout << "SIG SPEC IS CONST" << std::endl;
        } else {
          std::cout << "unknown type" << std::endl;
        }
      }

      auto lookupTable =
          builder.getBoolArrayAttr(llvm::ArrayRef<bool>(lutValues));
      lookupTable.dump();
      auto truthOp = builder.create<circt::comb::TruthTableOp>(
          function.getLoc(), inputValues, lookupTable);
      // Hookup result with the \\Y connection result
      auto resultSigSpec =
          cell->connections().at(Yosys::RTLIL::IdString("\\Y"));
      if (resultSigSpec.is_wire()) {
        std::cout << "adding result wire to map "
                  << resultSigSpec.as_wire()->name.str() << std::endl;
        // Replace all uses with if it is already contained. If not, just add to
        // map.
        addWireValue(resultSigSpec.as_wire(), truthOp->getResult(0));
      } else if (resultSigSpec.is_bit()) {
        // Hold onto result bits? Will Yosys ever declare an intermediate
        // multi-bit? For now let's assume no and that this is the output. I can
        // add an assert.
        assert(retBitValues.contains(resultSigSpec.as_bit().wire));
        if (resultSigSpec.as_bit().is_wire()) {
          retBitValues[resultSigSpec.as_bit().wire]
                      [resultSigSpec.as_bit().offset] = truthOp->getResult(0);
        }
      }
    }

    // Wire up any remaining connections. These should (?) by output
    // connections.
    for (const auto &conn : rModule->connections()) {
      if (conn.first.is_wire()) {
        // This is a return value.
        if (retBitValues.contains(conn.first.as_wire())) {
          // Map return wire to input wire or input bit.
          if (conn.second.is_wire()) {
            addWireValue(conn.first.as_wire(),
                         getOrCreateValue(conn.second.as_wire()));
          } else if (conn.second.is_bit() && conn.second.as_bit().is_wire()) {
            // We are mapping a return wire to an input bit of a wire.
            auto arg = getOrCreateValue(conn.second.as_bit().wire);
            auto extractOp = builder.create<circt::comb::ExtractOp>(
                function.getLoc(), builder.getIntegerType(1), arg,
                conn.second.as_bit().offset);
            addWireValue(conn.first.as_wire(), extractOp->getResult(0));
          }
        }
      } else if (conn.first.is_bit() && conn.first.as_bit().is_wire()) {
        // This is a return value.
        if (retBitValues.contains(conn.first.as_bit().wire)) {
          if (conn.second.is_wire()) {
            retBitValues[conn.first.as_bit().wire][conn.first.as_bit().offset] =
                getOrCreateValue(conn.second.as_wire());
          } else if (conn.second.is_bit() && conn.second.as_bit().is_wire()) {
            // Map a return bit to a bit of the input.
            auto arg = getOrCreateValue(conn.second.as_bit().wire);
            auto extractOp = builder.create<circt::comb::ExtractOp>(
                function.getLoc(), builder.getIntegerType(1), arg,
                conn.second.as_bit().offset);
            retBitValues[conn.first.as_bit().wire][conn.first.as_bit().offset] =
                extractOp->getResult(0);
          }
        }
      }
    }

    // Concat result bits and add func::ReturnOp
    for (const auto &[resultWire, retBits] : retBitValues) {
      if (retBits.size() > 1) {
        for (auto ret : retBits) {
          ret.dump();
        }
        std::cout << "add concat op" << std::endl;
        // Insert concat op and return that
        auto concatOp =
            builder.create<circt::comb::ConcatOp>(function.getLoc(), retBits);
        builder.create<func::ReturnOp>(function.getLoc(), concatOp.getResult());
      } else {
        // Otherwise get ret wire from the map and return that. It must be an
        // output.
        builder.create<func::ReturnOp>(function.getLoc(),
                                       getOrCreateValue(resultWire));
      }
    }

    // When to add return statement?
    function.dump();
  }
  return success();
}

}  // namespace heir
}  // namespace mlir
