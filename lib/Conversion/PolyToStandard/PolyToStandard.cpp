#include "include/Conversion/PolyToStandard/PolyToStandard.h"

#include "include/Dialect/Poly/IR/PolyOps.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "lib/Conversion/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "include/Conversion/PolyToStandard/PolyToStandard.h.inc"

namespace {

// buildPolyDivMod builds an implementation of Euclidean division to compute the
// remainder of the input tensor modulo the ideal polynomial.
Operation *buildPolyDivMod(ImplicitLocOpBuilder b, RankedTensorType input,
                           RingAttr ring) {
  // Require euclidean division with the ideal polynomial
  // Lowerings will fail if you don't have a prime coeff OR monic polynomial in
  // a commutative ring
  auto resultTy = RankedTensorType::get(
      {ring.ideal().getDegree()},
      b.getIntegerType(ring.coefficientModulus().getBitWidth()));
  return b.create<tensor::EmptyOp>(resultTy.getShape(),
                                   resultTy.getElementType());
}

}  // namespace

class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](PolyType type) -> Type {
      RingAttr attr = type.getRing();
      uint32_t idealDegree = attr.ideal().getDegree();
      IntegerType elementTy =
          IntegerType::get(ctx, attr.coefficientModulus().getBitWidth(),
                           IntegerType::SignednessSemantics::Signless);
      // We must remove the ring attribute on the tensor, since the
      // unrealized_conversion_casts cannot carry the poly.ring attribute
      // through.
      return RankedTensorType::get({idealDegree}, elementTy);
    });

    // We don't include any custom materialization ops because this lowering is
    // all done in a single pass. The dialect conversion framework works by
    // resolving intermediate (mid-pass) type conflicts by inserting
    // unrealized_conversion_cast ops, and only converting those to custom
    // materializations if they persist at the end of the pass. In our case,
    // we'd only need to use custom materializations if we split this lowering
    // across multiple passes.
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTy = typeConverter->convertType(op->getResultTypes()[0]);
    auto resultTensorTy = cast<RankedTensorType>(resultTy);
    auto resultShape = resultTensorTy.getShape()[0];
    auto resultEltTy = resultTensorTy.getElementType();

    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];
    auto inputEltTy = inputTensorTy.getElementType();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getOperands()[0];
    // Extend element type if needed.
    if (inputEltTy != resultEltTy) {
      // FromTensorOp verifies that the coefficient tensor's elements fit into
      // the polynomial.
      assert(inputEltTy.getIntOrFloatBitWidth() <
             resultEltTy.getIntOrFloatBitWidth());

      coeffValue = b.create<arith::ExtUIOp>(
          RankedTensorType::get(inputShape, resultEltTy), coeffValue);
    }

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
      coeffValue = b.create<tensor::PadOp>(
          resultTy, coeffValue, low, high,
          b.create<arith::ConstantOp>(rewriter.getIntegerAttr(resultEltTy, 0)),
          /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Convert add lowers a poly.add operation to arith operations. A poly.add
  // operation is defined within the polynomial ring. Coefficients are added
  // element-wise as elements of the ring, so they are performed modulo the
  // coefficient modulus.
  //
  // To perform modular addition, assume that `cmod` is the coefficient modulus
  // of the ring, and that `N` is the bitwidth used to store the ring elements.
  // This may be much larger than `log_2(cmod)`.
  //
  // Let `x` and `y` be the inputs to modular addition, then:
  //    c1, n1 = addui_extended(x, y)
  // If the coefficient modulus divides `2^N`, then return
  //    c0 = c1 % cmod
  // Otherwise, compute the adjusted result:
  //    c0 = ((c1 % cmod) + (n1 * 2^N % cmod)) % cmod
  //
  // Note that `(c1 % cmod) + (n1 * 2^N % cmod)` will not overflow mod `2^N`.
  // If it did, then it would require that `cmod > (2^N) / 2`.
  // This would imply that `2^N % cmod = 2^N - cmod`.
  // If the sum overflowed, then we would have
  //    ((c1 % cmod) + (2^N % cmod)) > 2^N
  //    ((c1 % cmod) + (2^N - cmod)) > 2^N
  //    ((c1 % cmod) > cmod
  // Which is a contradiction.
  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto type = adaptor.getLhs().getType();

    APInt mod =
        cast<PolyType>(op.getResult().getType()).getRing().coefficientModulus();
    auto cmod = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(cast<ShapedType>(type), {mod}));

    auto addExtendedOp =
        b.create<arith::AddUIExtendedOp>(adaptor.getLhs(), adaptor.getRhs());
    auto c1ModOp = b.create<arith::RemUIOp>(addExtendedOp->getResult(0), cmod);
    // If mod divides 2^N, c1modOp is our result.
    if (mod.isPowerOf2()) {
      rewriter.replaceOp(op, c1ModOp.getResult());
      return success();
    }
    // Otherwise, add (n1 * 2^N % cmod)
    APInt quotient, remainder;
    APInt bigMod = APInt(mod.getBitWidth() + 1, 2) << (mod.getBitWidth() - 1);
    APInt::udivrem(bigMod, mod.zext(bigMod.getBitWidth()), quotient, remainder);
    remainder = remainder.trunc(mod.getBitWidth());

    auto bitwidth = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(cast<ShapedType>(type), {remainder}));
    auto adjustOp = b.create<arith::AddIOp>(c1ModOp, bitwidth);

    auto selectOp = b.create<arith::SelectOp>(addExtendedOp.getResult(1),
                                              c1ModOp, adjustOp);
    // Mod the final result.
    rewriter.replaceOp(op, b.create<arith::RemUIOp>(selectOp, cmod));

    return success();
  }
};

// TODO(https://github.com/google/heir/issues/104): implement
struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  // Naive polynomial multiplication implementation.
  // 1-D convolution (extend bits?)
  // Remainder modulo ring ideal
  // n = batch size, c = channels, w = width
  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // TODO: Assumes single poly
    auto polyTy = dyn_cast<PolyType>(op.getResult().getType());
    if (!polyTy) {
      return failure();
    }

    // TODO: Upgrade to power of two
    auto convWidth = polyTy.getRing().coefficientModulus().getActiveBits() * 2;
    auto eltType = b.getIntegerType(convWidth);

    auto convDegree = 2 * polyTy.getRing().getIdeal().getDegree() + 1;
    auto convType = RankedTensorType::get({convDegree}, eltType);

    // Create 1-D convolution
    auto convOutput = b.create<tensor::EmptyOp>(convType.getShape(),
                                                convType.getElementType());
    linalg::Conv1DOp conv = b.create<linalg::Conv1DOp>(
        adaptor.getOperands(), ValueRange{convOutput.getResult()});

    // 2N + 1 sized result tensor -> reduce modulo ideal to get a N sized tensor

    // rewriter.replaceOp(op, buildPolyDivMod(b, convResult,
    // polyTy.getRing().getIdeal()));

    return success();
  }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ConversionTarget target(*context);
    PolyToStandardTypeConverter typeConverter(context);

    target.addLegalDialect<arith::ArithDialect>();

    // target.addIllegalDialect<PolyDialect>();
    target.addIllegalOp<FromTensorOp, ToTensorOp, AddOp>();
    // target.addIllegalOp<AddOp>();
    // target.addIllegalOp<MulOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertFromTensor, ConvertToTensor, ConvertAdd, ConvertMul>(
        typeConverter, context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    // TODO(https://github.com/google/heir/issues/143): Handle tensor of polys.
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace heir
}  // namespace mlir
