#include "include/Dialect/Secret/IR/SecretTypes.h"

#include "include/Dialect/Secret/IR/SecretPatterns.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

Type SecretType::castToSecretType(Type valueType) {
  if (llvm::isa<IntegerType>(valueType)) {
    return SecretType::get(valueType);
  }
  ShapedType shapedType = llvm::dyn_cast<ShapedType>(valueType);
  auto elementType = llvm::dyn_cast<IntegerType>(shapedType.getElementType());
  assert(elementType && "expected underlying integer element type");
  return shapedType.cloneWith(shapedType.getShape(),
                              SecretType::get(elementType));
}

Type SecretType::castFromSecretType(Type valueType) {
  if (auto secretType = llvm::dyn_cast<SecretType>(valueType)) {
    return secretType.getValueType();
  }
  ShapedType shapedType = llvm::dyn_cast<ShapedType>(valueType);
  if (shapedType) {
    auto secretType = llvm::dyn_cast<SecretType>(shapedType.getElementType());
    assert(secretType && "expected underlying secret element type");
    return shapedType.cloneWith(shapedType.getShape(), castFromSecretType(secretType));
  }
  return nullptr;
}

bool SecretType::isSecretType(Type candidateType) {
  if (llvm::isa<SecretType>(candidateType)) {
    return true;
  }
  if (ShapedType shapedType = llvm::dyn_cast<ShapedType>(candidateType)) {
    return llvm::isa<SecretType>(shapedType.getElementType());
  }
  return false;
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
