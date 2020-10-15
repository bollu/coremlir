#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/Parser.h>
#include <sstream>

// Standard dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

// pattern matching
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

// dilect lowering
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "hask-ops"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace standalone {

struct OptimizeHaskPass : public Pass {
  OptimizeHaskPass() : Pass(mlir::TypeID::get<OptimizeHaskPass>()){};
  StringRef getName() const override { return "LowerHaskToStandardPass"; }

  std::unique_ptr<Pass> clonePass() const override {
    auto newInst = std::make_unique<OptimizeHaskPass>(
        *static_cast<const OptimizeHaskPass *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  void runOnOperation() {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addIllegalDialect<HaskDialect>();
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::OwningRewritePatternList patterns;
    // patterns.insert<MakeDataConstructorOpConversionPattern>(&getContext());
    // patterns.insert<HaskADTOpConversionPattern>(&getContext());

    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
    if (failed(mlir::applyFullConversion(getOperation(), target, patterns))) {
      llvm::errs() << "===Hask optimization failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
    };
  };
};

std::unique_ptr<mlir::Pass> createHaskOptimizationPass() {
  return std::make_unique<OptimizeHaskPass>();
}

} // namespace standalone
} // namespace mlir

