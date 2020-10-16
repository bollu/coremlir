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

struct ForceOfKnownApCanonicalizationPattern
    : public mlir::OpRewritePattern<ForceOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ForceOfKnownApCanonicalizationPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ForceOp force,
                  mlir::PatternRewriter &rewriter) const override {
    ModuleOp mod = force.getParentOfType<ModuleOp>();
    HaskFuncOp fn = force.getParentOfType<HaskFuncOp>();


    ApOp ap = force.getResult().getDefiningOp<ApOp>();
    if (!ap) {
      return failure();
    }
    HaskRefOp ref = ap.getFn().getDefiningOp<HaskRefOp>();
    if (!ref) {
      return failure();
    }

    llvm::errs() << "\nref: " << ref << "\n"
                 << "\nap: " << ap << "\n"
                 << "\nforce: " << force << " \n";

    HaskFuncOp forcedFn = mod.lookupSymbol<HaskFuncOp>(ref.getRef());

    // cannot inline a recursive function. Can replace with
    // an apEager
    if (forcedFn.isRecursive()) {
      rewriter.setInsertionPoint(ap);
      ApEagerOp eager =
          rewriter.create<ApEagerOp>(ap.getLoc(), ref, ap.getFnArguments());
      rewriter.replaceOp(force, eager.getResult());
      return success();
    }

    Block *forcedFnBB = forcedFn.getBodyBB();
    // Block &forcedFnBB = forcedFn.getLambda().getBodyBB();

    llvm::errs() << "\nforced fn body:\n-------\n";
    forcedFnBB->dump();

    llvm::errs() << "\nforce parent(original):\n----\n";
    force.getParentOfType<HaskFuncOp>().dump();

    //    Block *clonedBB = cloneBlock(*forcedFnBB);
    Block *clonedBB = nullptr;
    assert(false && "trying to manipulate BBs");
    clonedBB->insertBefore(force.getOperation()->getBlock());
    llvm::errs() << "\nforced called fn(cloned BB):\n----\n";
    clonedBB->dump();
    HaskReturnOp ret = dyn_cast<HaskReturnOp>(clonedBB->getTerminator());

    llvm::errs() << "\nforce parent(inlined):\n-----------\n";
    force.getParentOfType<HaskFuncOp>().dump();
    llvm::errs() << "\n";

    llvm::errs() << "\nforce parent(inlined+merged):\n-----------\n";
    rewriter.mergeBlockBefore(clonedBB, force.getOperation(),
                              ap.getFnArguments());
    force.getParentOfType<HaskFuncOp>().dump();
    llvm::errs() << "\n";

    llvm::errs() << "\nreturnop:\n------\n" << ret << "\n";

    llvm::errs()
        << "\nforce parent(inlined+merged+force-replaced):\n-----------\n";
    rewriter.replaceOp(force, ret.getOperand());
    rewriter.eraseOp(ret);
    fn.dump();
    llvm::errs() << "\n";
    return success();
  }
};

struct ForceOfThunkifyCanonicalizationPattern
    : public mlir::OpRewritePattern<ForceOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ForceOfThunkifyCanonicalizationPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ForceOp force,
                  mlir::PatternRewriter &rewriter) const override {
    HaskFuncOp fn = force.getParentOfType<HaskFuncOp>();
    ThunkifyOp thunkify = force.getOperand().getDefiningOp<ThunkifyOp>();
    if (!thunkify) {
      return failure();
    }
    rewriter.replaceOp(force, thunkify.getOperand());
    return success();
  }
};


struct WorkerWrapperPass : public Pass {
  WorkerWrapperPass() : Pass(mlir::TypeID::get<WorkerWrapperPass>()){};
  StringRef getName() const override { return "WorkerWrapperPass"; }

  std::unique_ptr<Pass> clonePass() const override {
    auto newInst = std::make_unique<WorkerWrapperPass>(
        *static_cast<const WorkerWrapperPass *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  void runOnOperation() {
    mlir::OwningRewritePatternList patterns;
    patterns.insert<ForceOfKnownApCanonicalizationPattern>(&getContext());
    patterns.insert<ForceOfThunkifyCanonicalizationPattern>(&getContext());
    if (failed(mlir::applyPatternsAndFoldGreedily (getOperation(), patterns))) {
      llvm::errs() << "===Worker wrapper failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
    };
  };
};


std::unique_ptr<mlir::Pass> createWorkerWrapperPass() {
  return std::make_unique<WorkerWrapperPass>();
}

} // namespace standalone
} // namespace mlir

