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
#include "mlir/Transforms/InliningUtils.h"
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


struct ForceOfKnownApPattern : public mlir::OpRewritePattern<ForceOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ForceOfKnownApPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ForceOp force,
                  mlir::PatternRewriter &rewriter) const override {

    ModuleOp mod = force.getParentOfType<ModuleOp>();
    HaskFuncOp fn = force.getParentOfType<HaskFuncOp>();

    ApOp ap = force.getOperand().getDefiningOp<ApOp>();
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
  }
};

struct ForceOfThunkifyPattern : public mlir::OpRewritePattern<ForceOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ForceOfThunkifyPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ForceOp force,
                  mlir::PatternRewriter &rewriter) const override {
    HaskFuncOp fn = force.getParentOfType<HaskFuncOp>();
    ThunkifyOp thunkify = force.getOperand().getDefiningOp<ThunkifyOp>();
    if (!thunkify) {
      return failure();
    }
//    assert(false && "force of thunkify");
    rewriter.replaceOp(force, thunkify.getOperand());
    return success();
  }
};

struct InlineApEagerPattern : public mlir::OpRewritePattern<ApEagerOp> {
  InlineApEagerPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ApEagerOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult
  matchAndRewrite(ApEagerOp ap,
                  mlir::PatternRewriter &rewriter) const override {

    HaskFuncOp parent = ap.getParentOfType<HaskFuncOp>();
    ModuleOp mod = ap.getParentOfType<ModuleOp>();

    auto ref = ap.getFn().getDefiningOp<HaskRefOp>();
    if (!ref) {
      return failure();
    }
    llvm::errs() << ap << "\n";
    if (parent.getName() == ref.getRef()) {

      llvm::errs() << "  - RECURSIVE: |" << ap << "|\n";
      return failure();
    }

    HaskFuncOp called = mod.lookupSymbol<HaskFuncOp>(ref.getRef());
    assert(called && "unable to find called function.");

    BlockAndValueMapping mapper;
    assert(called.getBody().getNumArguments() == ap.getNumFnArguments() &&
           "argument arity mismatch");
    for(int i = 0; i < ap.getNumFnArguments(); ++i ) {
      mapper.map(called.getBody().getArgument(i), ap.getFnArgument(i));
    }

    // TODO: setup mapping for arguments in mapper
    // This is not safe! Fuck me x(
    // consider f () { stmt; f(); } | g() { f (); }
    // this will expand into
    //   g() { f(); } -> g() { stmt; f(); } -> g { stmt; stmt; f(); } -> ...
    InlinerInterface inliner(rewriter.getContext());
    if (true) {
      LogicalResult isInlined =
          inlineRegion(inliner, &called.getBody(), ap, ap.getFnArguments(), ap.getResult());
      assert(succeeded(isInlined) && "unable to inline");
    }

//    rewriter.eraseOp(ap);
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
    patterns.insert<ForceOfKnownApPattern>(&getContext());
//    patterns.insert<ForceOfThunkifyPattern>(&getContext());
    patterns.insert<InlineApEagerPattern>(&getContext());

    llvm::errs() << "===Enabling Debugging...===\n";
    ::llvm::DebugFlag = true;


    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patterns))) {
      llvm::errs() << "===Worker wrapper failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
    };

    llvm::errs() << "===Disabling Debugging...===\n";
    ::llvm::DebugFlag = false;

  };
};

std::unique_ptr<mlir::Pass> createWorkerWrapperPass() {
  return std::make_unique<WorkerWrapperPass>();
}

} // namespace standalone
} // namespace mlir

