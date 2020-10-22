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
    rewriter.setInsertionPoint(ap);
    ApEagerOp eager =
        rewriter.create<ApEagerOp>(ap.getLoc(), ref, ap.getFnArguments());
    rewriter.replaceOp(force, eager.getResult());
    return success();
  };
};

// outline stuff that occurs after the force of a constructor at the
// top-level of a function.
struct OutlineUknownForcePattern : public mlir::OpRewritePattern<ForceOp> {
  OutlineUknownForcePattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ForceOp force,
                  mlir::PatternRewriter &rewriter) const override {

    // we should not have a force of thunkify, or force of known ap
    if (force.getOperand().getDefiningOp<ThunkifyOp>()) {
      return failure();
    }

    // TODO: think about what to do.
    if (ApOp ap = force.getOperand().getDefiningOp<ApOp>()) {
      if (HaskRefOp ref = ap.getFn().getDefiningOp<HaskRefOp>()) {
        return failure();
      }
    }

    // how to get dominance information? I should outline everything in the
    // region that is dominated by the BB that `force` lives in.
    // For now, approximate.
    if (force.getOperation()->getBlock() != &force.getParentRegion()->front()) {
      assert(false && "force not in entry BB");
      return failure();
    }
    llvm::errs() << "- UNK FORCE: " << force << "\n";

    // is this going to break *completely*? or only partially?
    std::unique_ptr<Region> r = std::make_unique<Region>();
    // create a hask func op.
    HaskFuncOp parentfn = force.getParentOfType<HaskFuncOp>();

    ModuleOp module = parentfn.getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(&module.getBodyRegion().front());

    HaskFuncOp outlinedFn = rewriter.create<HaskFuncOp>(
        force.getLoc(), parentfn.getName().str() + "_outline",
        parentfn.getFunctionType());

    rewriter.eraseOp(force);
    //    assert(false);
    return success();
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

    // TODO: setup mapping for arguments in mapper
    // This is not safe! Fuck me x(
    // consider f () { stmt; f(); } | g() { f (); }
    // this will expand into
    //   g() { f(); } -> g() { stmt; f(); } -> g { stmt; stmt; f(); } -> ...
    InlinerInterface inliner(rewriter.getContext());
    if (!called.isRecursive()) {
      LogicalResult isInlined = inlineRegion(
          inliner, &called.getBody(), ap, ap.getFnArguments(), ap.getResult());
      assert(succeeded(isInlined) && "unable to inline");
      return success();
    } else {
      return failure();
    }
  }
};


// TODO: we need to know which argument was forced.
HaskFnType mkForcedFnType(HaskFnType fty) {
  SmallVector<Type, 4> forcedTys;
  for(Type ty : fty.getInputTypes()) {
    ThunkType thunkty = ty.dyn_cast<ThunkType>();
    assert(thunkty);
    forcedTys.push_back(thunkty.getElementType());
  }
  return HaskFnType::get(fty.getContext(), forcedTys, fty.getResultType());

}

struct OutlineRecursiveApEagerPattern
    : public mlir::OpRewritePattern<ApEagerOp> {
  OutlineRecursiveApEagerPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ApEagerOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult
  matchAndRewrite(ApEagerOp ap,
                  mlir::PatternRewriter &rewriter) const override {

    HaskFuncOp parentfn = ap.getParentOfType<HaskFuncOp>();
    ModuleOp mod = ap.getParentOfType<ModuleOp>();

    auto ref = ap.getFn().getDefiningOp<HaskRefOp>();
    if (!ref) {
      return failure();
    }

    if (parentfn.getName() != ref.getRef()) {
      return failure();
    }

    HaskFuncOp called = mod.lookupSymbol<HaskFuncOp>(ref.getRef());
    assert(called && "unable to find called function.");

    // TODO: this is an over-approximation of course, we only need
    // a single argument (really, the *same* argument to be reused).
    // I've moved the code here to test that the crash isn't because of a
    // bail-out.

    SmallVector<Value, 4> clonedFnCallArgs;
    for (int i = 0; i < called.getBody().getNumArguments(); ++i) {
      Value arg = called.getBody().getArgument(i);
      if (!arg.hasOneUse()) {
        return failure();
      }
      ForceOp uniqueForceOfArg = dyn_cast<ForceOp>(arg.use_begin().getUser());
      if (!uniqueForceOfArg) { return failure(); }
    }


    std::string clonedFnName = called.getName().str() + "rec_force_outline";


    rewriter.setInsertionPoint(ap);
    HaskRefOp clonedFnRef = rewriter.create<HaskRefOp>(
        ref.getLoc(), clonedFnName, mkForcedFnType(called.getFunctionType()));

    llvm::errs() << "clonedFnRef: |" << clonedFnRef << "|\n";
    rewriter.replaceOpWithNewOp<ApEagerOp>(ap, clonedFnRef,
                                           ap.getFnArguments());

    // TODO: this disastrous house of cards depends on the order of cloning.
    // We first replace the reucrsive call, and *then* clone the function.
    HaskFuncOp clonedfn = parentfn.clone();
    clonedfn.setName(clonedFnName);

    // TODO: consider if going forward is more sensible or going back is
    // more sensible. Right now I am reaching forward, but perhaps
    // it makes sense to reach back.
    for (int i = 0; i < clonedfn.getBody().getNumArguments(); ++i) {
      Value arg = clonedfn.getBody().getArgument(i);
      if (!arg.hasOneUse()) {
        assert(false && "this precondition as already been checked!");
      }
      // This is of course crazy. We should handle the case if we have
      // multiple force()s.

//      llvm::errs() << "\n--- use: " << *arg.use_begin().getUser() << "\n";
      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
      ForceOp uniqueForceOfArg = dyn_cast<ForceOp>(arg.use_begin().getUser());
      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

      if (!uniqueForceOfArg) {
        assert(false && "this precondition has already been checked!");
        return failure();
      }

      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";


      // we are safe to create a new function because we have a unique force
      // of an argument. We can change the type of the function and we can
      // change the argument.

      // replace argument.
      uniqueForceOfArg.replaceAllUsesWith(arg);
      arg.setType(uniqueForceOfArg.getType());
      rewriter.eraseOp(uniqueForceOfArg);
      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    }
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    mod.push_back(clonedfn);
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    llvm::errs() << "=======\n";
    llvm::errs() << mod;
    llvm::errs() << "\n=======\n";
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    return success();

    // LogicalResult isInlined = inlineRegion(
    //     inliner, &called.getBody(), ap, ap.getFnArguments(), ap.getResult());
    // assert(succeeded(isInlined) && "unable to inline");
    // return success();
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
    patterns.insert<ForceOfThunkifyPattern>(&getContext());
    // patterns.insert<OutlineUknownForcePattern>(&getContext());
    patterns.insert<OutlineRecursiveApEagerPattern>(&getContext());
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
