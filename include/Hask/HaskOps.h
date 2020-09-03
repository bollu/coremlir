//===- HaskOps.h - Hask dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STANDALONEOPS_H
#define STANDALONE_STANDALONEOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "mlir/Pass/Pass.h"


namespace mlir {
namespace standalone {

#define GET_OP_CLASSES
#include "Hask/HaskOps.h.inc"


class HaskReturnOp : public Op<HaskReturnOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.return"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getValue() { return this->getOperation()->getOperand(0); }
  Value getInput() { return this->getValue(); }
  void print(OpAsmPrinter &p);

};


class MakeI64Op : public Op<MakeI64Op, OpTrait::OneResult, OpTrait::ZeroSuccessor> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_i64"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  IntegerAttr getValue() {
    return this->getOperation()->getAttrOfType<IntegerAttr>("value");
  }
  void print(OpAsmPrinter &p);
};

class MakeStringOp : public Op<MakeStringOp, OpTrait::OneResult, OpTrait::ZeroSuccessor> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_string"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  Attribute getValue() { return this->getOperation()->getAttr("value"); }
  void print(OpAsmPrinter &p);
};



class MakeDataConstructorOp : public Op<MakeDataConstructorOp, OpTrait::ZeroResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_data_constructor"; };
  llvm::StringRef getDataConstructorName(); 
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class ApSSAOp : public Op<ApSSAOp, OpTrait::OneResult, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.apSSA"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  // void print(OpAsmPrinter &p);
  /*
  Value getFn() { return getOperation()->getOperand(0); }
  int getNumFnArguments() { return getOperation()->getNumOperands()-1; }
  Value getFnArgument(int i) { return getOperation()->getOperand(1+i); }
  */
  // return true if the fn is symbolic.
  Value getFn();
  int getNumFnArguments();
  Value getFnArgument(int i);
  SmallVector<Value, 4> getFnArguments();
  void print(OpAsmPrinter &p);
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value fn, SmallVectorImpl<Value> &params);

  // no side effects
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {}
};

class CaseSSAOp : public Op<CaseSSAOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.caseSSA"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  int getNumAlts() { return this->getOperation()->getNumRegions(); }
  Region &getAltRHS(int i) { return this->getOperation()->getRegion(i); }
  mlir::DictionaryAttr getAltLHSs() { return this->getOperation()->getAttrDictionary(); }
  Attribute getAltLHS(int i) { return getAltLHSs().get("alt" + std::to_string(i)); }
  void print(OpAsmPrinter &p);
  llvm::Optional<int> getDefaultAltIndex();

};

class LambdaSSAOp : public Op<LambdaSSAOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.lambdaSSA"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  Region &getBody() { assert(this->getOperation()->getNumRegions() == 1);  return this->getOperation()->getRegion(0);  }
  Block::BlockArgListType inputRange() { this->getBody().begin()->getArguments();   }
  int getNumInputs() { this->getBody().begin()->getNumArguments(); }
  mlir::BlockArgument getInput(int i) { assert(i < getNumInputs()); return this->getBody().begin()->getArgument(i); }

};


class HaskRefOp : public Op<HaskRefOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.ref"; };
  void print(OpAsmPrinter &p);
  llvm::StringRef getRef();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  // build a single region.

  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultTy);
};

class HaskFuncOp : public Op<HaskFuncOp,
                OpTrait::ZeroOperands,
                OpTrait::ZeroResult,
                OpTrait::OneRegion, // OpTrait::IsIsolatedFromAbove,
                // OpTrait::AffineScope,
                // CallableOpInterface::Trait,
                SymbolOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.func"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  void print(OpAsmPrinter &p);
  llvm::StringRef getFuncName();
  LambdaSSAOp getLambda();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
};




// replace case x of name { default -> ... } with name = force(x);
class ForceOp : public Op<ForceOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.force"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  void print(OpAsmPrinter &p);
};


// replace y = case x of name { default -> ...; return val } with
//  name = force(x);
//  ...
//  y = copy(val)
class CopyOp : public Op<CopyOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.copy"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  void print(OpAsmPrinter &p);
};


class HaskADTOp : public Op<HaskADTOp, OpTrait::ZeroResult, OpTrait::ZeroOperands> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.adt"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class HaskGlobalOp : public Op<HaskGlobalOp,
                OpTrait::ZeroOperands,
                OpTrait::ZeroResult,
                OpTrait::OneRegion, // OpTrait::IsIsolatedFromAbove,
                SymbolOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.global"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  void print(OpAsmPrinter &p);
  llvm::StringRef getGlobalName();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
};



// lower hask to standard.
std::unique_ptr<mlir::Pass> createLowerHaskToStandardPass();
// lower hask+standard to LLVM by eliminating all the junk.
std::unique_ptr<mlir::Pass> createLowerHaskStandardToLLVMPass();


} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEOPS_H
