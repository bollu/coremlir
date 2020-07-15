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
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


namespace mlir {
namespace standalone {

#define GET_OP_CLASSES
#include "Hask/HaskOps.h.inc"

class LambdaOp : public Op<LambdaOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.lambda"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  Region &getBody() { assert(this->getOperation()->getNumRegions() == 1);  return this->getOperation()->getRegion(0);  }
  Block::BlockArgListType inputRange() { this->getBody().begin()->getArguments();   }
  int getNumInputs() { this->getBody().begin()->getNumArguments(); }
  mlir::BlockArgument getInput(int i) { assert(i < getNumInputs()); return this->getBody().begin()->getArgument(i); }

  // LogicalResult verify();
};

class CaseOp : public Op<CaseOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.case"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Region &getScrutineeRegion() { this->getOperation()->getRegion(0); }
  int getNumAlts() { return this->getOperation()->getNumRegions() - 1; }
  Region &getAltRHS(int i) { return this->getOperation()->getRegion(i +1); }
  mlir::DictionaryAttr getAltLHSs() { return this->getOperation()->getAttrDictionary(); }
  Attribute getAltLHS(int i) { return getAltLHSs().get("arg" + std::to_string(i)); }
  void print(OpAsmPrinter &p);

};


class ApOp : public Op<ApOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.ap"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Region &getFn() { return getOperation()->getRegion(0); }
  int getNumFnArguments() { return getOperation()->getNumRegions()-1; }
  Region &getFnArgument(int i) { return getOperation()->getRegion(1+i); }
  void print(OpAsmPrinter &p);

};

class ReturnOp : public Op<ReturnOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.return"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getValue() { return this->getOperation()->getOperand(0); }
  Value getInput() { return this->getValue(); }
  void print(OpAsmPrinter &p);

};


class MakeI32Op : public Op<MakeI32Op, OpTrait::OneResult, OpTrait::ZeroSuccessor> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_i32"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  Attribute getValue() { return this->getOperation()->getAttr("value"); }
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



class MakeDataConstructorOp : public Op<MakeDataConstructorOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_data_constructor"; };
    //  getDataConstructorName() { this->getOperation()->getAttr("name") ; };
  Attribute getDataConstructorNameAttr() { this->getOperation()->getAttr("name"); };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class DominanceFreeScopeOp : public Op<DominanceFreeScopeOp, OpTrait::OneRegion, OpTrait::ZeroOperands, RegionKindInterface::Trait, OpTrait::ZeroResult, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.dominance_free_scope"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  void print(OpAsmPrinter &p);
  static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  // build a single region.

  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultTy);

};


class TopLevelBindingOp : public Op<TopLevelBindingOp, OpTrait::OneResult, OpTrait::OneRegion, RegionKindInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.toplevel_binding"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  Region &getBody() { this->getRegion(); };
  static RegionKind getRegionKind(unsigned index) { return RegionKind::SSACFG; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};


class ModuleOp : public Op<ModuleOp, OpTrait::ZeroResult, OpTrait::OneRegion, OpTrait::SymbolTable> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.module"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  Region &getBody() { this->getRegion(); };
  static RegionKind getRegionKind(unsigned index) { return RegionKind::SSACFG; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};


class DummyFinishOp : public Op<DummyFinishOp, OpTrait::ZeroResult, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.dummy_finish"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);

};

class ConstantOp : public Op<ConstantOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.constant"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  Value getConstantValue() { return this->getOperation()->getOperand(0); }
  Type getConstantType() { return this->getConstantValue().getType(); }

};

class ApSSAOp : public Op<ApSSAOp, OpTrait::OneResult> {
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
  Optional<StringAttr> fnSymbolicAttr();
  // Optional<Value> fnValue();
  Value getFn();
  int getNumFnArguments();
  Value getFnArgument(int i);
  void print(OpAsmPrinter &p);
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);
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


class RecursiveRefOp : public Op<RecursiveRefOp, OpTrait::OneRegion, OpTrait::ZeroOperands, RegionKindInterface::Trait, OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.recursive_ref"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  void print(OpAsmPrinter &p);
  static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
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
  static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
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


} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEOPS_H
