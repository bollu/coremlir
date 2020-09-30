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

// #define GET_OP_CLASSES
// #include "Hask/HaskOps.h.inc"

class HaskReturnOp
    : public Op<HaskReturnOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor,
                OpTrait::IsTerminator, OpTrait::OneOperand> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.return"; };
  Value getInput() { return this->getOperation()->getOperand(0); }
  Type getType() { return this->getInput().getType(); }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class MakeI64Op
    : public Op<MakeI64Op, OpTrait::OneResult, OpTrait::ZeroSuccessor,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_i64"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  IntegerAttr getValue() {
    return this->getOperation()->getAttrOfType<IntegerAttr>("value");
  }
  void print(OpAsmPrinter &p);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class MakeStringOp
    : public Op<MakeStringOp, OpTrait::OneResult, OpTrait::ZeroSuccessor,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.make_string"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  Attribute getValue() { return this->getOperation()->getAttr("value"); }
  void print(OpAsmPrinter &p);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class ApOp
    : public Op<ApOp, OpTrait::OneResult, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.ap"; };
  Value getFn() { return this->getOperation()->getOperand(0); };
  int getNumFnArguments() {
    return this->getOperation()->getNumOperands() - 1;
  };
  Value getFnArgument(int i) {
    return this->getOperation()->getOperand(i + 1);
  };

  SmallVector<Value, 4> getFnArguments() {
    SmallVector<Value, 4> args;
    for (int i = 0; i < getNumFnArguments(); ++i) {
      args.push_back(getFnArgument(i));
    }
    return args;
  }
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value fn, SmallVectorImpl<Value> &params);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class ApEagerOp
    : public Op<ApEagerOp, OpTrait::OneResult, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.apEager"; };
  Value getFn() { return this->getOperation()->getOperand(0); };
  int getNumFnArguments() {
    return this->getOperation()->getNumOperands() - 1;
  };
  Value getFnArgument(int i) {
    return this->getOperation()->getOperand(i + 1);
  };

  SmallVector<Value, 4> getFnArguments() {
    SmallVector<Value, 4> args;
    for (int i = 0; i < getNumFnArguments(); ++i) {
      args.push_back(getFnArgument(i));
    }
    return args;
  }
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value fn, SmallVectorImpl<Value> &params);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class CaseOp
    : public Op<CaseOp, OpTrait::OneResult, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.case"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  int getNumAlts() { return this->getOperation()->getNumRegions(); }
  Region &getAltRHS(int i) { return this->getOperation()->getRegion(i); }
  Region &getDefaultRHS() {
    assert(this->getDefaultAltIndex().hasValue());
    return this->getAltRHS(*this->getDefaultAltIndex());
  }
  mlir::DictionaryAttr getAltLHSs() {
    return this->getOperation()->getAttrDictionary();
  }
  FlatSymbolRefAttr getAltLHS(int i) {
    return getAltLHSs()
        .get("alt" + std::to_string(i))
        .cast<FlatSymbolRefAttr>();
  }

  static const char *getCaseTypeKey() { return "constructorName"; }
  void print(OpAsmPrinter &p);
  llvm::Optional<int> getDefaultAltIndex();
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class CaseIntOp
    : public Op<CaseIntOp, OpTrait::OneResult, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.caseint"; };
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  int getNumAlts() { return this->getOperation()->getNumRegions(); }
  Region &getAltRHS(int i) { return this->getOperation()->getRegion(i); }
  Region &getDefaultRHS() {
    assert(this->getDefaultAltIndex().hasValue());
    return this->getAltRHS(*this->getDefaultAltIndex());
  }

  mlir::DictionaryAttr getAltLHSs() {
    return this->getOperation()->getAttrDictionary();
  }
  Optional<IntegerAttr> getAltLHS(int i) {
    Attribute lhs = getAltLHSs().get("alt" + std::to_string(i));
    if (lhs.isa<IntegerAttr>()) {
      return {lhs.cast<IntegerAttr>()};
    }
    return {};
  }
  Attribute getAltLHSRaw(int i) {
    return getAltLHSs().get("alt" + std::to_string(i));
  }

  void print(OpAsmPrinter &p);
  llvm::Optional<int> getDefaultAltIndex();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class DefaultCaseOp : public Op<DefaultCaseOp, OpTrait::OneResult,
                                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.defaultcase"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  static const char *getCaseTypeKey() { return "constructorName"; }
  std::string getConstructorTag() {
    return this->getOperation()
        ->getAttrOfType<FlatSymbolRefAttr>(getCaseTypeKey())
        .getValue()
        .str();
  }
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  void print(OpAsmPrinter &p);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class LambdaOp : public Op<LambdaOp, OpTrait::OneResult, OpTrait::OneRegion> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.lambda"; };
  Region &getBody() {
    assert(this->getOperation()->getNumRegions() == 1);
    return this->getOperation()->getRegion(0);
  }
  Block &getBodyBB() { return this->getBody().front(); }
  Block::BlockArgListType inputRange() {
    this->getBody().begin()->getArguments();
  }
  int getNumInputs() { this->getBody().begin()->getNumArguments(); }
  mlir::BlockArgument getInput(int i) {
    assert(i < getNumInputs());
    return this->getBody().begin()->getArgument(i);
  }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class HaskRefOp
    : public Op<HaskRefOp, OpTrait::OneResult, OpTrait::ZeroOperands,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.ref"; };
  llvm::StringRef getRef() {
    return getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
        .getValue();
  }
  static void build(OpBuilder &odsBuilder, OperationState &odsState,
                    Type resultTy);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class HaskFuncOp : public Op<HaskFuncOp, OpTrait::ZeroOperands,
                             OpTrait::ZeroResult, OpTrait::OneRegion,
                             // OpTrait::AffineScope,
                             // CallableOpInterface::Trait,
                             SymbolOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.func"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  void print(OpAsmPrinter &p);
  llvm::StringRef getFuncName();
  LambdaOp getLambda();
  bool isRecursive();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);
};

// replace case x of name { default -> ... } with name = force(x);
class ForceOp : public Op<ForceOp, OpTrait::OneResult, OpTrait::OneOperand,
                          MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.force"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  void print(OpAsmPrinter &p);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value scrutinee);

  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class HaskADTOp
    : public Op<HaskADTOp, OpTrait::ZeroResult, OpTrait::ZeroOperands> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.adt"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

// do I need this? unclear.
class HaskGlobalOp
    : public Op<HaskGlobalOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                OpTrait::OneRegion, // OpTrait::IsIsolatedFromAbove,
                SymbolOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.global"; };
  Region &getRegion() { return this->getOperation()->getRegion(0); };
  Type getType() {
    Region &r = getRegion();
    HaskReturnOp ret =
        dyn_cast<HaskReturnOp>(r.getBlocks().front().getTerminator());
    assert(ret && "global does not have a return value");
    return ret.getType();
  }
  llvm::StringRef getGlobalName();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

class HaskConstructOp
    : public Op<HaskConstructOp, OpTrait::OneResult, OpTrait::ZeroRegion,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.construct"; };
  static const char *getDataConstructorAttrName() { return "dataconstructor"; }
  static const char *getDataTypeAttrName() { return "datatype"; }
  StringRef getDataConstructorName() {
    return getAttrOfType<FlatSymbolRefAttr>(getDataConstructorAttrName())
        .getValue();
  }

  StringRef getDataTypeName() {
    return getAttrOfType<FlatSymbolRefAttr>(getDataTypeAttrName()).getValue();
  }

  int getNumOperands() { this->getOperation()->getNumOperands(); }
  Value getOperand(int i) { return this->getOperation()->getOperand(i); }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class HaskPrimopAddOp
    : public Op<HaskPrimopAddOp, OpTrait::OneResult,
                OpTrait::NOperands<2>::Impl, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.primop_add"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class HaskPrimopSubOp
    : public Op<HaskPrimopSubOp, OpTrait::OneResult,
                OpTrait::NOperands<2>::Impl, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.primop_sub"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class ThunkifyOp
    : public Op<ThunkifyOp, OpTrait::OneResult, OpTrait::OneOperand,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.thunkify"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getScrutinee() { this->getOperation()->getOperand(0); }
  void print(OpAsmPrinter &p);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value scrutinee);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

class TransmuteOp
    : public Op<TransmuteOp, OpTrait::OneResult, OpTrait::OneOperand,
                MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.transmute"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value scrutinee);
  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects) {}
};

// lower hask to standard.
std::unique_ptr<mlir::Pass> createLowerHaskToStandardPass();
// lower hask+standard to LLVM by eliminating all the junk.
std::unique_ptr<mlir::Pass> createLowerHaskStandardToLLVMPass();
} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEOPS_H
