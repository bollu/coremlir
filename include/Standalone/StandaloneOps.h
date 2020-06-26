//===- StandaloneOps.h - Standalone dialect ops -----------------*- C++ -*-===//
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

namespace mlir {
namespace standalone {

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.h.inc"

class LambdaOp : public Op<LambdaOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.lambda"; };
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
  static StringRef getOperationName() { return "standalone.case"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Region &getScrutineeRegion() { this->getOperation()->getRegion(0); }
  int getNumAlts() { return this->getOperation()->getNumRegions() - 1; }
  Region &getAltRHS(int i) { return this->getOperation()->getRegion(i +1); }
  mlir::DictionaryAttr getAltLHSs() { return this->getOperation()->getAttrDictionary(); }
  Attribute getAltLHS(int i) { return getAltLHSs().get("arg" + std::to_string(i)); }
  void print(OpAsmPrinter &p);

};


class ApOp : public Op<ApOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.ap"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);

};

class ReturnOp : public Op<ReturnOp, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "standalone.return"; };
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  Value getValue() { return this->getOperation()->getOperand(0); }
  Value getInput() { return this->getValue(); }

  void print(OpAsmPrinter &p);

};



} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEOPS_H
