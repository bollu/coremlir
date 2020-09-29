//===- HaskOps.cpp - Hask dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hask/HaskOps.h"
#include "Hask/HaskDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LogicalResult.h"
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
#define GET_OP_CLASSES
#include "Hask/HaskOps.cpp.inc"

// === RETURN OP ===
// === RETURN OP ===
// === RETURN OP ===
// === RETURN OP ===
// === RETURN OP ===

ParseResult HaskReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::OpAsmParser::OperandType in;
  mlir::Type type;
  if (parser.parseLParen() || parser.parseOperand(in) || parser.parseRParen() ||
      parser.parseColon() || parser.parseType(type)) {
    return failure();
  }

  parser.resolveOperand(in, type, result.operands);
  return success();
};

void HaskReturnOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(" << getInput() << ")"
    << " : " << getInput().getType();
};

// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===

ParseResult MakeI64Op::parse(OpAsmParser &parser, OperationState &result) {
  // mlir::OpAsmParser::OperandType i;
  Attribute attr;
  if (parser.parseLParen() ||
      parser.parseAttribute(attr, "value", result.attributes) ||
      parser.parseRParen())
    return failure();
  // result.addAttribute("value", attr);
  // SmallVector<Value, 1> vi;
  // parser.resolveOperand(i, parser.getBuilder().getIntegerType(32), vi);

  // TODO: convert this to emitParserError, etc.
  // assert (attr.getType().isSignedInteger() && "expected parameter to make_i32
  // to be integer");

  result.addTypes(parser.getBuilder().getType<ValueType>());
  return success();
};

void MakeI64Op::print(OpAsmPrinter &p) {
  p << getOperationName() << "(" << getValue() << ")";
};

// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===

/*
ParseResult DeclareDataConstructorOp::parse(OpAsmParser &parser, OperationState
&result) {
    // parser.parseAttribute(, parser.getBuilder().getStringAttr )
    // if(parser.parseLess()) return failure();

    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr,
::mlir::SymbolTable::getSymbolAttrName(), result.attributes)) { return
failure();
    }
    // if(parser.parseAttribute(attr, "name", result.attributes)) {
    //     assert(false && "unable to parse attribute!");  return failure();
    // }
    // if(parser.parseGreater()) return failure();
    // result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

llvm::StringRef DeclareDataConstructorOp::getDataConstructorName() {
    return
getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName()).getValue();
}

void DeclareDataConstructorOp::print(OpAsmPrinter &p) {
    p << getOperationName() << " ";
    p.printSymbolName(getDataConstructorName());
};

*/

// === APOP OP ===
// === APOP OP ===
// === APOP OP ===
// === APOP OP ===
// === APOP OP ===

ParseResult ApOp::parse(OpAsmParser &parser, OperationState &result) {
  // OpAsmParser::OperandType operand_fn;
  OpAsmParser::OperandType op_fn;
  // (<fn-arg>
  if (parser.parseLParen()) {
    return failure();
  }
  if (parser.parseOperand(op_fn)) {
    return failure();
  }

  // : type
  HaskType ratorty;
  if (parser.parseColonType<HaskType>(ratorty)) {
    return failure();
  }
  if (parser.resolveOperand(op_fn, ratorty, result.operands)) {
    return failure();
  }

  if (HaskFnType fnty = ratorty.dyn_cast<HaskFnType>()) {
    std::vector<Type> paramtys = fnty.getInputTypes();
    Type retty = fnty.getResultType();

    for (int i = 0; i < paramtys.size(); ++i) {
      if (parser.parseComma()) {
        return failure();
      }
      OpAsmParser::OperandType op;
      if (parser.parseOperand(op))
        return failure();
      if (parser.resolveOperand(op, paramtys[i], result.operands)) {
        return failure();
      }
    }

    //)
    if (parser.parseRParen())
      return failure();
    result.addTypes(
        parser.getBuilder().getType<ThunkType>(fnty.getResultType()));
  } else {
    InFlightDiagnostic err =
        parser.emitError(parser.getCurrentLocation(),
                         "expected function type, got non function type: [");
    err << ratorty << "]";
    return failure();
  }

  return success();
};

void ApOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(";
  p << this->getFn() << " :" << this->getFn().getType();

  for (int i = 0; i < this->getNumFnArguments(); ++i) {
    p << ", " << this->getFnArgument(i);
  }
  p << ")";
};

void ApOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 Value fn, SmallVectorImpl<Value> &params) {

  // hack! we need to construct the type properly.
  state.addOperands(fn);
  assert(fn.getType().isa<HaskFnType>());
  HaskFnType fnty = fn.getType().cast<HaskFnType>();

  std::vector<Type> paramtys;
  Type retty;

  assert(params.size() == paramtys.size());

  for (int i = 0; i < params.size(); ++i) {
    assert(paramtys[i] == params[i].getType());
  }

  state.addOperands(params);
  state.addTypes(builder.getType<ThunkType>(fnty.getResultType()));
};

// === APEAGEROP OP ===
// === APEAGEROP OP ===
// === APEAGEROP OP ===
// === APEAGEROP OP ===
// === APEAGEROP OP ===

ParseResult ApEagerOp::parse(OpAsmParser &parser, OperationState &result) {
  // OpAsmParser::OperandType operand_fn;
  OpAsmParser::OperandType op_fn;
  // (<fn-arg>
  if (parser.parseLParen()) {
    return failure();
  }
  if (parser.parseOperand(op_fn)) {
    return failure();
  }

  // : type
  HaskType ratorty;
  if (parser.parseColonType<HaskType>(ratorty)) {
    return failure();
  }
  if (parser.resolveOperand(op_fn, ratorty, result.operands)) {
    return failure();
  }

  if (HaskFnType fnty = ratorty.dyn_cast<HaskFnType>()) {
    std::vector<Type> paramtys = fnty.getInputTypes();
    Type retty = fnty.getResultType();

    for (int i = 0; i < paramtys.size(); ++i) {
      if (parser.parseComma()) {
        return failure();
      }
      OpAsmParser::OperandType op;
      if (parser.parseOperand(op))
        return failure();
      if (parser.resolveOperand(op, paramtys[i], result.operands)) {
        return failure();
      }
    }

    //)
    if (parser.parseRParen())
      return failure();
    result.addTypes(
        parser.getBuilder().getType<ThunkType>(fnty.getResultType()));
  } else {
    InFlightDiagnostic err =
        parser.emitError(parser.getCurrentLocation(),
                         "expected function type, got non function type: [");
    err << ratorty << "]";
    return failure();
  }

  return success();
};

void ApEagerOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(";
  p << this->getFn() << " :" << this->getFn().getType();

  for (int i = 0; i < this->getNumFnArguments(); ++i) {
    p << ", " << this->getFnArgument(i);
  }
  p << ")";
};

void ApEagerOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 Value fn, SmallVectorImpl<Value> &params) {

  // hack! we need to construct the type properly.
  state.addOperands(fn);
  assert(fn.getType().isa<HaskFnType>());
  HaskFnType fnty = fn.getType().cast<HaskFnType>();


  assert(params.size() == fnty.getInputTypes().size());

  for (int i = 0; i < params.size(); ++i) {
    assert(params[i].getType() == fnty.getInputType(i));
  }

  state.addOperands(params);
  state.addTypes(builder.getType<ThunkType>(fnty.getResultType()));
};

// === CASESSA OP ===
// === CASESSA OP ===
// === CASESSA OP ===
// === CASESSA OP ===
// === CASESSA OP ===
ParseResult CaseOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;

  FlatSymbolRefAttr constructorName;
  if (parser.parseAttribute<FlatSymbolRefAttr>(
          constructorName, CaseOp::getCaseTypeKey(), result.attributes)) {
    return failure();
  };

  if (parser.parseOperand(scrutinee))
    return failure();
  if (parser.resolveOperand(
          scrutinee, parser.getBuilder().getType<ADTType>(constructorName),
          result.operands)) {
    return failure();
  }

  llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
  // if(parser.parseOptionalAttrDict(result.attributes)) return failure();
  llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

  // "[" altname "->" region "]"
  int nattr = 0;
  SmallVector<Region *, 4> altRegions;
  while (succeeded(parser.parseOptionalLSquare())) {
    FlatSymbolRefAttr alt_type_attr;
    const std::string attrname = "alt" + std::to_string(nattr);
    parser.parseAttribute<FlatSymbolRefAttr>(alt_type_attr, attrname,
                                             result.attributes);
    nattr++;
    parser.parseArrow();
    Region *r = result.addRegion();
    altRegions.push_back(r);
    if (parser.parseRegion(*r, {}, {}))
      return failure();
    parser.parseRSquare();
  }

  assert(altRegions.size() > 0);

  HaskReturnOp retFirst =
      cast<HaskReturnOp>(altRegions[0]->getBlocks().front().getTerminator());
  for (int i = 1; i < altRegions.size(); ++i) {
    HaskReturnOp ret =
        cast<HaskReturnOp>(altRegions[i]->getBlocks().front().getTerminator());
    assert(retFirst.getType() == ret.getType() &&
           "all case branches must return  same levity [value/thunk]");
  }

  result.addTypes(retFirst.getType());
  return success();
};

void CaseOp::print(OpAsmPrinter &p) {
  p << getOperationName() << " ";
  // p << "[ " << this->getOperation()->getNumOperands() << " | " <<
  // this->getNumAlts() << "] "; p << this->getOperation()->getOperand(0);
  p << this->getOperation()->getAttrOfType<FlatSymbolRefAttr>(
      this->getCaseTypeKey());
  p << " " << this->getScrutinee();
  // p.printOptionalAttrDict(this->getAltLHSs().getValue());
  for (int i = 0; i < this->getNumAlts(); ++i) {
    p << " [" << this->getAltLHS(i) << " -> ";
    p.printRegion(this->getAltRHS(i));
    p << "]\n";
  }
};

llvm::Optional<int> CaseOp::getDefaultAltIndex() {
  for (int i = 0; i < getNumAlts(); ++i) {
    Attribute ai = this->getAltLHS(i);
    StringAttr sai = ai.dyn_cast<StringAttr>();
    if (sai && sai.getValue() == "default") {
      return i;
    }
  }
  return llvm::Optional<int>();
}

// === DEFAULTCASE OP ===
// === DEFAULTCASE OP ===
// === DEFAULTCASE OP ===
// === DEFAULTCASE OP ===
// === DEFAULTCASE OP ===
// === DEFAULTCASE OP ===

ParseResult DefaultCaseOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;

  if (parser.parseLParen()) { return failure(); }
  FlatSymbolRefAttr constructorName;
  if (parser.parseAttribute<FlatSymbolRefAttr>(
          constructorName, CaseOp::getCaseTypeKey(), result.attributes)) {
    return failure();
  }

  parser.parseComma();
  if (parser.parseOperand(scrutinee) || parser.parseRParen()) {
    return failure();
  }
  Type retty;
  if (parser.parseColonType(retty))  { return failure(); }
  parser.resolveOperand(scrutinee, 
          ADTType::get(parser.getBuilder().getContext(), constructorName),
          result.operands);


  result.addTypes(retty);
  return success();
};

void DefaultCaseOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(";
  p << this->getOperation()->getAttrOfType<FlatSymbolRefAttr>(
          this->getCaseTypeKey());
  p << ", " << this->getScrutinee()  << ")" << " : " << this->getResult().getType();
};

// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===

ParseResult LambdaOp::parse(OpAsmParser &parser, OperationState &result) {

  if (parser.parseLParen()) {
    return failure();
  }

  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTys;
  if (succeeded(parser.parseOptionalRParen())) {
    // we have no params.
  } else {
    while (1) {
      OpAsmParser::OperandType arg;
      if (parser.parseRegionArgument(arg)) {
        return failure();
      };
      args.push_back(arg);

      if (parser.parseColon()) {
        return failure();
      }
      Type argType;
      if (parser.parseType(argType)) {
        return failure();
      }
      argTys.push_back(argType);

      if (!(argType.isa<ThunkType>() || argType.isa<ValueType>() ||
            argType.isa<HaskFnType>())) {
        return parser.emitError(
            arg.location,
            "argument must either ValueType, ThunkType, or HaskFnType");
      }

      if (succeeded(parser.parseOptionalRParen())) {
        break;
      } else if (parser.parseComma()) {
        return failure();
      }
    }
  }

  Region *r = result.addRegion();
  if (parser.parseRegion(*r, {args}, {argTys}))
    return failure();

  HaskReturnOp ret = cast<HaskReturnOp>(r->getBlocks().front().getTerminator());
  Value retval = ret.getInput();
  Type rettyy = retval.getType();

  result.addTypes(parser.getBuilder().getType<HaskFnType>(argTys, rettyy));
  return success();
}

void LambdaOp::print(OpAsmPrinter &p) {
  p << "hask.lambda";
  p << "(";
  for (int i = 0; i < this->getNumInputs(); ++i) {
    p << this->getInput(i);
    p << ":" << this->getInput(i).getType();
    if (i < this->getNumInputs() - 1) {
      p << ",";
    }
  }
  p << ")";
  p.printRegion(this->getBody(), /*printEntryBlockArgs=*/false);
  // p.printRegion(this->getBody(), /*printEntryBlockArgs=*/true);
}

// === RECURSIVEREF OP ===
// === RECURSIVEREF OP ===
// === RECURSIVEREF OP ===
// === RECURSIVEREF OP ===
// === RECURSIVEREF OP ===

ParseResult HaskRefOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  // ( <arg> ) : <type>
  Type ty;
  if (parser.parseLParen() ||
      parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseRParen() || parser.parseColonType(ty)) {
    return failure();
  }

  // TODO: extract this out as a separate function or something.
  if (!ty.isa<HaskType>()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected a haskell type");
  }

  // if (!(ty.isa<ValueType>() || ty.isa<ThunkType>() || ty.isa<HaskFnType>() ||
  // ty.isa<UntypedType>())) {
  //     return parser.emitError(parser.getCurrentLocation(), "expected value,
  //     thunk, function, or untyped type");
  // }

  result.addTypes(ty);
  return success();
}

void HaskRefOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(";
  p.printSymbolName(this->getRef());
  p << ")"
    << " : " << this->getResult().getType();
};

LogicalResult HaskRefOp::verify() {
  ModuleOp mod = this->getOperation()->getParentOfType<mlir::ModuleOp>();
  HaskFuncOp fn = mod.lookupSymbol<HaskFuncOp>(this->getRef());
  HaskGlobalOp global = mod.lookupSymbol<HaskGlobalOp>(this->getRef());
  if (fn) {
    LambdaOp lam = fn.getLambda();
    // how do I attach an error message?
    if (lam.getType() == this->getResult().getType()) {
      return success();
    }
    llvm::errs() << "ERROR at HaskRefOp type verification:"
                 << "\n-mismatch of types at ref."
                 << "\n-Found from function"
                 << " " << fn.getLoc() << " "
                 << "name:" << this->getRef() << " [" << lam.getType()
                 << "]\n"
                    "-Declared at ref as ["
                 << this->getLoc() << " " << *this << "]\n";
    return failure();
  } else if (global) {
    if (global.getType() == this->getResult().getType()) {
      return success();
    }
    llvm::errs() << "ERROR at HaskRefOp type verification:"
                 << "\n-mismatch of types at ref."
                 << "\n-Found from global"
                 << " " << global.getLoc() << " "
                 << "name:" << this->getRef() << " [" << global.getType()
                 << "]\n"
                    "-Declared at ref as ["
                 << this->getLoc() << " " << *this << "]\n";
    return failure();
  } else {
    llvm::errs() << "ERROR at HaskRefOpVerification:"
                 << "\n-unable to find referenced function/global |"
                 << this->getRef() << "|\n";
    // TODO: forward declare stuff like +#
    return failure();
  }
}

// === MakeString OP ===
// === MakeString OP ===
// === MakeString OP ===
// === MakeString OP ===
// === MakeString OP ===

ParseResult MakeStringOp::parse(OpAsmParser &parser, OperationState &result) {
  // mlir::OpAsmParser::OperandType i;
  Attribute attr;

  if (parser.parseLParen() ||
      parser.parseAttribute(attr, "value", result.attributes) ||
      parser.parseRParen())
    return failure();
  // result.addAttribute("value", attr);
  // SmallVector<Value, 1> vi;
  // parser.resolveOperand(i, parser.getBuilder().getIntegerType(32), vi);

  // TODO: check if attr is string.

  result.addTypes(parser.getBuilder().getType<ValueType>());
  return success();
};

void MakeStringOp::print(OpAsmPrinter &p) {
  p << "hask.make_string(" << getValue() << ")";
};

// === HASKFUNC OP ===
// === HASKFUNC OP ===
// === HASKFUNC OP ===
// === HASKFUNC OP ===
// === HASKFUNC OP ===

ParseResult HaskFuncOp::parse(OpAsmParser &parser, OperationState &result) {

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  };

  auto *body = result.addRegion();
  return parser.parseRegion(*body, {}, ArrayRef<Type>());
};

void HaskFuncOp::print(OpAsmPrinter &p) {
  p << "hask.func" << ' ';
  p.printSymbolName(getFuncName());
  // Print the body if this is not an external function.
  Region &body = this->getRegion();
  assert(!body.empty());
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

llvm::StringRef HaskFuncOp::getFuncName() {
  return getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
      .getValue();
}

LambdaOp HaskFuncOp::getLambda() {
  assert(this->getOperation()->getNumRegions() == 1 &&
         "func needs exactly one region");
  Region &region = this->getRegion();
  // TODO: put this in a `verify` block.
  assert(region.getBlocks().size() == 1 && "func has more than one BB");
  Block &entry = region.front();
  HaskReturnOp ret = cast<HaskReturnOp>(entry.getTerminator());
  Value retval = ret.getInput();
  return cast<LambdaOp>(retval.getDefiningOp());
}

bool HaskFuncOp::isRecursive() {
  // https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
  llvm::errs() << "Checking: isFunctionRecursive? " << this->getFuncName() << "\n";
  bool isrec = false;
  this->walk([&](HaskRefOp ref) {
      if (ref.getRef() == this->getFuncName()) {
        isrec = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
  });
  return isrec;
}


// === FORCE OP ===
// === FORCE OP ===
// === FORCE OP ===
// === FORCE OP ===
// === FORCE OP ===

ParseResult ForceOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;
  mlir::Type retty;

  if (parser.parseLParen() || parser.parseOperand(scrutinee) ||
      parser.parseRParen() || parser.parseColon() || parser.parseType(retty)) {
    return failure();
  }

  SmallVector<Value, 4> results;
  if (parser.resolveOperand(
          scrutinee, parser.getBuilder().getType<ThunkType>(retty), results))
    return failure();
  result.addOperands(results);
  result.addTypes(retty);
  return success();
};

void ForceOp::print(OpAsmPrinter &p) {
  p << "hask.force(" << this->getScrutinee() << ")"
    << ":" << this->getResult().getType();
};

void ForceOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value scrutinee) {

  assert(scrutinee.getType().isa<ThunkType>());
  ThunkType t = scrutinee.getType().cast<ThunkType>();
  state.addOperands(scrutinee);
  state.addTypes(t.getElementType());
}






// === ADT OP ===
// === ADT OP ===
// === ADT OP ===
// === ADT OP ===
// === ADT OP ===
//

ParseResult HaskADTOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;

  SmallVector<Value, 4> results;
  Attribute name;
  Attribute constructors;
  if (parser.parseAttribute(name, "name", result.attributes)) {
    return failure();
  }
  if (parser.parseAttribute(name, "constructors", result.attributes)) {
    return failure();
  }

  //    result.addAttribute("name", name);
  //    result.addAttribute("constructors", constructors);
  //    if(parser.parseAttribute(constructors)) { return failure(); }

  llvm::errs() << "ADT: " << name << "\n"
               << "cons: " << constructors << "\n";
  return success();
};

void HaskADTOp::print(OpAsmPrinter &p) {
  p << getOperationName();
  p << " " << this->getAttr("name") << " " << this->getAttr("constructors");
};

// === GLOBAL OP ===
// === GLOBAL OP ===
// === GLOBAL OP ===
// === GLOBAL OP ===
// === GLOBAL OP ===

ParseResult HaskGlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  };

  auto *body = result.addRegion();
  return parser.parseRegion(*body, {}, ArrayRef<Type>());
};

void HaskGlobalOp::print(OpAsmPrinter &p) {
  p << getOperationName() << ' ';
  p.printSymbolName(getName());
  Region &body = this->getRegion();
  assert(!body.empty());
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
};

// === CONSTRUCT OP ===
// === CONSTRUCT OP ===
// === CONSTRUCT OP ===
// === CONSTRUCT OP ===
// === CONSTRUCT OP ===

// do I even need this? I'm not sure. Don't think so?
ParseResult HaskConstructOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  // (<constructor name>, arg1, ..., argn) : hask.adt<type>
  if (parser.parseLParen()) {
    return failure();
  }

  // get constructor name.
  FlatSymbolRefAttr constructor;
  if (parser.parseAttribute<FlatSymbolRefAttr>(
          constructor, getDataConstructorAttrName(), result.attributes)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalRParen())) {
    // empty
  } else {
    if (parser.parseComma()) {
      return failure();
    }
    while (1) {
      OpAsmParser::OperandType param;
      if (parser.parseOperand(param)) {
        return failure();
      }
      if (parser.parseColon()) {
        return failure();
      }
      Type t;
      if (parser.parseType(t)) {
        return failure();
      }
      if (parser.resolveOperand(param, t, result.operands)) {
        return failure();
      }

      // either we need a close right paren, or a comma.
      if (succeeded(parser.parseOptionalRParen())) {
        break;
      }
      if (parser.parseComma()) {
        return failure();
      }
    }
  }
  // : <type>
  Type ty;
  if (parser.parseColon() || parser.parseType(ty)) {
    return failure();
  }
  if (!ty.isa<ADTType>()) {
    InFlightDiagnostic err = parser.emitError(parser.getCurrentLocation(),
                                              "expected ADT type, found: [");
    err << ty << "]";
    return failure();
  }

  result.addTypes(ty);
  return success();
}

void HaskConstructOp::print(OpAsmPrinter &p) {
  p << this->getOperationName();
  p << "(";
  p.printSymbolName(this->getDataConstructorName());
  if (this->getNumOperands() > 0) {
    p << ", ";
  }

  for (int i = 0; i < this->getNumOperands(); ++i) {
    p << this->getOperand(i) << " : " << this->getOperand(i).getType();
    if (i + 1 < this->getNumOperands()) {
      p << ", ";
    }
  }
  p << ") : " << this->getOperation()->getResult(0).getType();
}

// === PRIMOP ADD OP ===
// === PRIMOP ADD OP ===
// === PRIMOP ADD OP ===
// === PRIMOP ADD OP ===
// === PRIMOP ADD OP ===

ParseResult HaskPrimopAddOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::OperandType lhs, rhs;
  if (parser.parseLParen() || parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseRParen()) {
    return failure();
  }
  parser.resolveOperand(lhs, parser.getBuilder().getType<ValueType>(),
                        result.operands);
  parser.resolveOperand(rhs, parser.getBuilder().getType<ValueType>(),
                        result.operands);
  result.addTypes(parser.getBuilder().getType<ValueType>());
  return success();
};

void HaskPrimopAddOp::print(OpAsmPrinter &p) {
  p << this->getOperation()->getName() << "(" << this->getOperand(0) << ","
    << this->getOperand(1) << ")";
};

// === PRIMOP SUB OP ===
// === PRIMOP SUB OP ===
// === PRIMOP SUB OP ===
// === PRIMOP SUB OP ===
// === PRIMOP SUB OP ===

ParseResult HaskPrimopSubOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::OperandType lhs, rhs;
  if (parser.parseLParen() || parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseRParen()) {
    return failure();
  }
  parser.resolveOperand(lhs, parser.getBuilder().getType<ValueType>(),
                        result.operands);
  parser.resolveOperand(rhs, parser.getBuilder().getType<ValueType>(),
                        result.operands);
  result.addTypes(parser.getBuilder().getType<ValueType>());
  return success();
};

void HaskPrimopSubOp::print(OpAsmPrinter &p) {
  p << this->getOperation()->getName() << "(" << this->getOperand(0) << ","
    << this->getOperand(1) << ")";
};

// === CASE INT OP ===
// === CASE INT OP ===
// === CASE INT OP ===
// === CASE INT OP ===
// === CASE INT OP ===
ParseResult CaseIntOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;
  if (parser.parseOperand(scrutinee))
    return failure();
  if (parser.resolveOperand(scrutinee, parser.getBuilder().getType<ValueType>(),
                            result.operands)) {
    return failure();
  }

  llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
  // if(parser.parseOptionalAttrDict(result.attributes)) return failure();
  llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

  // "[" altname "->" region "]"
  int nattr = 0;
  SmallVector<Region *, 4> altRegions;
  while (succeeded(parser.parseOptionalLSquare())) {
    Attribute alt_type_attr;
    const std::string attrname = "alt" + std::to_string(nattr);
    parser.parseAttribute(alt_type_attr, attrname, result.attributes);
    assert(alt_type_attr.isa<IntegerAttr>() ||
           alt_type_attr.isa<FlatSymbolRefAttr>());
    nattr++;
    parser.parseArrow();
    Region *r = result.addRegion();
    altRegions.push_back(r);
    if (parser.parseRegion(*r, {}, {}))
      return failure();
    parser.parseRSquare();
  }

  assert(altRegions.size() > 0);

  HaskReturnOp retFirst =
      cast<HaskReturnOp>(altRegions[0]->getBlocks().front().getTerminator());
  for (int i = 1; i < altRegions.size(); ++i) {
    HaskReturnOp ret =
        cast<HaskReturnOp>(altRegions[i]->getBlocks().front().getTerminator());
    assert(retFirst.getType() == ret.getType() &&
           "all case branches must return  same levity [value/thunk]");
  }

  result.addTypes(retFirst.getType());
  return success();
};

void CaseIntOp::print(OpAsmPrinter &p) {
  p << getOperationName() << " ";
  p << this->getScrutinee();
  for (int i = 0; i < this->getNumAlts(); ++i) {
    p << " [" << this->getAltLHSRaw(i) << " -> ";
    p.printRegion(this->getAltRHS(i));
    p << "]\n";
  }
};

llvm::Optional<int> CaseIntOp::getDefaultAltIndex() {
  for (int i = 0; i < getNumAlts(); ++i) {
    Attribute ai = this->getAltLHSRaw(i);
    FlatSymbolRefAttr sai = ai.dyn_cast<FlatSymbolRefAttr>();
    if (sai && sai.getValue() == "default") {
      return i;
    }
  }
  return llvm::Optional<int>();
}

// === THUNKIFY OP ===
// === THUNKIFY OP ===
// === THUNKIFY OP ===
// === THUNKIFY OP ===
// === THUNKIFY OP ===

ParseResult ThunkifyOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;
  mlir::Type type, retty;

  if (parser.parseLParen() || parser.parseOperand(scrutinee) ||
      parser.parseColon() || parser.parseType(type) || parser.parseRParen() ||
      parser.parseColon() || parser.parseType(retty)) {
    return failure();
  }

  SmallVector<Value, 4> results;
  if (parser.resolveOperand(scrutinee, type, results))
    return failure();
  result.addOperands(results);
  result.addTypes(retty);
  return success();
};

void ThunkifyOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(" << this->getScrutinee() << " :"
    << this->getScrutinee().getType() << ")"
    << ":" << this->getResult().getType();
};

void ThunkifyOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       Value scrutinee) {
  state.addOperands(scrutinee);
  state.addTypes(builder.getType<ThunkType>(scrutinee.getType()));
}

// === TRANSMUTE OP ===
// === TRANSMUTE OP ===
// === TRANSMUTE OP ===
// === TRANSMUTE OP ===
// === TRANSMUTE OP ===

ParseResult TransmuteOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType scrutinee;
  mlir::Type type, retty;

  if (parser.parseLParen() || parser.parseOperand(scrutinee) ||
      parser.parseColon() || parser.parseType(type) || parser.parseRParen() ||
      parser.parseColon() || parser.parseType(retty)) {
    return failure();
  }

  SmallVector<Value, 4> results;
  if (parser.resolveOperand(scrutinee, type, results))
    return failure();
  result.addOperands(results);
  result.addTypes(retty);
  return success();
};

void TransmuteOp::print(OpAsmPrinter &p) {
  p << getOperationName() << "(" << this->getOperand() << " :"
    << this->getOperand().getType() << ")"
    << ":" << this->getResult().getType();
};

// ==REWRITES==
// ==REWRITES==
// ==REWRITES==
// ==REWRITES==
// ==REWRITES==
// ==REWRITES==

// =SATURATE AP=
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/toyc.cpp
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/mlir/Dialect.cpp

// ===== FORCE OP REWRITES =====

// clone the basic block toBeCloned into `beforeInDest`, before location `before`, using `args
// for the arguments.
// https://github.com/llvm/llvm-project/blob/f91f28c350df6815d37c521e8f3dc0641a3ca467/mlir/lib/IR/Region.cpp#L79
Block *cloneBlock(Block &toBeCloned) {
  BlockAndValueMapping mapper;
  Block *newBlock = new Block();
  mapper.map(&toBeCloned, newBlock);
  for (BlockArgument &arg : toBeCloned.getArguments()) {
      mapper.map(arg, newBlock->addArgument(arg.getType()));
  }

  // Clone and remap the operations within this block.
    for (auto &op : toBeCloned) {
    newBlock->push_back(op.clone(mapper));
  }

  auto remapOperands = [&](Operation &op) {
    for (auto &operand : op.getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto &succOp : op.getBlockOperands())
      if (auto *mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };

  for (Operation &op : *newBlock) {
    remapOperands(op);
  }

  return newBlock;
};

//https://github.com/llvm/llvm-project/blob/1372e23c7d4b25fd23689842246e66f70c949b46/mlir/lib/IR/PatternMatch.cpp#L136
Block *cloneBlockBefore(mlir::PatternRewriter &rewriter, Operation *beforeAtDest, Block &src,
                        ValueRange argValues = llvm::None) {
  assert(beforeAtDest);
  Block *newbb = cloneBlock(src);
  rewriter.mergeBlockBefore(&src, beforeAtDest, argValues);
  return newbb;
}


struct ForceOfKnownApCanonicalizationPattern : public mlir::OpRewritePattern<ForceOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ForceOfKnownApCanonicalizationPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) { }


  mlir::LogicalResult
  matchAndRewrite(ForceOp force, mlir::PatternRewriter &rewriter) const override {
      rewriter.create<LLVM::AddOp>(force.getLoc());

      ModuleOp mod = force.getParentOfType<ModuleOp>();
      HaskFuncOp fn = force.getParentOfType<HaskFuncOp>();
      if (!ap) { return failure(); }
      HaskRefOp ref = ap.getFn().getDefiningOp<HaskRefOp>();
      if (!ref) { return failure(); }

      llvm::errs() << "\nref: " << ref  << "\n"
          << "\nap: " << ap << "\n"
          << "\nforce: " << force <<" \n";


      HaskFuncOp forcedFn = mod.lookupSymbol<HaskFuncOp>(ref.getRef());

      // cannot inline a recursive function. Can replace with
      // an apEager
      if (forcedFn.isRecursive()) { 
          rewriter.setInsertionPoint(ap);
          ApEagerOp eager = rewriter.create<ApEagerOp>(ap.getLoc(), ref, ap.getFnArguments());
          rewriter.replaceOp(force, eager.getResult());
          return success();
      }

      Block &forcedFnBB = forcedFn.getLambda().getBodyBB();


      llvm::errs() << "\nforced fn body:\n-------\n";
      forcedFnBB.dump();

      llvm::errs() << "\nforce parent(original):\n----\n";
      force.getParentOfType<HaskFuncOp>().dump();

      Block *clonedBB = cloneBlock(forcedFnBB);
      clonedBB->insertBefore(force.getOperation()->getBlock());
      llvm::errs() << "\nforced called fn(cloned BB):\n----\n";
      clonedBB->dump();
      HaskReturnOp ret = dyn_cast<HaskReturnOp>(clonedBB->getTerminator());

      llvm::errs() << "\nforce parent(inlined):\n-----------\n";
      force.getParentOfType<HaskFuncOp>().dump();
      llvm::errs() << "\n";

      llvm::errs() << "\nforce parent(inlined+merged):\n-----------\n";
      rewriter.mergeBlockBefore(clonedBB, force.getOperation(), ap.getFnArguments());
      force.getParentOfType<HaskFuncOp>().dump();
      llvm::errs() << "\n";



      llvm::errs() << "\nreturnop:\n------\n" << ret << "\n";

      llvm::errs() << "\nforce parent(inlined+merged+force-replaced):\n-----------\n";
      rewriter.replaceOp(force, ret.getOperand());
      rewriter.eraseOp(ret);
      fn.dump();
      llvm::errs() << "\n";
      return success();

  }

};

struct ForceOfThunkifyCanonicalizationPattern : public mlir::OpRewritePattern<ForceOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ForceOfThunkifyCanonicalizationPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ForceOp>(context, /*benefit=*/1) { }


  mlir::LogicalResult
  matchAndRewrite(ForceOp force, mlir::PatternRewriter &rewriter) const override {
      HaskFuncOp fn = force.getParentOfType<HaskFuncOp>();
      ThunkifyOp thunkify = force.getOperand().getDefiningOp<ThunkifyOp>();
      if (!thunkify) { return failure(); }
      rewriter.replaceOp(force, thunkify.getOperand());
      return success();
  }

};



void ForceOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *context) {
  results.insert<ForceOfThunkifyCanonicalizationPattern>(context);
  results.insert<ForceOfKnownApCanonicalizationPattern>(context);
}


// worker/wrapper
void HaskFuncOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                                MLIRContext *context) {

}



// === LOWERING ===
// === LOWERING ===
// === LOWERING ===
// === LOWERING ===
// === LOWERING ===
// === LOWERING ===
// === LOWERING ===

// TODO: fix this to use the TypeConverter machinery, not this hacky
// stuff I wrote.
Value transmuteToVoidPtr(Value v, ConversionPatternRewriter &rewriter,
                         Location loc) {
  llvm::errs() << "v: " << v << " |ty: " << v.getType() << "\n";
  if (v.getType().isa<LLVM::LLVMType>()) {
    LLVM::LLVMType vty = v.getType().cast<LLVM::LLVMType>();
    if (vty.isPointerTy()) {
      return rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMType::getInt8PtrTy(rewriter.getContext()),
          ValueRange(v));
    } else if (vty.isIntegerTy()) {

      return rewriter.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMType::getInt8PtrTy(rewriter.getContext()), v);
    } else {
      assert(false && "unable to transmute into void pointer");
    }
  } else {
    assert(v.getType().isa<HaskFnType>() || v.getType().isa<ValueType>() ||
           v.getType().isa<ThunkType>());
    if (v.getType().isa<HaskFnType>()) {
      return rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMType::getInt8PtrTy(rewriter.getContext()), v);
    } else {
      return rewriter.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMType::getInt8PtrTy(rewriter.getContext()), v);
    }
  }
}

Value transmuteToInt(Value v, ConversionPatternRewriter &rewriter,
                     Location loc) {
  llvm::errs() << "v: " << v << " |ty: " << v.getType() << "\n";
  if (v.getType().isa<LLVM::LLVMType>()) {
    LLVM::LLVMType vty = v.getType().cast<LLVM::LLVMType>();
    if (vty.isPointerTy()) {
      return rewriter.create<LLVM::PtrToIntOp>(
          loc, LLVM::LLVMType::getInt64Ty(rewriter.getContext()), v);
    } else if (vty.isIntegerTy()) {
      return v;
    } else {
      assert(false && "unable to transmute LLVM type into int");
    }
  } else {
    assert(v.getType().isa<HaskFnType>() || v.getType().isa<ValueType>() ||
           v.getType().isa<ThunkType>());
    // this maybe completely borked x(
    return rewriter.create<LLVM::PtrToIntOp>(
        loc, LLVM::LLVMType::getInt64Ty(rewriter.getContext()), v);
  }
}

// Value transmuteFromVoidPtr(Value v, LLVM::LLVMType desired,
// ConversionPatternRewriter &rewriter) {
//     assert(vty.isPointerTy());
//     LLVM::LLVMType vty = v.getType().cast<LLVM::LLVMType>();
//
//     if (desired.isPointerTy()) {
//         return rewriter.create<LLVM::BitcastOp>(
//                 LLVM::LLVMType::getInt8PtrTy(rewriter.getContext()), v);
//     } else if (vty.isIntegerTy()) {
//
//         return rewriter.create<LLVM::IntToPtrOp>(
//                 LLVM::LLVMType::getInt8PtrTy(rewriter.getContext()), v);
//     }
//     else {
//         assert(false && "unable to transmute into void pointer");
//     }
//
// }

class HaskToLLVMTypeConverter : public mlir::TypeConverter {
  using TypeConverter::TypeConverter;
};

mlir::LLVM::LLVMType haskToLLVMType(MLIRContext *context, Type t) {
  using namespace mlir::LLVM;

  llvm::errs() << __FUNCTION__ << "(" << t << ")\n";
  if (t.isa<ValueType>() || t.isa<ThunkType>()) {
    // return LLVMType::getInt64Ty(context);
    return LLVMType::getInt8PtrTy(context);
  } else if (auto fnty = t.dyn_cast<HaskFnType>()) {
    ArrayRef<Type> argTys = fnty.getInputTypes();
    Type retty = fnty.getResultType();

    // recall that functions can occur in negative position:
    // (a -> a) -> a
    // should become (int -> int) -> int
    std::vector<LLVMType> llvmArgTys;
    for (Type arg : argTys) {
      llvmArgTys.push_back(LLVMType::getInt8PtrTy(context));
      // llvmArgTys.push_back(haskToLLVMType(context, arg));
    }
    return LLVMType::getFunctionTy(haskToLLVMType(context, retty), llvmArgTys,
                                   /*isVaridic=*/false);
  } else {
    assert(false && "unknown haskell type");
  }
};

// http://localhost:8000/structanonymous__namespace_02ConvertStandardToLLVM_8cpp_03_1_1FuncOpConversion.html#a9043f45e0e37eb828942ff867c4fe38d
class HaskGlobalOpConversionPattern : public ConversionPattern {
public:
  explicit HaskGlobalOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::HaskGlobalOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using namespace mlir::LLVM;

    Operation *module = op->getParentOp();
    // Now lower the [LambdaOp + HaskFuncOp together]
    // TODO: deal with free floating lambads. This LambdaOp is going to
    // become a top-level function. Other lambdas will become toplevel functions
    // with synthetic names.
    llvm::errs() << "running HaskGlobalOpConversionPattern on: "
                 << op->getName() << " | " << op->getLoc() << "\n";
    auto fn = cast<HaskGlobalOp>(op);

    auto I8PtrTy = LLVMType::getInt8PtrTy(rewriter.getContext());
    LLVMFuncOp llvmfn = rewriter.create<LLVMFuncOp>(
        fn.getLoc(), fn.getName().str(), LLVMFunctionType::get(I8PtrTy, {}));

    Block *llvmfnEntry = llvmfn.addEntryBlock();
    rewriter.mergeBlocks(&fn.getRegion().getBlocks().front(), llvmfnEntry,
                         llvmfnEntry->getArguments());
    rewriter.eraseOp(op);

    // TODO: consider building a struct with the default closure
    // interface?
    // {
    //    mlir::LLVM::LLVMStructType closure0ArgsStruct;
    //    rewriter.create<mlir::LLVM::GlobalOp>(fn.getLoc(), fn.getName(),
    //                                          closure0ArgsStruct)

    return success();
  }
};

// http://localhost:8000/structanonymous__namespace_02ConvertStandardToLLVM_8cpp_03_1_1FuncOpConversion.html#a9043f45e0e37eb828942ff867c4fe38d
class HaskFuncOpConversionPattern : public ConversionPattern {
public:
  explicit HaskFuncOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::HaskFuncOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using namespace mlir::LLVM;

    Operation *module = op->getParentOp();
    // Now lower the [LambdaOp + HaskFuncOp together]
    // TODO: deal with free floating lambads. This LambdaOp is going to
    // become a top-level function. Other lambdas will become toplevel functions
    // with synthetic names.
    llvm::errs() << "running HaskFuncOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    auto fn = cast<HaskFuncOp>(op);
    LambdaOp lam = fn.getLambda();

    SmallVector<LLVM::LLVMType, 4> fnArgTys;
    auto I8PtrTy = LLVMType::getInt8PtrTy(rewriter.getContext());
    for (int i = 0; i < lam.getNumInputs(); ++i) {
      // fnArgTys.push_back(LLVMType::getInt64Ty(rewriter.getContext()));
      fnArgTys.push_back(I8PtrTy);
    }

    LLVMFuncOp llvmfn =
        rewriter.create<LLVMFuncOp>(fn.getLoc(), fn.getFuncName().str(),
                                    LLVMFunctionType::get(I8PtrTy, fnArgTys));

    Region &lamBody = lam.getBody();
    Block *llvmfnEntry = llvmfn.addEntryBlock();
    Block &lamEntry = lamBody.getBlocks().front();

    llvm::errs() << "converting lambda:\n";
    llvm::errs() << *lamBody.getParentOp() << "\n";
    rewriter.mergeBlocks(&lamBody.getBlocks().front(), llvmfnEntry,
                         llvmfnEntry->getArguments());

    rewriter.eraseOp(op);
    // llvm::errs() << *module << "\n";
    // assert(false);
    // assert(false);

    /*
    FuncOp stdFunc = ::mlir::FuncOp::create(fn.getLoc(),
            fn.getFuncName().str(),
            FunctionType::get({}, {rewriter.getI64Type()},
    rewriter.getContext())); rewriter.inlineRegionBefore(lam.getBody(),
    stdFunc.getBody(), stdFunc.end()); rewriter.insert(stdFunc);
    */

    return success();
  }
};

// isConstructorTagEq(TAG : char *, constructor: void * -> bool)
static FlatSymbolRefAttr
getOrInsertIsConstructorTagEq(PatternRewriter &rewriter, ModuleOp module) {
  const std::string name = "isConstructorTagEq";
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  auto llvmI1Ty = LLVM::LLVMType::getInt1Ty(rewriter.getContext());

  // constructor, string constructor name
  SmallVector<mlir::LLVM::LLVMType, 4> argsTy{llvmI8PtrTy, llvmI8PtrTy};
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI1Ty, argsTy,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

// extractConstructorArgN(constructor: void *, arg_ix: int) -> bool)
static FlatSymbolRefAttr
getOrInsertExtractConstructorArg(PatternRewriter &rewriter, ModuleOp module) {
  const std::string name = "extractConstructorArg";
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());

  // string constructor name, <n> arguments.
  SmallVector<mlir::LLVM::LLVMType, 4> argsTy{llvmI8PtrTy, llvmI64Ty};
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI8PtrTy, argsTy,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMType::getArrayTy(
        LLVM::LLVMType::getInt8Ty(builder.getContext()), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value));
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(builder.getContext()),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMType::getInt8PtrTy(builder.getContext()), globalPtr,
      ArrayRef<Value>({cst0, cst0}));
}

class CaseOpConversionPattern : public ConversionPattern {
public:
  explicit CaseOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::CaseOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using namespace mlir::LLVM;
    auto caseop = cast<CaseOp>(op);
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    const Optional<int> default_ix = caseop.getDefaultAltIndex();

    llvm::errs() << "running CaseOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    llvm::errs() << caseop << "\n";

    // delete the use of the case.
    // TODO: Change the IR so that we create a landing pad BB where the
    // case uses all wind up.
    for (Operation *user : op->getUsers()) {
      rewriter.eraseOp(user);
    }
    Value scrutinee = caseop.getScrutinee();

    rewriter.setInsertionPoint(caseop);
    // TODO: get block of current caseop?

    Block *prevBB = rewriter.getInsertionBlock();
    for (int i = 0; i < caseop.getNumAlts(); ++i) {
      if (default_ix && i == *default_ix) {
        continue;
      }
      // Type result, IntegerAttr predicate, Value lhs, Value rhs
      FlatSymbolRefAttr is_cons_tag_eq =
          getOrInsertIsConstructorTagEq(rewriter, mod);
      Value lhsName = getOrCreateGlobalString(
          caseop.getLoc(), rewriter, caseop.getAltLHS(i).getValue(),
          caseop.getAltLHS(i).getValue(), mod);
      SmallVector<Value, 4> is_cons_tag_eq_params{scrutinee, lhsName};
      LLVM::CallOp scrut_eq_alt = rewriter.create<LLVM::CallOp>(
          caseop.getLoc(), LLVMType::getInt1Ty(rewriter.getContext()),
          is_cons_tag_eq, is_cons_tag_eq_params);

      Block *thenBB =
          rewriter.createBlock(caseop.getParentRegion(), /*insertPt=*/{});

      rewriter.setInsertionPointToEnd(thenBB);
      Block &altRhs = caseop.getAltRHS(i).getBlocks().front();
      llvm::errs() << "--MERGE BLOCKS (CaseOp)--\n";
      SmallVector<Value, 4> extractedFields;
      FlatSymbolRefAttr extractConstructorArg =
          getOrInsertExtractConstructorArg(rewriter, mod);
      llvm::errs() << "-number of arguments: [" << altRhs.getNumArguments()
                   << "]--\n";
      for (int i = 0; i < altRhs.getNumArguments(); ++i) {
        Value ival = rewriter.create<LLVM::ConstantOp>(
            caseop.getLoc(), LLVMType::getInt64Ty(rewriter.getContext()),
            rewriter.getI64IntegerAttr(i));
        SmallVector<Value, 2> args = {scrutinee, ival};
        LLVM::CallOp call = rewriter.create<LLVM::CallOp>(
            caseop.getLoc(), LLVMType::getInt8PtrTy(rewriter.getContext()),
            extractConstructorArg, args);
        extractedFields.push_back(call.getResult(0));
      }
      llvm::errs() << "-----merging blocks-----\n";
      rewriter.mergeBlocks(&altRhs, thenBB, extractedFields);
      llvm::errs() << "--DONE MERGE BLOCKS (CaseOp)--\n";

      Block *elseBB =
          rewriter.createBlock(caseop.getParentRegion(), /*insertPt=*/{});
      rewriter.setInsertionPointToEnd(prevBB);
      rewriter.create<LLVM::CondBrOp>(
          rewriter.getUnknownLoc(), scrut_eq_alt.getResult(0), thenBB, elseBB);
      rewriter.setInsertionPointToEnd(elseBB);
    }

    // we have a default block
    if (default_ix) {
      // default block should have ha no parameters!
      llvm::errs() << "--MERGE BLOCKS (CaseOp/Default)--\n";
      rewriter.mergeBlocks(&caseop.getAltRHS(*default_ix).front(),
                           rewriter.getInsertionBlock(), scrutinee);
      llvm::errs() << "--MERGE BLOCKS (CaseOp/Default)--\n";
    } else {
      // wut?
      // if (!rewriter.getInsertionBlock()->getTerminator()) {
      rewriter.create<mlir::LLVM::UnreachableOp>(rewriter.getUnknownLoc());
      // }
    }
    rewriter.eraseOp(caseop);
    return success();
  }
};

class LambdaOpConversionPattern : public ConversionPattern {
public:
  explicit LambdaOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::LambdaOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto lam = cast<LambdaOp>(op);
    assert(lam);
    llvm::errs() << "running LambdaOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";

    return failure();
  }
};

// Humongous hack: we run a sort of "type inference" algorithm, where at the
// call-site, we convert from a !hask.untyped to a concrete (say, int64)
// type. We bail with an error if we are unable to replace the type.
void unifyOpTypeWithType(Value src, Type dstty) {
  if (src.getType() == dstty) {
    return;
  }
  assert(false && "unable to unify types!");
}

static FlatSymbolRefAttr getOrInsertMkClosure(PatternRewriter &rewriter,
                                              ModuleOp module, int n) {

  const std::string name = "mkClosure_capture0_args" + std::to_string(n);
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto I8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  llvm::SmallVector<LLVM::LLVMType, 4> argTys(n + 1, I8PtrTy);
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(I8PtrTy, argTys,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

class ApOpConversionPattern : public ConversionPattern {
public:
  explicit ApOpConversionPattern(MLIRContext *context)
      : ConversionPattern(ApOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using namespace mlir::LLVM;
    llvm::errs() << "running ApSSAConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    ApOp ap = cast<ApOp>(op);
    ModuleOp module = ap.getParentOfType<ModuleOp>();

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    LLVMType kparamty =
        haskToLLVMType(rewriter.getContext(), ap.getResult().getType());

    // Wow, in what order does the conversion happen? I have no idea.
    // LLVMFuncOp parent = ap.getParentOfType<LLVMFuncOp>();
    // assert(parent && "found lambda parent");
    // LLVMType kretty =
    // parent.getType().cast<LLVMFunctionType>().getReturnType(); llvm::errs()
    // << "kretty: " << kretty << "\n";

    // LLVMType kFnTy = LLVM::LLVMType::getFunctionTy(kretty, kparamty,
    //         /*isVarArg=*/false);
    // llvm::errs() << "kFnTy: " << kFnTy << "\n";
    // // I deserve to be shot for this. This is not even deterministic!
    // // I'm not even sure how to get deterministic names inside MLIR Which is
    // multi threaded.
    // // What information uniquely identifies an `ap`? it's parameters? but we
    // want names that are unique
    // // across functions. So hash(fn name + hash(params))? this is crazy.
    // // K = kontinuation.
    // std::string kname = "ap_" + std::to_string(rand());
    // // Insert the printf function into the body of the parent module.
    // PatternRewriter::InsertionGuard insertGuard(rewriter);
    // rewriter.setInsertionPointToStart(module.getBody());
    // LLVMFuncOp apK = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
    // kname, kFnTy); SymbolRefAttr kfn =  SymbolRefAttr::get(kname,
    // rewriter.getContext()); Block *entry = apK.addEntryBlock();
    // rewriter.setInsertionPoint(entry, entry->end());

    rewriter.setInsertionPointAfter(ap);
    SmallVector<Value, 4> llvmFnArgs;
    llvmFnArgs.push_back(transmuteToVoidPtr(ap.getFn(), rewriter, ap.getLoc()));
    for (int i = 0; i < ap.getNumFnArguments(); ++i) {
      llvmFnArgs.push_back(
          transmuteToVoidPtr(ap.getFnArgument(i), rewriter, ap.getLoc()));
    }

    FlatSymbolRefAttr mkclosure =
        getOrInsertMkClosure(rewriter, module, ap.getNumFnArguments());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, LLVMType::getInt8PtrTy(rewriter.getContext()), mkclosure,
        llvmFnArgs);
    return success();
  }
};

class MakeI64OpConversionPattern : public ConversionPattern {
public:
  explicit MakeI64OpConversionPattern(MLIRContext *context)
      : ConversionPattern(MakeI64Op::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running MakeI64OpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    MakeI64Op makei64 = cast<MakeI64Op>(op);
    auto I64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
    Value v = rewriter.create<mlir::LLVM::ConstantOp>(makei64.getLoc(), I64Ty,
                                                      makei64.getValue());
    auto I8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(makei64, I8PtrTy, v);
    //    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op,rewriter.getI64Type(),
    //    makei64.getValue());
    return success();
  }
};

static FlatSymbolRefAttr getOrInsertEvalClosure(PatternRewriter &rewriter,
                                                ModuleOp module) {
  const std::string name = "evalClosure";
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto VoidPtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(VoidPtrTy, VoidPtrTy,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

class ForceOpConversionPattern : public ConversionPattern {
public:
  explicit ForceOpConversionPattern(MLIRContext *context)
      : ConversionPattern(ForceOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ForceOp force = cast<ForceOp>(op);
    llvm::errs() << "running ForceOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    using namespace mlir::LLVM;
    // assert(force.getScrutinee().getType().isa<LLVMFunctionType>());
    // LLVMFunctionType scrutty =
    // force.getScrutinee().getType().cast<LLVMFunctionType>();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, LLVMType::getInt8PtrTy(rewriter.getContext()),
        getOrInsertEvalClosure(rewriter, force.getParentOfType<ModuleOp>()),
        force.getScrutinee());
    return success();
  }
};

class HaskReturnOpConversionPattern : public ConversionPattern {
public:
  explicit HaskReturnOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskReturnOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    HaskReturnOp ret = cast<HaskReturnOp>(op);
    llvm::errs() << "running HaskReturnOpConversionPattern on: "
                 << op->getName() << " | " << op->getLoc() << "\n";
    using namespace mlir::LLVM;
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(ret, ret.getInput());
    return success();
  }
};

class HaskRefOpConversionPattern : public ConversionPattern {
public:
  explicit HaskRefOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskRefOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    HaskRefOp ref = cast<HaskRefOp>(op);
    ModuleOp mod = ref.getParentOfType<ModuleOp>();

    llvm::errs() << "running HaskRefOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    using namespace mlir::LLVM;

    auto I8PtrTy = mlir::LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());

    Operation *referenced = mod.lookupSymbol(ref.getRef());
    assert(referenced && "reference does not exist");

    if (LLVMFuncOp llvmfn = dyn_cast<LLVMFuncOp>(referenced)) {
      LLVM::AddressOfOp addr = rewriter.create<LLVM::AddressOfOp>(
          op->getLoc(), llvmfn.getType().getPointerTo(), ref.getRef());
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, I8PtrTy, addr);
      return success();
    } else {
      // not yet converted.
      LLVMType llvmty;
      if (HaskFuncOp fn = mod.lookupSymbol<HaskFuncOp>(ref.getRef())) {
        llvmty =
            haskToLLVMType(rewriter.getContext(), ref.getResult().getType());

      } else if (mod.lookupSymbol<HaskGlobalOp>(ref.getRef())) {
        llvmty = LLVMType::getFunctionTy(I8PtrTy, {}, /*isVaridic=*/false);
      } else {
        assert(false && "unknown symbol");
      }
      LLVM::AddressOfOp addr = rewriter.create<LLVM::AddressOfOp>(
          op->getLoc(), llvmty, ref.getRef());
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, I8PtrTy, addr);
      return success();
    }
    return success();
  }
};

static FlatSymbolRefAttr getOrInsertMalloc(PatternRewriter &rewriter,
                                           ModuleOp module) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("malloc")) {
    return SymbolRefAttr::get("malloc", rewriter.getContext());
  }

  auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI8PtrTy, llvmI64Ty,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "malloc", llvmFnType);
  return SymbolRefAttr::get("malloc", rewriter.getContext());
}

static FlatSymbolRefAttr getOrInsertMkConstructor(PatternRewriter &rewriter,
                                                  ModuleOp module, int n) {
  const std::string name = "mkConstructor" + std::to_string(n);
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());

  // string constructor name, <n> arguments.
  SmallVector<mlir::LLVM::LLVMType, 4> argsTy(n + 1, llvmI8PtrTy);
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI8PtrTy, argsTy,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

class HaskConstructOpConversionPattern : public ConversionPattern {
public:
  explicit HaskConstructOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskConstructOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running HaskConstructOpConversionPattern on: "
                 << op->getName() << " | " << op->getLoc() << "\n";

    HaskConstructOp cons = cast<HaskConstructOp>(op);
    ModuleOp mod = cons.getParentOfType<ModuleOp>();

    using namespace mlir::LLVM;

    FlatSymbolRefAttr mkConstructor = getOrInsertMkConstructor(
        rewriter, op->getParentOfType<ModuleOp>(), cons.getNumOperands());

    Value consName = getOrCreateGlobalString(
        cons.getLoc(), rewriter, cons.getDataConstructorName(),
        cons.getDataConstructorName(), mod);
    SmallVector<Value, 4> args = {consName};
    for (int i = 0; i < cons.getNumOperands(); ++i) {
      // args.push_back(transmuteToVoidPtr(cons.getOperand(i), rewriter,
      //             cons.getLoc()));
      args.push_back(cons.getOperand(i));
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, LLVMType::getInt8PtrTy(rewriter.getContext()), mkConstructor, args);

    // FlatSymbolRefAttr malloc = getOrInsertMalloc(rewriter,
    // op->getParentOfType<ModuleOp>());
    // // allocate some huge amount because we can't be arsed to calculate the
    // correct ammount. static const int HUGE = 4200; mlir::LLVM::ConstantOp
    // mallocSz = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(),
    //                                         LLVMType::getInt64Ty(rewriter.getContext()),
    //                                         rewriter.getI32IntegerAttr(HUGE));

    // SmallVector<Value, 4> llvmFnArgs = {mallocSz};

    // rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op,
    //                                     LLVMType::getInt8PtrTy(rewriter.getContext()),
    //                                     malloc,
    //                                     llvmFnArgs);

    return success();
  }
};

class HaskPrimopAddOpConversionPattern : public ConversionPattern {
public:
  explicit HaskPrimopAddOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskPrimopAddOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    using namespace mlir::LLVM;
    HaskPrimopAddOp add = cast<HaskPrimopAddOp>(op);
    auto I64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
    auto I8PtrTy = LLVMType::getInt8PtrTy(rewriter.getContext());

    LLVM::PtrToIntOp lhs = rewriter.create<LLVM::PtrToIntOp>(
        op->getLoc(), I64Ty, add.getOperand(0));
    LLVM::PtrToIntOp rhs = rewriter.create<LLVM::PtrToIntOp>(
        op->getLoc(), I64Ty, add.getOperand(1));

    LLVM::AddOp out =
        rewriter.create<LLVM::AddOp>(op->getLoc(), I64Ty, lhs, rhs);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, I8PtrTy, out);

    return success();
  }
};

class HaskPrimopSubOpConversionPattern : public ConversionPattern {
public:
  explicit HaskPrimopSubOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskPrimopSubOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using namespace mlir::LLVM;
    HaskPrimopSubOp sub = cast<HaskPrimopSubOp>(op);
    auto I64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
    auto I8PtrTy = LLVMType::getInt8PtrTy(rewriter.getContext());

    LLVM::PtrToIntOp lhs = rewriter.create<LLVM::PtrToIntOp>(
        op->getLoc(), I64Ty, sub.getOperand(0));
    LLVM::PtrToIntOp rhs = rewriter.create<LLVM::PtrToIntOp>(
        op->getLoc(), I64Ty, sub.getOperand(1));

    LLVM::SubOp out =
        rewriter.create<LLVM::SubOp>(op->getLoc(), I64Ty, lhs, rhs);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, I8PtrTy, out);

    return success();
  }
};

static FlatSymbolRefAttr getOrInsertIsIntEq(PatternRewriter &rewriter,
                                            ModuleOp module) {
  const std::string name = "isIntEq";
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  auto llvmI1Ty = LLVM::LLVMType::getInt1Ty(rewriter.getContext());
  SmallVector<mlir::LLVM::LLVMType, 4> argsTy{llvmI8PtrTy, llvmI8PtrTy};
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI1Ty, argsTy,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

Block *splitBlockAfter(PatternRewriter &rewriter, Block::iterator after) {
  return rewriter.splitBlock(after->getBlock(), ++after);
}

class CaseIntOpConversionPattern : public ConversionPattern {
public:
  explicit CaseIntOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::CaseIntOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using namespace mlir::LLVM;
    auto caseop = cast<CaseIntOp>(op);


    const Optional<int> default_ix = caseop.getDefaultAltIndex();

    llvm::errs() << "running CaseIntOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";
    llvm::errs() << caseop << "\n";

    Value scrutineeInt = rewriter.create<LLVM::PtrToIntOp>(
        caseop.getLoc(), LLVM::LLVMType::getInt64Ty(rewriter.getContext()),
        caseop.getScrutinee());
    Type caseRetty = caseop.getResult().getType();

    Block *prevBB = caseop.getOperation()->getBlock();
    Block *afterCaseBB = splitBlockAfter(rewriter, caseop.getOperation()->getIterator());

    afterCaseBB->addArgument(caseRetty);

    // Type result, IntegerAttr predicate, Value lhs, Value rhs
    auto I64Ty = LLVM::LLVMType::getInt64Ty(rewriter.getContext());
    auto I8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());

    // TODO: get block of current caseop?
    llvm::errs() << __LINE__ << "\n";
    for (int i = 0; i < caseop.getNumAlts(); ++i) {

      if (default_ix && i == *default_ix) { continue; }

      Block *thenBB =
          rewriter.createBlock(caseop.getParentRegion(), /*insertPt=*/{});
      Block *elseBB =
          rewriter.createBlock(caseop.getParentRegion(), /*insertPt=*/{});

      LLVM::ConstantOp altLhsInt = rewriter.create<LLVM::ConstantOp>(
          caseop.getLoc(), I64Ty, *caseop.getAltLHS(i));

      // prev -> {then, else}
      rewriter.setInsertionPointToEnd(prevBB);
      Value scrut_eq_alt = rewriter.create<LLVM::ICmpOp>(
          caseop.getLoc(), LLVM::ICmpPredicate::eq, scrutineeInt, altLhsInt);

      rewriter.create<LLVM::CondBrOp>(rewriter.getUnknownLoc(), scrut_eq_alt,
                                      thenBB, elseBB);

      // then -> code
      rewriter.setInsertionPointToEnd(thenBB);
      Block &altRhs = caseop.getAltRHS(i).getBlocks().front();
      HaskReturnOp altRhsRet = cast<HaskReturnOp>(altRhs.getTerminator());
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(altRhsRet, altRhsRet.getOperand(),
                                              afterCaseBB);
      rewriter.mergeBlocks(&altRhs, thenBB, scrutineeInt);

      // next
      prevBB = elseBB;
    }

    // we have a default block
    if (default_ix) {
      // default block should have have no parameters?
      Block &defaultRhsBlock = caseop.getAltRHS(*default_ix).front();
      HaskReturnOp defaultRet = cast<HaskReturnOp>(defaultRhsBlock.getTerminator());
      rewriter.mergeBlocks(&defaultRhsBlock,
                           prevBB, {});
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(defaultRet,
                                              defaultRet.getOperand(),
                                              afterCaseBB);

    } else {
      rewriter.setInsertionPointToEnd(prevBB);
      rewriter.create<mlir::LLVM::UnreachableOp>(rewriter.getUnknownLoc());
    }

    rewriter.replaceOp(caseop, afterCaseBB->getArgument(0));
    return success();
  }

};

static FlatSymbolRefAttr getOrInsertMkClosureThunkify(PatternRewriter &rewriter,
                                                      ModuleOp module) {

  const std::string name = "mkClosure_thunkify";
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return SymbolRefAttr::get(name, rewriter.getContext());
  }

  auto I8PtrTy = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
  llvm::SmallVector<LLVM::LLVMType, 4> argTys{I8PtrTy};
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(I8PtrTy, argTys,
                                                  /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, llvmFnType);
  return SymbolRefAttr::get(name, rewriter.getContext());
}

class ThunkifyOpConversionPattern : public ConversionPattern {
public:
  explicit ThunkifyOpConversionPattern(MLIRContext *context)
      : ConversionPattern(ThunkifyOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running ThunkifyOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";

    ThunkifyOp thunkify = cast<ThunkifyOp>(op);
    ModuleOp mod = thunkify.getParentOfType<ModuleOp>();

    using namespace mlir::LLVM;

    FlatSymbolRefAttr llvmThunkifyFn =
        getOrInsertMkClosureThunkify(rewriter, mod);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, LLVMType::getInt8PtrTy(rewriter.getContext()), llvmThunkifyFn,
        thunkify.getScrutinee());
    return success();
  }
};

class TransmuteOpConversionPattern : public ConversionPattern {
public:
  explicit TransmuteOpConversionPattern(MLIRContext *context)
      : ConversionPattern(TransmuteOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running ThunkifyOpConversionPattern on: " << op->getName()
                 << " | " << op->getLoc() << "\n";

    TransmuteOp transmute = cast<TransmuteOp>(op);
    rewriter.replaceOp(transmute, transmute.getOperand());
    return success();
  }
};

// === LowerHaskToLLVMPass ===
// === LowerHaskToLLVMPass ===
// === LowerHaskToLLVMPass ===

// === LowerHaskToStandardPass ===
// === LowerHaskToStandardPass ===
// === LowerHaskToStandardPass ===
// === LowerHaskToStandardPass ===

namespace {
struct LowerHaskToStandardPass : public Pass {
  LowerHaskToStandardPass()
      : Pass(mlir::TypeID::get<LowerHaskToStandardPass>()){};
  void runOnOperation();
  StringRef getName() const override { return "LowerHaskToStandardPass"; }

  std::unique_ptr<Pass> clonePass() const override {
    auto newInst = std::make_unique<LowerHaskToStandardPass>(
        *static_cast<const LowerHaskToStandardPass *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }
};
} // end anonymous namespace.

// http://localhost:8000/ConvertSPIRVToLLVMPass_8cpp_source.html#l00031
void LowerHaskToStandardPass::runOnOperation() {
  ConversionTarget target(getContext());
  // do I not need a pointer to the dialect? I am so confused :(
  HaskToLLVMTypeConverter converter();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::LLVM::UnreachableOp>();

  // Why do I need this? Isn't adding StandardOpsDialect enough?
  target.addLegalOp<mlir::FuncOp>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::ModuleTerminatorOp>();

  target.addIllegalDialect<standalone::HaskDialect>();
  // target.addLegalOp<HaskRefOp>();
  target.addLegalOp<HaskADTOp>();
  target.addLegalOp<LambdaOp>();
  // target.addLegalOp<HaskGlobalOp>();
  // target.addLegalOp<HaskReturnOp>();
  // target.addLegalOp<CaseOp>();
  // target.addLegalOp<ApOp>();
  // target.addLegalOp<HaskConstructOp>();
  // target.addLegalOp<ForceOp>();

  OwningRewritePatternList patterns;
  patterns.insert<HaskFuncOpConversionPattern>(&getContext());
  patterns.insert<HaskGlobalOpConversionPattern>(&getContext());
  patterns.insert<CaseOpConversionPattern>(&getContext());
  // patterns.insert<LambdaOpConversionPattern>(&getContext());
  patterns.insert<MakeI64OpConversionPattern>(&getContext());
  patterns.insert<HaskReturnOpConversionPattern>(&getContext());
  patterns.insert<HaskRefOpConversionPattern>(&getContext());
  patterns.insert<HaskConstructOpConversionPattern>(&getContext());
  patterns.insert<ForceOpConversionPattern>(&getContext());
  patterns.insert<ApOpConversionPattern>(&getContext());
  patterns.insert<HaskPrimopAddOpConversionPattern>(&getContext());
  patterns.insert<HaskPrimopSubOpConversionPattern>(&getContext());
  patterns.insert<CaseIntOpConversionPattern>(&getContext());
  patterns.insert<ThunkifyOpConversionPattern>(&getContext());
  patterns.insert<TransmuteOpConversionPattern>(&getContext());

  // llvm::errs() << "===Enabling Debugging...===\n";
  //::llvm::DebugFlag = true;

  if (failed(applyPartialConversion(this->getOperation(), target, patterns))) {
    llvm::errs() << "===Partial conversion failed===\n";
    getOperation()->print(llvm::errs());
    llvm::errs() << "\n===\n";
    signalPassFailure();
  } else {
    llvm::errs() << "===Partial conversion succeeded===\n";
    getOperation()->print(llvm::errs());
    llvm::errs() << "\n===\n";
  }

  //  ::llvm::DebugFlag = false;
  return;
}

std::unique_ptr<mlir::Pass> createLowerHaskToStandardPass() {
  return std::make_unique<LowerHaskToStandardPass>();
}

// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===

// this is run in a second phase, so we delete data constructors only
// after we are done processing data constructors.
/*
class MakeDataConstructorOpConversionPattern : public ConversionPattern {
public:
  explicit MakeDataConstructorOpConversionPattern(MLIRContext *context)
      : ConversionPattern(DeclareDataConstructorOp::getOperationName(), 1,
context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
      rewriter.eraseOp(op);
    return success();
  }
};
*/

class HaskADTOpConversionPattern : public ConversionPattern {
public:
  explicit HaskADTOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskADTOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

namespace {
struct LowerHaskStandardToLLVMPass : public Pass {
  LowerHaskStandardToLLVMPass()
      : Pass(mlir::TypeID::get<LowerHaskStandardToLLVMPass>()){};
  StringRef getName() const override { return "LowerHaskToStandardPass"; }

  std::unique_ptr<Pass> clonePass() const override {
    auto newInst = std::make_unique<LowerHaskStandardToLLVMPass>(
        *static_cast<const LowerHaskStandardToLLVMPass *>(this));
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
    patterns.insert<HaskADTOpConversionPattern>(&getContext());

    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
    if (failed(mlir::applyFullConversion(getOperation(), target, patterns))) {
      llvm::errs() << "===Hask+Std -> LLVM lowering failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
    };
  };
};
} // end anonymous namespace.

std::unique_ptr<mlir::Pass> createLowerHaskStandardToLLVMPass() {
  return std::make_unique<LowerHaskStandardToLLVMPass>();
}

} // namespace standalone
} // namespace mlir
