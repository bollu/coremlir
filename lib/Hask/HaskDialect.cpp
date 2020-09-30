//===- HaskDialect.cpp - Hask dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"
#include "mlir/IR/StandardTypes.h"

// includes
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::standalone;
//===----------------------------------------------------------------------===//
// Hask type.
//===----------------------------------------------------------------------===//

bool HaskType::classof(Type type) {
  return llvm::isa<HaskDialect>(type.getDialect());
}

HaskDialect &HaskType::getDialect() {
  return static_cast<HaskDialect &>(Type::getDialect());
}

//===----------------------------------------------------------------------===//
// Hask dialect.
//===----------------------------------------------------------------------===//

HaskDialect::HaskDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HaskDialect>()) {
//   addOperations<
// #define GET_OP_LIST
// #include "Hask/HaskOps.cpp.inc"
//       >();
  addOperations<HaskReturnOp, MakeI64Op,
                // DeclareDataConstructorOp,
                ApOp, ApEagerOp, CaseOp, DefaultCaseOp, HaskRefOp, LambdaOp, MakeStringOp, HaskFuncOp,
                ForceOp, HaskGlobalOp, HaskADTOp, HaskConstructOp,
                HaskPrimopAddOp, HaskPrimopSubOp, CaseIntOp, ThunkifyOp,
                TransmuteOp>();
  addTypes<ThunkType, ValueType, HaskFnType, ADTType>();
  addAttributes<DataConstructorAttr>();
}

mlir::Type HaskDialect::parseType(mlir::DialectAsmParser &parser) const {
  if (succeeded(parser.parseOptionalKeyword("thunk"))) {
    Type t;
    if (parser.parseLess() || parser.parseType(t) || parser.parseGreater()) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse ThunkType");
      return Type();
    }
    return ThunkType::get(parser.getBuilder().getContext(), t);

  } else if (succeeded(parser.parseOptionalKeyword("value"))) {
    return ValueType::get(parser.getBuilder().getContext());
  } else if (succeeded(parser.parseOptionalKeyword("adt"))) {
    FlatSymbolRefAttr name;
    if (parser.parseLess() ||
        parser.parseAttribute<mlir::FlatSymbolRefAttr>(name) ||
        parser.parseGreater()) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse ADT type. Missing `<`");
      return Type();
    }
    return ADTType::get(parser.getBuilder().getContext(), name);
  } else if (succeeded(parser.parseOptionalKeyword("fn"))) {
    SmallVector<Type, 4> params;
    Type res;
    if (parser.parseLess()) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse function type. Missing `<`");
      return Type();
    }

    if (parser.parseLParen()) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse function type. Missing `(`");
      return Type();
    }

    if (succeeded(parser.parseOptionalRParen())) {
      // empty function
    } else {
      while (1) {
        Type t;
        if (parser.parseType(t)) {
          parser.emitError(parser.getCurrentLocation(),
                           "unable to parse argument type");
          return Type();
        }

        params.push_back(t);

        if (succeeded(parser.parseOptionalRParen())) {
          break;
        }

        if (parser.parseComma()) {
          parser.emitError(parser.getCurrentLocation(),
                           "unable to parse function type. Missing `,` after"
                           "argument");
        }
      }
    }

    if (parser.parseArrow()) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse function type. Missing `->`"
                       "after argument list");
      return Type();
    }

    if (parser.parseType(res)) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse return type.");
      return Type();
    }

    if (parser.parseGreater()) {
      parser.emitError(parser.getCurrentLocation(),
                       "unable to parse function type. Missing `>`");
      return Type();
    }

    return HaskFnType::get(parser.getBuilder().getContext(), params, res);
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown type for hask dialect");
    assert(false && "unknown type");
  }
  return Type();
}

void HaskDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &p) const {
  if (type.isa<ThunkType>()) {
    ThunkType thunk = type.cast<ThunkType>();
    p << "thunk<" << thunk.getElementType() << ">";
  } else if (type.isa<ADTType>()) {
    ADTType adt = type.cast<ADTType>();
    p << "adt<@" << adt.getName().getValue() << ">";
  } else if (type.isa<ValueType>()) {
    p << "value";
  } else if (type.isa<HaskFnType>()) {
    HaskFnType fnty = type.cast<HaskFnType>();
    ArrayRef<Type> intys = fnty.getInputTypes();

    p << "fn<(";
    for (int i = 0; i < intys.size(); ++i) {
      p << intys[i];
      if (i + 1 < intys.size())
        p << ", ";
    }
    p << ") -> " << fnty.getResultType() << ">";
  } else {
    assert(false && "unknown type");
  }
}

// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===

mlir::Attribute HaskDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                            Type type) const {
  if (succeeded(parser.parseOptionalKeyword("data_constructor"))) {
    return parseDataConstructorAttribute(parser, type);
  } else {
    assert(false && "unable to parse attribute");
  }
  return Attribute();
};

void HaskDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  assert(attr);
  if (attr.isa<DataConstructorAttr>()) {
    DataConstructorAttr d = attr.cast<DataConstructorAttr>();
    p << "data_constructor<";
    p << *d.getName().data() << " " << *d.getArgTys().data();
    p << ">";
  } else {
    assert(false && "unknown attribute");
  }
}

// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===

Attribute standalone::parseDataConstructorAttribute(DialectAsmParser &parser,
                                                    Type type) {
  if (parser.parseLess())
    return Attribute();
  SymbolRefAttr name;
  if (parser.parseAttribute<SymbolRefAttr>(name))
    return Attribute();
  ArrayAttr paramTys;
  if (parser.parseAttribute<ArrayAttr>(paramTys))
    return Attribute();

  if (parser.parseGreater())
    return Attribute();

  Attribute a = DataConstructorAttr::get(parser.getBuilder().getContext(), name,
                                         paramTys);
  assert(a && "have valid attribute");
  llvm::errs() << __FUNCTION__ << "\n";
  llvm::errs() << "  attr: " << a << "\n";
  llvm::errs() << "===\n";
  return a;
};

// === LOWERING ===

