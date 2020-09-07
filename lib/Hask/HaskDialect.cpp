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
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::standalone;

//===----------------------------------------------------------------------===//
// Hask dialect.
//===----------------------------------------------------------------------===//


HaskDialect::HaskDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HaskDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Hask/HaskOps.cpp.inc"
  >();
 addOperations<HaskReturnOp, MakeI64Op,
  DeclareDataConstructorOp,
  ApOp, CaseOp, HaskRefOp, LambdaOp,
  MakeStringOp, HaskFuncOp, ForceOp, HaskGlobalOp, HaskADTOp,
  HaskConstructOp>();
  //addTypes<UntypedType>();
  addTypes<ThunkType, ValueType, HaskFnType>();
  addAttributes<DataConstructorAttr>();
}

mlir::Type HaskDialect::parseType(mlir::DialectAsmParser &parser) const {
  if(succeeded(parser.parseOptionalKeyword("thunk"))) {
    return ThunkType::get(parser.getBuilder().getContext());
  } else if(succeeded(parser.parseOptionalKeyword("value"))) {
    return ValueType::get(parser.getBuilder().getContext());
  } else if (succeeded(parser.parseOptionalKeyword("func"))) {
      Type param, res;
      if (parser.parseLess() ||
          parser.parseType(param) || parser.parseComma() || 
          parser.parseType(res) || parser.parseGreater()) {
          parser.emitError(parser.getCurrentLocation(),
                                  "unable to parse function type");
          return Type();
      }

      return HaskFnType::get(parser.getBuilder().getContext(), param, res);
  } else {
      assert(false && "unknown type");
  }
  return Type();
}


void HaskDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &p) const {
  if (type.isa<ThunkType>()) { p << "thunk"; }
  else if (type.isa<ValueType>()) { p << "value"; }
  else if (type.isa<HaskFnType>()) {
      HaskFnType fnty = type.cast<HaskFnType>();
      p << "fn<" << *fnty.getInputType().data() << ", " <<
                *fnty.getResultType().data() << ">";
  }
  else { assert(false && "unknown type"); }
}

// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===


mlir::Attribute HaskDialect::parseAttribute(mlir::DialectAsmParser &parser, Type type) const {
    if (succeeded(parser.parseOptionalKeyword("data_constructor"))) {
        return parseDataConstructorAttribute(parser, type);
    } else { assert(false && "unable to parse attribute"); }
    return Attribute();
};


void HaskDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
    assert(attr);
    if(attr.isa<DataConstructorAttr>()) {
        DataConstructorAttr d = attr.cast<DataConstructorAttr>();
        p << "data_constructor<";
        p << *d.getName().data()  << " " << *d.getArgTys().data();
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

Attribute standalone::parseDataConstructorAttribute(DialectAsmParser &parser, Type type) {
    if(parser.parseLess()) return Attribute();
    SymbolRefAttr name;
    if (parser.parseAttribute<SymbolRefAttr>(name)) return Attribute();
    ArrayAttr paramTys;
    if(parser.parseAttribute<ArrayAttr>(paramTys)) return Attribute();

    if(parser.parseGreater()) return Attribute();

    Attribute a =  DataConstructorAttr::get(parser.getBuilder().getContext(),
            name, paramTys);
    assert(a && "have valid attribute");
    llvm::errs() << __FUNCTION__  << "\n";
    llvm::errs() << "  attr: " << a << "\n";
    llvm::errs() << "===\n";
    return a;
};

// === LOWERING ===



