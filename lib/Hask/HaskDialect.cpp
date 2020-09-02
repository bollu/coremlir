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
  MakeDataConstructorOp,
  ApSSAOp, CaseSSAOp, HaskRefOp, LambdaSSAOp,
  MakeStringOp, HaskFuncOp, ForceOp, CopyOp>();
  addOperations<HaskADTOp>();
  addTypes<UntypedType>();
  addAttributes<DataConstructorAttr>();
}

mlir::Type HaskDialect::parseType(mlir::DialectAsmParser &parser) const {
  if(succeeded(parser.parseKeyword("untyped"))) {
    return UntypedType::get(parser.getBuilder().getContext());
  }
  return Type();
}


void HaskDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &p) const {
  if (type.isa<UntypedType>()) {
    p << "untyped";
  } else {
    assert(false && "unknown type");
  }
}

// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===
// === ATTRIBUTE HANDLING ===


  mlir::Attribute HaskDialect::parseAttribute(mlir::DialectAsmParser &parser, Type type) const {
    if (succeeded(parser.parseKeyword("data_constructor"))) {
      return parseDataConstructorAttribute(parser, type);
    }
    return Attribute();
  };


  void HaskDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
    assert(attr);
    if(attr.isa<DataConstructorAttr>()) {
      DataConstructorAttr d = attr.cast<DataConstructorAttr>();
      p << "data_constructor<";

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



