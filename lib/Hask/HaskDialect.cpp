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
  MakeStringOp, HaskFuncOp, ForceOp, CopyOp, HaskADTOp>();

  addTypes<UntypedType>();
}

mlir::Type HaskDialect::parseType(mlir::DialectAsmParser &parser) const {
  if(succeeded(parser.parseKeyword("untyped"))) {
    return UntypedType::get(parser.getBuilder().getContext());
  }
  return Type();
}


void HaskDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<UntypedType>()) {
    printer << "untyped";
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
      parseDataConstructorAttribute(parser, type);
    }
    return Attribute();
  };



// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===
// === DATA CONSTRUCTOR ATTRIBUTE ===

  Attribute standalone::parseDataConstructorAttribute(DialectAsmParser &parser, Type type) {
    return Attribute();
  };

// === LOWERING ===



