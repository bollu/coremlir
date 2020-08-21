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
 addOperations<HaskReturnOp, MakeI32Op,
  MakeDataConstructorOp, HaskModuleOp,
  DummyFinishOp, ApSSAOp, CaseSSAOp, RecursiveRefOp, LambdaSSAOp,
  MakeStringOp, HaskFuncOp, ForceOp, CopyOp>();

  addTypes<UntypedType>();
}

mlir::Type HaskDialect::parseType(mlir::DialectAsmParser &parser) const {
  if(failed(parser.parseKeyword("untyped"))) { return Type(); }
  return UntypedType::get(parser.getBuilder().getContext());
}

void HaskDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<UntypedType>()) {
    printer << "untyped";
  } else {
    assert(false && "unknown type");
  }
}

// === LOWERING ===



