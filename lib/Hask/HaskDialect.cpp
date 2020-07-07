//===- HaskDialect.cpp - Hask dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"

using namespace mlir;
using namespace mlir::standalone;

//===----------------------------------------------------------------------===//
// Hask dialect.
//===----------------------------------------------------------------------===//

HaskDialect::HaskDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "Hask/HaskOps.cpp.inc"
  >();
 addOperations<LambdaOp, CaseOp, ApOp, ReturnOp, MakeI32Op, 
  MakeDataConstructorOp, TopLevelBindingOp, DominanceFreeScopeOp, ModuleOp, 
  DummyFinishOp, ConstantOp, ApSSAOp, CaseSSAOp, RecursiveRefOp, LambdaSSAOp,
  MakeStringOp>();
}
