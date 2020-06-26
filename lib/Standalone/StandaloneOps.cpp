//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

// includes
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"


namespace mlir {
namespace standalone {
#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.cpp.inc"


// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
ParseResult LambdaOp::parse(OpAsmParser &parser, OperationState &result) {
    SmallVector<mlir::OpAsmParser::OperandType, 4> regionArgs;
    if (parser.parseRegionArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::OptionalSquare)) return failure();


    for(int i = 0; i < regionArgs.size(); ++ i) {
    	llvm::errs() << "-" << __FUNCTION__ << ":" << __LINE__ << ":" << regionArgs[i].name <<"\n";
    }

    Region *r = result.addRegion();
    return parser.parseRegion(*r, regionArgs, {parser.getBuilder().getNoneType()});
    // return success();

    // assert(	false);
}

void LambdaOp::print(OpAsmPrinter &p) {
    p << "standalone.lambda";
    p << "[";
        for(int i = 0; i < this->getNumInputs(); ++i) {
            p << this->getInput(i);
            if (i < this->getNumInputs() - 1) { p << ","; }
        } 
    p << "]";
    p.printRegion(this->getBody(), /*printEntryBlockArgs=*/false);
    // p.printRegion(this->getBody(), /*printEntryBlockArgs=*/true);
}


// === CASE OP ===
// === CASE OP ===
// === CASE OP ===
// === CASE OP ===
// === CASE OP ===

ParseResult CaseOp::parse(OpAsmParser &parser, OperationState &result) {
    Region *r = result.addRegion();
    if(parser.parseRegion(*r, {}, {})) return failure();
    llvm::errs() << "***" << __FUNCTION__ << ":" << __LINE__ << ": " <<  "\n"; 
    llvm::errs() << "***" << __FUNCTION__ << ":" << __LINE__ << ": " << "parsing attributes...\n"; 
    if(parser.parseOptionalAttrDict(result.attributes)) return failure();

    // for (auto attr : result.attributes.getAttrs()) llvm::errs() << "-" << attr.first <<"=" << attr.second << "\n";
    // llvm::errs() << << "\n";
    llvm::errs() <<  "***" << __FUNCTION__ << ":" << __LINE__ << ": " << " num parsed attributes: " << result.attributes.getAttrs().size() <<  "...\n"; 
    for(int i = 0; i < result.attributes.getAttrs().size(); ++i) {
        Region *r = result.addRegion();
        if(parser.parseRegion(*r, {}, {})) return failure(); 
    }

    return success();

};

void CaseOp::print(OpAsmPrinter &p) {
    p << "standalone.case";
    
    p.printRegion(this->getScrutineeRegion());
    p.printOptionalAttrDict(this->getAltLHSs().getValue());
    for(int i = 0; i < this->getNumAlts(); ++i) {
        p.printRegion(this->getAltRHS(i)); 
    }
};


// === AP OP ===
// === AP OP ===
// === AP OP ===
// === AP OP ===
// === AP OP ===


ParseResult ApOp::parse(OpAsmParser &parser, OperationState &result) {
    assert(false);
};

void ApOp::print(OpAsmPrinter &p) {
    p << "ap";
};



// === RETURN OP ===
// === RETURN OP ===
// === RETURN OP ===
// === RETURN OP ===
// === RETURN OP ===


ParseResult ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
    mlir::OpAsmParser::OperandType i;
    if (parser.parseLParen() || parser.parseOperand(i) || parser.parseRParen())
        return failure();
    SmallVector<Value, 1> vi;
    parser.resolveOperand(i, parser.getBuilder().getNoneType(), vi);
    result.addOperands(vi);
    return success();
};

void ReturnOp::print(OpAsmPrinter &p) {
    p << "standalone.return(" << getInput() << ")";
};

} // namespace standalone
} // namespace mlir
