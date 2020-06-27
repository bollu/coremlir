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
    
    Region *fn = result.addRegion();
    // (<fn-arg>
    if (parser.parseLParen() || parser.parseRegion(*fn, {}, {})) { return failure(); }


    // ["," <arg>]
    while(succeeded(parser.parseOptionalComma())) {
            Region *arg = result.addRegion(); parser.parseRegion(*arg, {}, {});
    }   
    //)
    return parser.parseRParen();
};

void ApOp::print(OpAsmPrinter &p) {
    p << "standalone.ap("; p.printRegion(getFn(), /*blockArgs=*/false);
    
    for(int i = 0; i < getNumFnArguments(); ++i) {
        p << ","; p.printRegion(getFnArgument(i), /*blockArgs=*/false);
    }
    p << ")";
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



// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===


ParseResult MakeI32Op::parse(OpAsmParser &parser, OperationState &result) {
    mlir::OpAsmParser::OperandType i;
    if (parser.parseLParen() || parser.parseOperand(i) || parser.parseRParen())
        return failure();
    SmallVector<Value, 1> vi;
    parser.resolveOperand(i, parser.getBuilder().getIntegerType(32), vi);
    result.addOperands(vi);
    return success();
};

void MakeI32Op::print(OpAsmPrinter &p) {
    p << "standalone.make_i32(" << getInput() << ")";
};

// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===

ParseResult MakeDataConstructorOp::parse(OpAsmParser &parser, OperationState &result) {
    // parser.parseAttribute(, parser.getBuilder().getStringAttr )
    Attribute attr;
    if(parser.parseLess()) return failure();
    if(parser.parseAttribute(attr, "name", result.attributes)) { assert(false && "unable to parse attribute!");  return failure(); }
    if(parser.parseGreater()) return failure();
    llvm::errs() << "- " << __FUNCTION__ << ":" << __LINE__ << "attribute: " << attr << "\n";
    llvm::errs() << "- " << __FUNCTION__ << ":" << __LINE__ << "attribute ty: " << attr.getType() << "\n";

    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};
void MakeDataConstructorOp::print(OpAsmPrinter &p) {
    p << getOperationName() << "<" << getAttr("name")  << ">";
};




// === DomninanceFreeScope OP ===
// === DomninanceFreeScope OP ===
// === DomninanceFreeScope OP ===
// === DomninanceFreeScope OP ===
// === DomninanceFreeScope OP ===



ParseResult DominanceFreeScopeOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // magic++, wtf++;
  // DominanceFreeScopeOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void DominanceFreeScopeOp::print(OpAsmPrinter &p) {
    p << getOperationName();
    p.printRegion(getRegion(), /*printEntry=*/false);
};

void DominanceFreeScopeOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType) {
      odsState.addTypes(resultType);
};



// === MakeTopLevelBinding OP ===
// === MakeTopLevelBinding OP ===
// === MakeTopLevelBinding OP ===
// === MakeTopLevelBinding OP ===
// === MakeTopLevelBinding OP ===

ParseResult TopLevelBindingOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType body;

    // if(parser.parseOperand(body)) return failure();

    if(parser.parseRegion(*result.addRegion(), {}, {})) return failure();
    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};

void TopLevelBindingOp::print(OpAsmPrinter &p) {
    p.printRegion(getBody(), /*printEntry=*/false);
};



} // namespace standalone
} // namespace mlir
