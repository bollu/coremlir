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
  return success();

  // magic++, wtf++;
  // DominanceFreeScopeOp::ensureTerminator(*body, parser.getBuilder(), result.location);
}
void DominanceFreeScopeOp::print(OpAsmPrinter &p) {
    p << getOperationName(); p.printRegion(getRegion(), /*printEntry=*/false);
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


    if(parser.parseRegion(*result.addRegion(), {}, {})) return failure();
    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};

void TopLevelBindingOp::print(OpAsmPrinter &p) {
    p << getOperationName(); p.printRegion(getBody(), /*printEntry=*/false);
};

// === Module OP ===
// === Module OP ===
// === Module OP ===
// === Module OP ===
// === Module OP ===

ParseResult ModuleOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType body;
    if(parser.parseRegion(*result.addRegion(), {}, {})) return failure();
    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};

void ModuleOp::print(OpAsmPrinter &p) {
    p << getOperationName(); p.printRegion(getBody(), /*printEntry=*/false);
};

// === DummyFinish OP ===
// === DummyFinish OP ===
// === DummyFinish OP ===
// === DummyFinish OP ===
// === DummyFinish OP ===

ParseResult DummyFinishOp::parse(OpAsmParser &parser, OperationState &result) {
    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};

void DummyFinishOp::print(OpAsmPrinter &p) {
    p << getOperationName();
};

// === CONSTANT OP ===
// === CONSTANT OP ===
// === CONSTANT OP ===
// === CONSTANT OP ===
// === CONSTANT OP ===

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
    // TODO: how to parse an int32?

    mlir::OpAsmParser::OperandType const_operand;
    Type const_type;
    
    if (parser.parseLParen() || parser.parseOperand(const_operand) ||
        parser.parseComma() || parser.parseType(const_type) || parser.parseRParen()) {
            return failure();
        }
    SmallVector<mlir::Value, 1> const_value;
    if(parser.resolveOperand(const_operand, const_type, const_value)) { return failure(); }
    result.addOperands(const_value);
    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};

void ConstantOp::print(OpAsmPrinter &p) {
    // p << getOperationName() << "(" << "[" << getOperation()->getNumOperands() << "]" << ")";
    p << getOperationName() << "(" << getConstantValue() << ", " << getConstantType() << ")";
};

// === APSSA OP ===
// === APSSA OP ===
// === APSSA OP ===
// === APSSA OP ===
// === APSSA OP ===

ParseResult ApSSAOp::parse(OpAsmParser &parser, OperationState &result) {
    // OpAsmParser::OperandType operand_fn;
    OpAsmParser::OperandType op_fn;
    SmallVector<Value, 4> results;
    // (<fn-arg>
    if (parser.parseLParen() || parser.parseOperand(op_fn)) { return failure(); }
    if(parser.resolveOperand(op_fn, parser.getBuilder().getNoneType(), results)) return failure();

    // ["," <arg>]
    while(succeeded(parser.parseOptionalComma())) {
        OpAsmParser::OperandType op;
        if (parser.parseOperand(op)) return failure();
        if(parser.resolveOperand(op, parser.getBuilder().getNoneType(), results)) return failure();
    }   
    
    if (parser.parseRParen()) return failure();

    result.addOperands(results);
    result.addTypes(parser.getBuilder().getNoneType());
    return success();
};

void ApSSAOp::print(OpAsmPrinter &p) {
    p << "standalone.apSSA("; 
    p.printOperand(getFn());
    for(int i = 0; i < getNumFnArguments(); ++i) {
        p << ","; p.printOperand(getFnArgument(i));
    }
    p << ")";
};

// === CASESSA OP ===
// === CASESSA OP ===
// === CASESSA OP ===
// === CASESSA OP ===
// === CASESSA OP ===
ParseResult CaseSSAOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType scrutinee;

    if(parser.parseOperand(scrutinee)) return failure();

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    if(parser.parseOptionalAttrDict(result.attributes)) return failure();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    // for (auto attr : result.attributes.getAttrs()) llvm::errs() << "-" << attr.first <<"=" << attr.second << "\n";
    // llvm::errs() << << "\n";
    for(int i = 0; i < result.attributes.getAttrs().size(); ++i) {
        llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
        Region *r = result.addRegion();
        if(parser.parseRegion(*r, {}, {})) return failure(); 
        llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    }
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    SmallVector<Value, 4> results;
    if(parser.resolveOperand(scrutinee, parser.getBuilder().getNoneType(), results)) return failure();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    result.addOperands(results);
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    result.addTypes(parser.getBuilder().getNoneType());
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    return success();

};

void CaseSSAOp::print(OpAsmPrinter &p) {
    p << "standalone.caseSSA ";
    p << "[ " << this->getOperation()->getNumOperands() << " | " << this->getNumAlts() << "] ";
    // p << this->getOperation()->getOperand(0);
    p <<  this->getScrutinee();
    p.printOptionalAttrDict(this->getAltLHSs().getValue());
    for(int i = 0; i < this->getNumAlts(); ++i) {
        p.printRegion(this->getAltRHS(i)); 
    }
};





} // namespace standalone
} // namespace mlir
