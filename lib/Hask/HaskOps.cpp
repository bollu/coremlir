//===- HaskOps.cpp - Hask dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hask/HaskOps.h"
#include "Hask/HaskDialect.h"
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

// Standard dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

// pattern matching
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

// dilect lowering
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "hask-ops"
#include "llvm/Support/Debug.h"


namespace mlir {
namespace standalone {
#define GET_OP_CLASSES
#include "Hask/HaskOps.cpp.inc"


// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
// === LAMBDA OP ===
ParseResult LambdaOp::parse(OpAsmParser &parser, OperationState &result) {
    SmallVector<mlir::OpAsmParser::OperandType, 4> regionArgs;
    if (parser.parseRegionArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::Paren)) return failure();


    for(int i = 0; i < regionArgs.size(); ++ i) {
    	llvm::errs() << "-" << __FUNCTION__ << ":" << __LINE__ << ":" << regionArgs[i].name <<"\n";
    }

    Region *r = result.addRegion();
    return parser.parseRegion(*r, regionArgs, {parser.getBuilder().getType<UntypedType>()});
}

void LambdaOp::print(OpAsmPrinter &p) {
    p << "hask.lambda";
    p << "(";
        for(int i = 0; i < this->getNumInputs(); ++i) {
            p << this->getInput(i);
            if (i < this->getNumInputs() - 1) { p << ","; }
        } 
    p << ")";
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
    p << "hask.case";
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
    p << "hask.ap("; p.printRegion(getFn(), /*blockArgs=*/false);
    
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


ParseResult HaskReturnOp::parse(OpAsmParser &parser, OperationState &result) {
    mlir::OpAsmParser::OperandType i;
    if (parser.parseLParen() || parser.parseOperand(i) || parser.parseRParen())
        return failure();
    SmallVector<Value, 1> vi;
    parser.resolveOperand(i, parser.getBuilder().getType<UntypedType>(), vi);
    result.addOperands(vi);
    return success();
};

void HaskReturnOp::print(OpAsmPrinter &p) {
    p << "hask.return(" << getInput() << ")";
};



// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===
// === MakeI32 OP ===


ParseResult MakeI32Op::parse(OpAsmParser &parser, OperationState &result) {
    // mlir::OpAsmParser::OperandType i;
    Attribute attr;
    
    if (parser.parseLParen() || parser.parseAttribute(attr, "value", result.attributes) || parser.parseRParen())
        return failure();
    // result.addAttribute("value", attr);
    //SmallVector<Value, 1> vi;
    //parser.resolveOperand(i, parser.getBuilder().getIntegerType(32), vi);
    
    // TODO: convert this to emitParserError, etc.
    // assert (attr.getType().isSignedInteger() && "expected parameter to make_i32 to be integer");

    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

void MakeI32Op::print(OpAsmPrinter &p) {
    p << "hask.make_i32(" << getValue() << ")";
};

// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===
// === MakeDataConstructor OP ===

ParseResult MakeDataConstructorOp::parse(OpAsmParser &parser, OperationState &result) {
    // parser.parseAttribute(, parser.getBuilder().getStringAttr )
    // if(parser.parseLess()) return failure();

    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
        return failure();
    }
    // if(parser.parseAttribute(attr, "name", result.attributes)) {
    //     assert(false && "unable to parse attribute!");  return failure();
    // }
    // if(parser.parseGreater()) return failure();
    // result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

llvm::StringRef MakeDataConstructorOp::getDataConstructorName() {
    return getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName()).getValue();
}

void MakeDataConstructorOp::print(OpAsmPrinter &p) {
    p << getOperationName() << " ";
    p.printSymbolName(getDataConstructorName());
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
    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

void TopLevelBindingOp::print(OpAsmPrinter &p) {
    p << getOperationName(); p.printRegion(getBody(), /*printEntry=*/false);
};

// === HaskModule OP ===
// === HaskModule OP ===
// === HaskModule OP ===
// === HaskModule OP ===
// === HaskModule OP ===

ParseResult HaskModuleOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType body;
    if(parser.parseRegion(*result.addRegion(), {}, {})) return failure();
    // result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

void HaskModuleOp::print(OpAsmPrinter &p) {
    p << getOperationName(); p.printRegion(getBody(), /*printEntry=*/false);
};

// === DummyFinish OP ===
// === DummyFinish OP ===
// === DummyFinish OP ===
// === DummyFinish OP ===
// === DummyFinish OP ===

ParseResult DummyFinishOp::parse(OpAsmParser &parser, OperationState &result) {
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
    result.addTypes(parser.getBuilder().getType<UntypedType>());
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
    if (parser.parseLParen()) { return failure(); }
    
    StringAttr nameAttr;
    // parser.parseAttribute
    if (succeeded(parser.parseOptionalSymbolName(nameAttr,
        ::mlir::SymbolTable::getSymbolAttrName(),
        result.attributes))) {
        SymbolRefAttr attr = SymbolRefAttr::get(nameAttr.getValue(), parser.getBuilder().getContext());
        

        // success; nothing to do.
    }
    else if(succeeded(parser.parseOperand(op_fn))) { 
        if(parser.resolveOperand(op_fn, parser.getBuilder().getType<UntypedType>(), results)) {
            return failure();
        }
    }
    else { return failure(); }

    // ["," <arg>]
    while(succeeded(parser.parseOptionalComma())) {
        OpAsmParser::OperandType op;
        if (parser.parseOperand(op)) return failure();
        if(parser.resolveOperand(op, parser.getBuilder().getType<UntypedType>(), results)) return failure();
    }   
    
    if (parser.parseRParen()) return failure();

    result.addOperands(results);
    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

FlatSymbolRefAttr ApSSAOp::fnSymbolicAttr() {
    StringAttr attr_name = this->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName());
    if (attr_name) { return SymbolRefAttr::get(attr_name.getValue(), this->getContext()); }
    return FlatSymbolRefAttr();
};

Value ApSSAOp::fnValue() {
    if (this->fnSymbolicAttr()) { return Value(); }
    return this->getOperation()->getOperand(0);
};

int ApSSAOp::getNumFnArguments() {
    if(fnValue()) { return this->getOperation()->getNumOperands() - 1; }
    return this->getOperation()->getNumOperands();
};

Value ApSSAOp::getFnArgument(int i) {
    if(fnValue()) { return this->getOperation()->getOperand(i + 1); }
    return this->getOperation()->getOperand(i);
};

void ApSSAOp::print(OpAsmPrinter &p) {
    p << "hask.apSSA(";

    // TODO: propose actually strongly typing this?This is just sick.
    SymbolRefAttr fnSymbolic = fnSymbolicAttr();
    if (fnSymbolic) { p << fnSymbolic; }
    for(int i = 0; i < this->getOperation()->getNumOperands(); ++i) {
        if (i > 0 || (i == 0 && fnSymbolic)) { p << ", "; }
        p.printOperand(this->getOperation()->getOperand(i));
    }
    p << ")";
};

void ApSSAOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value fn, SmallVectorImpl<Value> &params) {
    state.addOperands(fn);
    state.addOperands(params);
    state.addTypes(builder.getType<UntypedType>());
};

void ApSSAOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    FlatSymbolRefAttr fn, SmallVectorImpl<Value> &params) {
    state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), fn);
    state.addOperands(params);
    state.addTypes(builder.getType<UntypedType>());

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
    // if(parser.parseOptionalAttrDict(result.attributes)) return failure();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    // "[" altname "->" region "]"
    int nattr = 0;
    while (succeeded(parser.parseOptionalLSquare())) {
        Attribute alt_type_attr;
        const std::string attrname = "alt" + std::to_string(nattr);
        parser.parseAttribute(alt_type_attr, attrname, result.attributes);
        nattr++;
        parser.parseArrow();
        Region *r = result.addRegion();
        if(parser.parseRegion(*r, {}, {})) return failure(); 
        parser.parseRSquare();
    }

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    SmallVector<Value, 4> results;
    if(parser.resolveOperand(scrutinee, parser.getBuilder().getType<UntypedType>(), results)) return failure();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    result.addOperands(results);
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    result.addTypes(parser.getBuilder().getType<UntypedType>());
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    return success();

};

void CaseSSAOp::print(OpAsmPrinter &p) {
    p << "hask.caseSSA ";
    // p << "[ " << this->getOperation()->getNumOperands() << " | " << this->getNumAlts() << "] ";
    // p << this->getOperation()->getOperand(0);
    p <<  this->getScrutinee();
    // p.printOptionalAttrDict(this->getAltLHSs().getValue());
    for(int i = 0; i < this->getNumAlts(); ++i) {
        p <<" [" << this->getAltLHS(i) <<" -> ";
        p.printRegion(this->getAltRHS(i)); 
        p << "]\n";
    }
};

// === LAMBDASSA OP ===
// === LAMBDASSA OP ===
// === LAMBDASSA OP ===
// === LAMBDASSA OP ===
// === LAMBDASSA OP ===

ParseResult LambdaSSAOp::parse(OpAsmParser &parser, OperationState &result) {
    SmallVector<mlir::OpAsmParser::OperandType, 4> regionArgs;
    if (parser.parseRegionArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::Paren)) return failure();


    for(int i = 0; i < regionArgs.size(); ++ i) {
    	llvm::errs() << "-" << __FUNCTION__ << ":" << __LINE__ << ":" << regionArgs[i].name <<"\n";
    }

    Region *r = result.addRegion();
    if(parser.parseRegion(*r, regionArgs, {parser.getBuilder().getType<UntypedType>()})) return failure();
    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
}

void LambdaSSAOp::print(OpAsmPrinter &p) {
    p << "hask.lambdaSSA";
    p << "(";
        for(int i = 0; i < this->getNumInputs(); ++i) {
            p << this->getInput(i);
            if (i < this->getNumInputs() - 1) { p << ","; }
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

ParseResult RecursiveRefOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  result.addTypes(parser.getBuilder().getType<UntypedType>());
  return success();

  // magic++, wtf++;
  // RecursiveRefOp::ensureTerminator(*body, parser.getBuilder(), result.location);
}
void RecursiveRefOp::print(OpAsmPrinter &p) {
    p << getOperationName(); p.printRegion(getRegion(), /*printEntry=*/false);
};


// === MakeString OP ===
// === MakeString OP ===
// === MakeString OP ===
// === MakeString OP ===
// === MakeString OP ===


ParseResult MakeStringOp::parse(OpAsmParser &parser, OperationState &result) {
    // mlir::OpAsmParser::OperandType i;
    Attribute attr;
    
    if (parser.parseLParen() || parser.parseAttribute(attr, "value", result.attributes) || parser.parseRParen())
        return failure();
    // result.addAttribute("value", attr);
    //SmallVector<Value, 1> vi;
    //parser.resolveOperand(i, parser.getBuilder().getIntegerType(32), vi);
    
    // TODO: check if attr is string.

    result.addTypes(parser.getBuilder().getType<UntypedType>());
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
    // TODO: how to parse an int32?

    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
        return failure();
    };

     // Parse the optional function body.
    auto *body = result.addRegion();
    return parser.parseOptionalRegion( *body, {}, ArrayRef<Type>() );
};

void HaskFuncOp::print(OpAsmPrinter &p) {
    p << "hask.func" << ' ';
    p.printSymbolName(getFuncName());
    // Print the body if this is not an external function.
    Region &body = this->getRegion();
    if (!body.empty()) {
        p.printRegion(body, /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/true);
    }
}

llvm::StringRef HaskFuncOp::getFuncName() {
    return getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName()).getValue();
}


LambdaSSAOp HaskFuncOp::getLambda() {
    Region &region = this->getRegion();
    // TODO: put this in a `verify` block.
    assert(region.getBlocks().size() == 1 && "func has more than one BB");
    Block &entry = region.front();
    HaskReturnOp ret = cast<HaskReturnOp>(entry.getTerminator());
    Value retval = ret.getValue();
    return cast<LambdaSSAOp>(retval.getDefiningOp());
}

// === FORCE OP ===
// === FORCE OP ===
// === FORCE OP ===
// === FORCE OP ===
// === FORCE OP ===

ParseResult ForceOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType scrutinee;
    
    if(parser.parseLParen() || parser.parseOperand(scrutinee) || parser.parseRParen()) {
        return failure();
    }

    SmallVector<Value, 4> results;
    if(parser.resolveOperand(scrutinee, parser.getBuilder().getType<UntypedType>(), results)) return failure();
    result.addOperands(results);
    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();

};

void ForceOp::print(OpAsmPrinter &p) {
    p << "hask.force(" << this->getScrutinee() << ")";
};

// === Copy OP ===
// === Copy OP ===
// === Copy OP ===
// === Copy OP ===
// === Copy OP ===

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType scrutinee;
    
    if(parser.parseLParen() || parser.parseOperand(scrutinee) || parser.parseRParen()) {
        return failure();
    }

    SmallVector<Value, 4> results;
    if(parser.resolveOperand(scrutinee, parser.getBuilder().getType<UntypedType>(), results)) return failure();
    result.addOperands(results);
    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();

};


void CopyOp::print(OpAsmPrinter &p) {
    p << "hask.copy(" << this->getScrutinee() << ")";
};


// === REF OP ===
// === REF OP ===
// === REF OP ===
// === REF OP ===
// === REF OP ===
ParseResult HaskRefOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType scrutinee;

    StringAttr nameAttr;
    if (parser.parseLParen() ||
        parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                result.attributes) ||
        parser.parseRParen()) {
        return failure();
    };

    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
}
StringRef HaskRefOp::getArgumentSymbolName() {
    return getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName()).getValue();
}
void HaskRefOp::print(OpAsmPrinter &p) {
    p << getOperationName() << " (";
    p.printSymbolName(getArgumentSymbolName()); 
    p << ")";
}

// ==REWRITES==

// =SATURATE AP= 
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/toyc.cpp
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/mlir/Dialect.cpp
struct RewriteUncurryApplication : public mlir::OpRewritePattern<ApSSAOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  RewriteUncurryApplication(mlir::MLIRContext *context)
      : OpRewritePattern<ApSSAOp>(context, /*benefit=*/1) {
      }

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  /// rewrite:
  ///   %fn_b = apSSA(%fn_a, var_a)
  ///   %out = apSSA(%fn_b, var_b) <- this is our instruction
  /// into:
  ///   %out = apSSA(%fn_a, var_a, var_b)
  /// ie rewrite:
  ///   %out = (fn_a var_a) var_b
  /// into:
  ///   %out fn_a (var_a, var_b)
  mlir::LogicalResult
  matchAndRewrite(ApSSAOp out,
                  mlir::PatternRewriter &rewriter) const override {

    
    Value fn_b = out.fnValue();
    if (!fn_b) {
        llvm::errs() << "no match: |" << out << "|\n";
        return failure();
    }

    ApSSAOp fn_b_op = fn_b.getDefiningOp<ApSSAOp>();
    if (!fn_b_op) { 
        llvm::errs() << "no match: |" << out << "|\n";
        return failure();
    }

    llvm::errs() << "found suitable op: |"  << out << " | fn_as_apOp: " << fn_b_op << "\n";
    SmallVector<Value, 4> args;
    // we have all args.
    for(int i = 0; i < fn_b_op.getNumFnArguments(); ++i) {
        args.push_back(fn_b_op.getFnArgument(i));
    }
    for(int i  = 0; i < out.getNumFnArguments(); ++i) {
        args.push_back(out.getFnArgument(i));
    }

    if (Value fncall = fn_b_op.fnValue()) {
        rewriter.replaceOpWithNewOp<ApSSAOp>(out, fncall, args);
    } else {
        assert(fn_b_op.fnSymbolicAttr() && "we must have symbol if we don't have Value");
        rewriter.replaceOpWithNewOp<ApSSAOp>(out, fn_b_op.fnSymbolicAttr(), args);
    }
    return failure();
  }
};

void ApSSAOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<RewriteUncurryApplication>(context);
}

// === LOWERING ===
// === LOWERING ===

class HaskToStdTypeConverter : public mlir::TypeConverter {
    using TypeConverter::TypeConverter;

};

// http://localhost:8000/structanonymous__namespace_02ConvertStandardToLLVM_8cpp_03_1_1FuncOpConversion.html#a9043f45e0e37eb828942ff867c4fe38d
class HaskFuncOpConversionPattern : public ConversionPattern {
public:
  explicit HaskFuncOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::HaskFuncOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Now lower the [LambdaSSAOp + HaskFuncOp together]
    // TODO: deal with free floating lambads. This LambdaSSAOp is going to
    // become a top-level function. Other lambdas will become toplevel functions with synthetic names.
    llvm::errs() << "running HaskFuncOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    auto fn = cast<HaskFuncOp>(op);
    LambdaSSAOp lam = fn.getLambda();

    /*
    SmallVector<NamedAttribute, 4> attrs;
    // TODO: HACK! types are hardcoded :)
    rewriter.setInsertionPointAfter(fn);
    llvm::errs() << "- stdFunc: " << stdFunc << "\n";
    llvm::errs() << "- fn.getParentOp():\n\n" << *fn.getParentOp() << "\n";

    rewriter.inlineRegionBefore(lam.getBody(), stdFunc.getBody(), stdFunc.end());
    llvm::errs() << "- stdFunc: " << stdFunc << "\n";
    // llvm::errs() << "- fn.getParentOp():\n\n" << *fn.getParentOp() << "\n";

    // TODO: Tell the rewriter to convert the region signature.
    // rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
    */
    FuncOp stdFunc = ::mlir::FuncOp::create(fn.getLoc(),
            fn.getFuncName().str() + "_lowered",
            FunctionType::get({rewriter.getType<UntypedType>()}, {}, rewriter.getContext()));
    rewriter.inlineRegionBefore(lam.getBody(), stdFunc.getBody(), stdFunc.end());
    rewriter.insert(stdFunc);

    llvm::errs() << "- stdFunc: " << stdFunc << "\n";
    rewriter.eraseOp(fn);

    // llvm::errs() << "- stdFunc.getParentOp()\n: " << *stdFunc.getParentOp() << "\n";
    return success();
  }
};

class CaseOpConversionPattern : public ConversionPattern {
public:
    explicit CaseOpConversionPattern(MLIRContext *context)
            : ConversionPattern(standalone::CaseOp::getOperationName(), 1, context) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        assert(false && "entered CaseOp");
        llvm::errs() << "running CaseOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";

        return failure();
    }
};

class LambdaSSAOpConversionPattern : public ConversionPattern {
public:
  explicit LambdaSSAOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::LambdaSSAOp::getOperationName(), 1, context) {}

  LogicalResult 
      matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto lam = cast<LambdaSSAOp>(op);
    assert(lam);
    llvm::errs() << "running LambdaSSAOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";

    return failure();
  }
};


class ApSSAConversionPattern : public ConversionPattern {
public:
  explicit ApSSAConversionPattern(MLIRContext *context)
      : ConversionPattern(ApSSAOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running ApSSAConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    return failure();
  }
};

class HaskModuleOpConversionPattern : public ConversionPattern {
public:
  explicit HaskModuleOpConversionPattern(MLIRContext *context)
      : ConversionPattern(standalone::HaskModuleOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "running HaskModuleOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    // mlir::ModuleOp mod = rewriter.create<mlir::ModuleOp>(op->getLoc());
    // llvm::errs() << "\n||op->numResults: " 
    //      << op->getNumResults() 
    //      << "[op]: " 
    //      << *op << "\n||mod->numResults: " 
    //      << mod.getOperation()->getNumResults() << "\n";
    return failure();
  }
};

class MakeI32OpConversionPattern : public ConversionPattern {
public:
  explicit MakeI32OpConversionPattern(MLIRContext *context)
      : ConversionPattern(MakeI32Op::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running MakeI32OpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    return failure();
  }
};

class ForceOpConversionPattern : public ConversionPattern {
public:
  explicit ForceOpConversionPattern(MLIRContext *context)
      : ConversionPattern(ForceOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running ForceOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    return failure();
  }
};

class HaskReturnOpConversionPattern : public ConversionPattern {
public:
  explicit HaskReturnOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskReturnOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running HaskReturnOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    return failure();
  }
};




// === LowerHaskToStandardPass === 
// === LowerHaskToStandardPass === 
// === LowerHaskToStandardPass === 
// === LowerHaskToStandardPass === 

namespace {
struct LowerHaskToStandardPass
    : public Pass {
    LowerHaskToStandardPass() : Pass(mlir::TypeID::get<LowerHaskToStandardPass>()) {};
    void runOnOperation();
    StringRef getName() const override { return "LowerHaskToStandardPass"; }

    std::unique_ptr<Pass> clonePass() const override {
        auto newInst = std::make_unique<LowerHaskToStandardPass>(*static_cast<const LowerHaskToStandardPass *>(this));
        newInst->copyOptionValuesFrom(this);
        return newInst;
    }
};
} // end anonymous namespace.

// http://localhost:8000/ConvertSPIRVToLLVMPass_8cpp_source.html#l00031
void LowerHaskToStandardPass::runOnOperation() {
    ConversionTarget target(getContext());
    // do I not need a pointer to the dialect? I am so confused :(
    HaskToStdTypeConverter converter();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    // Why do I need this? Isn't adding StandardOpsDialect enough?
    target.addLegalOp<FuncOp>();
    target.addIllegalDialect<standalone::HaskDialect>();
    target.addLegalOp<HaskModuleOp>();
    target.addLegalOp<MakeDataConstructorOp>();
    target.addLegalOp<LambdaSSAOp>();
    //target.addLegalOp<CaseSSAOp>();
    target.addLegalOp<MakeI32Op>();
    target.addLegalOp<ApSSAOp>();
    target.addLegalOp<ForceOp>();
    target.addLegalOp<HaskReturnOp>();
    target.addLegalOp<DummyFinishOp>();
    target.addLegalOp<HaskRefOp>();

    OwningRewritePatternList patterns;
    patterns.insert<HaskFuncOpConversionPattern>(&getContext());
    patterns.insert<CaseOpConversionPattern>(&getContext());
    patterns.insert<LambdaSSAOpConversionPattern>(&getContext());
    patterns.insert<ApSSAConversionPattern>(&getContext());
    patterns.insert<HaskModuleOpConversionPattern>(&getContext()); 
    patterns.insert<MakeI32OpConversionPattern>(&getContext()); 
    patterns.insert<ForceOpConversionPattern>(&getContext()); 
    patterns.insert<HaskReturnOpConversionPattern>(&getContext()); 

    llvm::errs() << "debugging? " << ::llvm::DebugFlag << "\n";
    LLVM_DEBUG({ assert(false && "llvm debug exists"); });
    ::llvm::DebugFlag = true; 

  
  if (failed(applyPartialConversion(this->getOperation(), target, patterns))) {
      llvm::errs() << "===Hask -> Std lowering failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
  }
  return;

  //   auto function = getFunction();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  //0 ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  //0 target.addLegalDialect<LeanDialect, StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  // target.addIllegalDialect<LeanDialect>();
  // target.addLegalOp<PrintUnboxedIntOp>();
  //0 target.addIllegalOp<PrintUnboxedIntOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  //0 OwningRewritePatternList patterns;
  //0 patterns.insert<PrintOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  //0 if (failed(applyPartialConversion(getFunction(), target, patterns))) {
  //0   llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
  //0   llvm::errs() << "fn\nvvvv\n";
  //0   getFunction().dump() ;
  //0   llvm::errs() << "\n^^^^^\n";
  //0   signalPassFailure();
  //0 }
}

std::unique_ptr<mlir::Pass> createLowerHaskToStandardPass() {
  return std::make_unique<LowerHaskToStandardPass>();
}

// === LowerHaskSSAOpToStandard === 
// === LowerHaskSSAOpToStandard === 
// === LowerHaskSSAOpToStandard === 
// === LowerHaskSSAOpToStandard === 
// === LowerHaskSSAOpToStandard === 

namespace {
struct LowerHaskSSAOpToStandardPass
    : public PassWrapper<LowerHaskSSAOpToStandardPass, OperationPass<ApSSAOp>> {
  void runOnOperation();
};
} // end anonymous namespace.



void LowerHaskSSAOpToStandardPass::runOnOperation() {
    ConversionTarget target(getContext());
  OwningRewritePatternList patterns;
  patterns.insert<ApSSAConversionPattern>(&getContext());
  if (failed(applyPartialConversion(this->getOperation(), target, patterns))) {
    llvm::errs() << "==" << __FUNCTION__ << " failed==\n";
    getOperation().dump() ;
    llvm::errs() << "====\n";
    signalPassFailure();
  }

  return;
}

std::unique_ptr<mlir::Pass> createHaskSSAOpLowering() {
      return std::make_unique<LowerHaskSSAOpToStandardPass>();

}


} // namespace standalone
} // namespace mlir
