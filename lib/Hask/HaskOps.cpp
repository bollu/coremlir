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
#include "mlir/Dialect/SCF/SCF.h"

// pattern matching
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

// dilect lowering
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "hask-ops"
#include "llvm/Support/Debug.h"


namespace mlir {
namespace standalone {
#define GET_OP_CLASSES
#include "Hask/HaskOps.cpp.inc"





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


ParseResult MakeI64Op::parse(OpAsmParser &parser, OperationState &result) {
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

void MakeI64Op::print(OpAsmPrinter &p) {
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

// === APSSA OP ===
// === APSSA OP ===
// === APSSA OP ===
// === APSSA OP ===
// === APSSA OP ===

ParseResult parseUntypedValueOrSymbol(OpAsmParser &parser, OperationState &result,
                               const char *symbolAttrName) {
  StringAttr name;
  if (parser.parseOptionalSymbolName(name,
        ::mlir::SymbolTable::getSymbolAttrName(),
        result.attributes)) {
    OpAsmParser::OperandType op;
    if(parser.parseOperand(op) ||
           parser.resolveOperand(op, parser.getBuilder().getType<UntypedType>(),
                                 result.operands)) {
      return failure();
    }
  }
  return success();
}

ParseResult ApSSAOp::parse(OpAsmParser &parser, OperationState &result) {
    // OpAsmParser::OperandType operand_fn;
    OpAsmParser::OperandType op_fn;
    SmallVector<Value, 4> results;
    // (<fn-arg>
    if (parser.parseLParen()) { return failure(); }

    if(parser.parseOperand(op_fn) ||
           parser.resolveOperand(op_fn, parser.getBuilder().getType<UntypedType>(),
                                 result.operands)) {
      return failure();
    }

    // ["," <arg>]
    while(succeeded(parser.parseOptionalComma())) {
        OpAsmParser::OperandType op;
        if (parser.parseOperand(op)) return failure();
        if(parser.resolveOperand(op, parser.getBuilder().getType<UntypedType>(), result.operands)) return failure();
    }

    //)
    if (parser.parseRParen()) return failure();

    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
};

Value ApSSAOp::getFn() { return this->getOperation()->getOperand(0); };

int ApSSAOp::getNumFnArguments() {
    if(getFn()) { return this->getOperation()->getNumOperands() - 1; }
    return this->getOperation()->getNumOperands();
};

Value ApSSAOp::getFnArgument(int i) {
    return this->getOperation()->getOperand(i + 1);
};

SmallVector<Value, 4> ApSSAOp::getFnArguments() {
    SmallVector<Value, 4> args;
    for(int i = 0; i < getNumFnArguments(); ++i) {
      args.push_back(getFnArgument(i));
    }
    return args;
};


void ApSSAOp::print(OpAsmPrinter &p) {
    p << "hask.apSSA(";

    // TODO: propose actually strongly typing this?This is just sick.
    for(int i = 0; i < this->getOperation()->getNumOperands(); ++i) {
        if (i > 0) { p << ", "; }
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

llvm::Optional<int> CaseSSAOp::getDefaultAltIndex() {
    for(int i = 0; i < getNumAlts(); ++i) {
        Attribute ai = this->getAltLHS(i);
        // TODO,HACK! We are assuming that any string will be DEFAULT
        // Of course, this is completely borked.
         StringAttr sai = ai.dyn_cast<StringAttr>();
         if (sai) { return i; }
    }
    return llvm::Optional<int>();
}

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

ParseResult HaskRefOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
    StringAttr nameAttr;
    if(parser.parseLParen() ||
            parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                result.attributes) ||
            parser.parseRParen()) { return failure(); }
    result.addTypes(parser.getBuilder().getType<UntypedType>());
    return success();
}

llvm::StringRef HaskRefOp::getRef() {
    return getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName()).getValue();
}

void HaskRefOp::print(OpAsmPrinter &p) {
    p << getOperationName() << "(";
    p.printSymbolName(this->getRef());
    p << ")";
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


// === ADT OP ===
// === ADT OP ===
// === ADT OP ===
// === ADT OP ===
// === ADT OP ===
//

ParseResult HaskADTOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType scrutinee;

    SmallVector<Value, 4> results;
    Attribute name;
    Attribute constructors;
    if(parser.parseAttribute(name, "name", result.attributes)) { return failure(); }
    if(parser.parseAttribute(name, "constructors", result.attributes)) { return failure(); }

//    result.addAttribute("name", name);
//    result.addAttribute("constructors", constructors);
//    if(parser.parseAttribute(constructors)) { return failure(); }

    llvm::errs() << "ADT: " << name << "\n" << "cons: " << constructors << "\n";
    return success();

};


void HaskADTOp::print(OpAsmPrinter &p) {
    p << getOperationName();
    p << "{ ";
   for (const std::pair<Identifier,Attribute> &it : this->getAttrs()) {
      p << it.first << ":" << it.second << " ";
   }
   p << " }";
};


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
  ///   %f2 = apSSA(%f1, v1) // ap1
  ///   %out = apSSA(%f2, v2) // ap2
  /// into:
  ///   %out = apSSA(%f1, v1, v2) 
  /// ie rewrite:
  mlir::LogicalResult
  matchAndRewrite(ApSSAOp ap2,
                  mlir::PatternRewriter &rewriter) const override {


    Value f2 = ap2.getFn();
    if (!f2) { return failure(); }
    ApSSAOp ap1 = f2.getDefiningOp<ApSSAOp>();
    if (!ap1) {  return failure(); }

    SmallVector<Value, 4> args;

    for(int i = 0; i < ap1.getNumFnArguments(); ++i) {
        args.push_back(ap1.getFnArgument(i));
    }
    for(int i  = 0; i < ap2.getNumFnArguments(); ++i) {
        args.push_back(ap2.getFnArgument(i));
    }



    llvm::errs() << "\n====\n";
    llvm::errs() << "running on:\n- ap1|" << ap1 << 
        " |\n- ap2" << "|" << ap2 << "|\n";

    llvm::errs() << "-[";
    for(int i = 0; i < args.size(); ++i) {
        llvm::errs() << args[i] << ", ";
    }
    llvm::errs() << "]\n";
    Value calledVal = ap1.getFn();
    llvm::errs() << "-calledVal: " << calledVal << "\n";
    // ApSSAOp replacement = rewriter.create<ApSSAOp>(ap2.getLoc(), calledVal, args);
    // llvm::errs() << "-replacement: " << replacement << "|" << __LINE__ << "\n";
    rewriter.replaceOpWithNewOp<ApSSAOp>(ap2, calledVal, args);
    llvm::errs() << "\n====\n";

    /*
    Value fn_b = out.getFn();
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

    if (Value fncall = fn_b_op.getFn()) {
        rewriter.replaceOpWithNewOp<ApSSAOp>(out, fncall, args);
    } else {
        assert(fn_b_op.fnSymbolName() && "we must have symbol if we don't have Value");
        rewriter.replaceOpWithNewOp<ApSSAOp>(out, fn_b_op.fnSymbolName(), args);
    }
    */
    return success();
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
            fn.getFuncName().str(),
            FunctionType::get({rewriter.getI64Type()}, {rewriter.getI64Type()}, rewriter.getContext()));
    rewriter.inlineRegionBefore(lam.getBody(), stdFunc.getBody(), stdFunc.end());

    // Block &funcEntry = stdFunc.getBody().getBlocks().front();
    // funcEntry.args_begin()->setType(rewriter.getType<UntypedType>());
    rewriter.insert(stdFunc);

    llvm::errs() << "- stdFunc: " << stdFunc << "\n";
    rewriter.eraseOp(fn);

    // llvm::errs() << "- stdFunc.getParentOp()\n: " << *stdFunc.getParentOp() << "\n";
    return success();
  }
};

class CaseSSAOpConversionPattern : public ConversionPattern {
public:
    explicit CaseSSAOpConversionPattern(MLIRContext *context)
            : ConversionPattern(standalone::CaseSSAOp::getOperationName(), 1, context) { }

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
      auto caseop = cast<CaseSSAOp>(op);
      const int default_ix = *caseop.getDefaultAltIndex();
      assert(caseop.getDefaultAltIndex() && "expected default case");


      // delete the use of the case.
      // TODO: Change the IR so that a case doesn't have a use.
      // TODO: This is so kludgy :(
      for(Operation *user : op->getUsers()) { rewriter.eraseOp(user); }

        Value scrutinee = caseop.getScrutinee();
        scrutinee.setType(rewriter.getI64Type());

        rewriter.setInsertionPoint(caseop);
        // TODO: get block of current caseop?
        for(int i = 0; i < caseop.getNumAlts(); ++i) {
            if (i == default_ix) { continue; }
            mlir::ConstantOp lhsConstant =
                rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(),
                                                  caseop.getAltLHS(i));
            // Type result, IntegerAttr predicate, Value lhs, Value rhs
            mlir::CmpIOp scrutinee_eq_val =
                rewriter.create<mlir::CmpIOp>(rewriter.getUnknownLoc(),
                                              rewriter.getI1Type(),
                                              mlir::CmpIPredicate::eq,
                                              lhsConstant,
                                              caseop.getScrutinee());
            Block *prevBB = rewriter.getInsertionBlock();
            SmallVector<Type, 1> BBArgs = { rewriter.getI64Type()};
            Block *thenBB = rewriter.createBlock(caseop.getParentRegion(), {},
                                                 rewriter.getI64Type());
            Block *elseBB = rewriter.createBlock(caseop.getParentRegion(), {},
                                                 rewriter.getI64Type());

            rewriter.setInsertionPointToEnd(prevBB);
            rewriter.create<mlir::CondBranchOp>(rewriter.getUnknownLoc(),
                                                scrutinee_eq_val,
                                                thenBB, caseop.getScrutinee(),
                                                elseBB, caseop.getScrutinee());

            rewriter.mergeBlocks(&caseop.getAltRHS(i).front(), thenBB, caseop.getScrutinee());
            rewriter.setInsertionPointToStart(elseBB);
        }

        /*
        assert(false);
        rewriter.mergeBlocks(&caseop.getAltRHS(default_ix).front(), elseBB, caseop.getScrutinee());
        */

        rewriter.mergeBlocks(&caseop.getAltRHS(default_ix).front(),
                             rewriter.getInsertionBlock(),
                             caseop.getScrutinee());
        rewriter.eraseOp(caseop);


        return success();
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

// Humongous hack: we run a sort of "type inference" algorithm, where at the
// call-site, we convert from a !hask.untyped to a concrete (say, int32)
// type. We bail with an error if we are unable to replace the type.
void unifyOpTypeWithType(Value src, Type dstty) {
    if (src.getType() == dstty) { return; }
    if (src.getType().isa<UntypedType>()) {
       src.setType(dstty);
    } else {
        assert(false && "unable to unify types!");
    }
}

class ApSSAConversionPattern : public ConversionPattern {
public:
  explicit ApSSAConversionPattern(MLIRContext *context)
      : ConversionPattern(ApSSAOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running ApSSAConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    ApSSAOp ap = cast<ApSSAOp>(op);
    if (HaskRefOp ref = (ap.getFn().getDefiningOp<HaskRefOp>())) {
        if (ref.getRef() == "-#") {
            assert(ap.getNumFnArguments() == 2 && "expected fully saturated function call!");
            rewriter.replaceOpWithNewOp<SubIOp>(ap, rewriter.getI64Type(), ap.getFnArgument(0), ap.getFnArgument(1));

            for(int i = 0; i < 2; ++i) {
                unifyOpTypeWithType(ap.getFnArgument(i), rewriter.getI64Type());
            }

            return success();
        }
        else if (ref.getRef() == "+#") {
            assert(ap.getNumFnArguments() == 2 && "expected fully saturated function call!");
            rewriter.replaceOpWithNewOp<AddIOp>(ap, rewriter.getI64Type(), ap.getFnArgument(0), ap.getFnArgument(1));
            for(int i = 0; i < 2; ++i) {
                unifyOpTypeWithType(ap.getFnArgument(i), rewriter.getI64Type());
            }
            return success();
        }
        else {
            llvm::StringRef fnName = ref.getRef();
            llvm::errs() << "function symbol name: |" << fnName << "|\n";
            // HACK: hardcoding type
            const Type retty = rewriter.getI64Type();
            rewriter.replaceOpWithNewOp<CallOp>(ap,
                    fnName, retty, ap.getFnArguments());

            return success();
        }
    } // end ap(HaskRefOp(...), ...)

    else { assert(false && "unhandled ApSSA type"); }
    return failure();
  } // end matchAndRewrite
};

class MakeI64OpConversionPattern : public ConversionPattern {
public:
  explicit MakeI64OpConversionPattern(MLIRContext *context)
      : ConversionPattern(MakeI64Op::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "running MakeI64OpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    MakeI64Op makei64 = cast<MakeI64Op>(op);
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op,rewriter.getI64Type(), makei64.getValue());
    return success();
  }
};

class ForceOpConversionPattern : public ConversionPattern {
public:
  explicit ForceOpConversionPattern(MLIRContext *context)
      : ConversionPattern(ForceOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ForceOp force = cast<ForceOp>(op);
    llvm::errs() << "running ForceOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    rewriter.replaceOp(op, force.getScrutinee());
    return success();
  }
};

class HaskReturnOpConversionPattern : public ConversionPattern {
public:
  explicit HaskReturnOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskReturnOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    HaskReturnOp ret = cast<HaskReturnOp>(op);
    llvm::errs() << "running HaskReturnOpConversionPattern on: " << op->getName() << " | " << op->getLoc() << "\n";
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(ret, ret.getValue());
    return success();
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
    target.addLegalDialect<mlir::scf::SCFDialect>();

    // Why do I need this? Isn't adding StandardOpsDialect enough?
    target.addLegalOp<FuncOp>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();

    target.addIllegalDialect<standalone::HaskDialect>();
    target.addLegalOp<MakeDataConstructorOp>();
    target.addLegalOp<HaskRefOp>();
    target.addLegalOp<HaskADTOp>();

    // target.addLegalOp<LambdaSSAOp>();
    // target.addLegalOp<CaseSSAOp>();
    // target.addLegalOp<MakeI64Op>();
    // target.addLegalOp<ApSSAOp>();
    // target.addLegalOp<ForceOp>();
    // target.addLegalOp<HaskReturnOp>();
    OwningRewritePatternList patterns;
    patterns.insert<HaskFuncOpConversionPattern>(&getContext());
    patterns.insert<CaseSSAOpConversionPattern>(&getContext());
    patterns.insert<LambdaSSAOpConversionPattern>(&getContext());
    patterns.insert<ApSSAConversionPattern>(&getContext());
    patterns.insert<MakeI64OpConversionPattern>(&getContext());
    patterns.insert<ForceOpConversionPattern>(&getContext()); 
    patterns.insert<HaskReturnOpConversionPattern>(&getContext());

    llvm::errs() << "===Enabling Debugging...===\n";
    ::llvm::DebugFlag = true;

  
  if (failed(applyPartialConversion(this->getOperation(), target, patterns))) {
      llvm::errs() << "===Hask -> Std lowering failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
  }

  ::llvm::DebugFlag = false;
  return;

}

std::unique_ptr<mlir::Pass> createLowerHaskToStandardPass() {
  return std::make_unique<LowerHaskToStandardPass>();
}


// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===
// === LowerHaskStandardToLLVMPass ===


// this is run in a second phase, so we delete data constructors only
// after we are done processing data constructors.
class MakeDataConstructorOpConversionPattern : public ConversionPattern {
public:
  explicit MakeDataConstructorOpConversionPattern(MLIRContext *context)
      : ConversionPattern(MakeDataConstructorOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
      rewriter.eraseOp(op);
    return success();
  }
};


// this is run in a second phase, so we refs only
// after we are done processing data constructors.
class HaskRefOpConversionPattern : public ConversionPattern {
public:
  explicit HaskRefOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskRefOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
      rewriter.eraseOp(op);
    return success();
  }
};

class HaskADTOpConversionPattern : public ConversionPattern {
public:
  explicit HaskADTOpConversionPattern(MLIRContext *context)
      : ConversionPattern(HaskADTOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

namespace {
struct LowerHaskStandardToLLVMPass : public Pass {
    LowerHaskStandardToLLVMPass() : Pass(mlir::TypeID::get<LowerHaskStandardToLLVMPass>()) {};
    StringRef getName() const override { return "LowerHaskToStandardPass"; }

    std::unique_ptr<Pass> clonePass() const override {
        auto newInst = std::make_unique<LowerHaskStandardToLLVMPass>(*static_cast<const LowerHaskStandardToLLVMPass *>(this));
        newInst->copyOptionValuesFrom(this);
        return newInst;
    }

    void runOnOperation() {

      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<mlir::LLVM::LLVMDialect>();
      target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
      target.addIllegalDialect<HaskDialect>();
      mlir::LLVMTypeConverter typeConverter(&getContext());
      mlir::OwningRewritePatternList patterns;
      patterns.insert<MakeDataConstructorOpConversionPattern>(&getContext());
      patterns.insert<HaskRefOpConversionPattern>(&getContext());
      patterns.insert<HaskADTOpConversionPattern>(&getContext());

      mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
      if (failed(mlir::applyFullConversion(getOperation(), target, patterns))) {
          llvm::errs() << "===Hask+Std -> LLVM lowering failed===\n";
          getOperation()->print(llvm::errs());
          llvm::errs() << "\n===\n";
          signalPassFailure();
      };

    };

};
} // end anonymous namespace.


std::unique_ptr<mlir::Pass> createLowerHaskStandardToLLVMPass() {
  return std::make_unique<LowerHaskStandardToLLVMPass>();
}

} // namespace standalone
} // namespace mlir
