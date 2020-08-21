//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"


// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/toyc.cpp
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/Passes.h"


#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"


// conversion
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"


static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<bool> enableOptimization("optimize", llvm::cl::desc("Enable optimizations"));

//0 static llvm::cl::opt<std::string>
//0     outputFilename("o", llvm::cl::desc("Output filename"),
//0                    llvm::cl::value_desc("filename"), llvm::cl::init("-"));
//0 
//0 static llvm::cl::opt<bool> splitInputFile(
//0     "split-input-file",
//0     llvm::cl::desc("Split the input file into pieces and process each "
//0                    "chunk independently"),
//0     llvm::cl::init(false));
//0 
//0 static llvm::cl::opt<bool> verifyDiagnostics(
//0     "verify-diagnostics",
//0     llvm::cl::desc("Check that emitted diagnostics match "
//0                    "expected-* lines on the corresponding line"),
//0     llvm::cl::init(false));
//0 
//0 static llvm::cl::opt<bool> verifyPasses(
//0     "verify-each",
//0     llvm::cl::desc("Run the verifier after each transformation pass"),
//0     llvm::cl::init(true));
//0 
//0 static llvm::cl::opt<bool> allowUnregisteredDialects(
//0     "allow-unregistered-dialect",
//0     llvm::cl::desc("Allow operation with no registered dialects"),
//0     llvm::cl::init(false));
//0 
//0 static llvm::cl::opt<bool>
//0     showDialects("show-dialects",
//0                  llvm::cl::desc("Print the list of registered dialects"),
//0                  llvm::cl::init(false));
//0 

// code stolen from:
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch3/toyc.cpp
int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  mlir::registerDialect<mlir::standalone::HaskDialect>();

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "hask core compiler\n");

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }


  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }


  // TODO: why does it add a module { ... } around my hask.module { ... }?
  llvm::errs() << "\n===Module: input===\n";
  module->print(llvm::errs());
  llvm::errs() << "\n===\n";


  if (enableOptimization) {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    // pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    llvm::errs() << "===Module: running canonicalization...===\n";
    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "===Run of canonicalizer failed.===\n";
      return 4;
    }
    llvm::errs() << "==canonicalization succeeded!===\n";
  }

  llvm::errs() << "===Module " << (enableOptimization ? "(+optimization)" : "(no optimization)") << "===\n";
  module->print(llvm::errs());
  llvm::errs() << "\n===\n";


  if (enableOptimization) {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    pm.addPass(mlir::createCSEPass());
    llvm::errs() << "===Module: running CSE...===\n";
    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "===CSE failed.===\n";
      return 4;
    }
    llvm::errs() << "===CSE succeeded!===\n";

  }
  llvm::errs() << "===Module " << (enableOptimization ? "(+optimization)" : "(no optimization)") << "===\n";
  module->print(llvm::errs());
  llvm::errs() << "\n===\n";

  // Lowering code to standard (?) Do I even need to (?)
  // Can I directly generate LLVM?

  
  // lowering code to Standard/SCF
  {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::standalone::createLowerHaskToStandardPass());
    llvm::errs() << "===Module: lowering to standard+SCF...===\n";
    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "===Lowering failed.===\n";
      llvm::errs() << "===Incorrectly lowered Module to Standard+SCF:===\n";
      module->print(llvm::errs());
      llvm::errs() << "\n===\n";
      return 4;
    }
    else {
      llvm::errs() << "===Lowering succeeded!===\n";
      llvm::errs() << "===Module  lowered to Standard+SCF:===\n";
      module->print(llvm::errs());
      llvm::errs() << "\n===\n";
    }

  }



  module->print(llvm::outs()); llvm::outs().flush();
  return 0;

  
  // Lowering code to LLVM
  {
    mlir::ConversionTarget target(context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    // target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addLegalOp<mlir::standalone::HaskModuleOp, mlir::standalone::DummyFinishOp>();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::OwningRewritePatternList patterns;
    mlir::PassManager pm(&context);
    // pm.addPass(mlir::standalone::createLowerApSSAPass());

    llvm::errs() << "Module: lowering to MLIR-LLVM....";
    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "Unable to lower module\n "; return 4;
    } else {
      llvm::errs() << "Success!";
    }
  }

  // llvm::errs() << "Module " << (enableOptimization ? "(+optimization)" : "(no optimization)") << " " << "lowered: ";
  // module->print(llvm::outs());

  return 0;
}

/*
int main_old(int argc, char **argv) {

  mlir::registerDialect<mlir::standalone::HaskDialect>();
  // TODO: Register standalone passes here.

  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  mlir::MLIRContext context;
  
  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    for (mlir::Dialect *dialect : context.getRegisteredDialects()) {
      llvm::outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects))) {
    return 1;
  }
  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;
}
*/
