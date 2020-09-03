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
#include "mlir/Target/LLVMIR.h"


// Execution
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"


static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<bool> lowerToStandard("lower-std", llvm::cl::desc("Enable lowering to standard"));
static llvm::cl::opt<bool> lowerToLLVM("lower-llvm", llvm::cl::desc("Enable lowering to LLVM"));

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


  {
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

  module->print(llvm::errs());
  llvm::errs() << "\n===\n";

  {

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

    llvm::errs() << "===Module===\n";
    module->print(llvm::errs());
    llvm::errs() << "\n===\n";
  }



  // Lowering code to standard (?) Do I even need to (?)
  // Can I directly generate LLVM?

  if (!lowerToStandard) { module->print(llvm::outs()); return 0; }
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


  if (!lowerToLLVM) { module->print(llvm::outs()); return 0; }
  // Lowering code to MLIR-LLVM
  {

    mlir::PassManager pm(&context);
    pm.addPass(mlir::standalone::createLowerHaskStandardToLLVMPass());

    llvm::errs() << "===Module: lowering to MLIR-LLVM...===\n";
    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "===Unable to lower module to MLIR-LLVM===\n";
      module->print(llvm::errs());
      llvm::errs() << "\n===\n";
      return 1;
    } else {
      llvm::errs() << "===Success!===\n";
      module->print(llvm::errs());
      llvm::errs() << "\n===\n";
    }
  }


  llvm::errs() << "===Printing MLIR-LLVM module to stdout...===\n";
  module->print(llvm::outs()); llvm::outs().flush();


  // Lower MLIR-LLVM all the way down to "real LLVM"
  // https://github.com/llvm/llvm-project/blob/670063eb220663b5a42fd4e9bd63f51d379c9aa0/mlir/examples/toy/Ch6/toyc.cpp#L193
  llvm::LLVMContext llvmContext;
  llvm::errs() << "===Lowering MLIR-LLVM module to LLVM===\n";
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, llvmContext);
  llvm::errs() << *llvmModule << "\n===\n";

  return 0;
}

