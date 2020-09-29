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
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
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
#include "Hask/Runtime.h"

// conversion
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
#include "mlir/Target/LLVMIR.h"

// Execution
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<bool>
    lowerToStandard("lower-std", llvm::cl::desc("Enable lowering to standard"));
static llvm::cl::opt<bool>
    lowerToLLVM("lower-llvm", llvm::cl::desc("Enable lowering to LLVM"));
static llvm::cl::opt<bool> jit("jit",
                               llvm::cl::desc("Enable lowering to LLVM"));

// 0 static llvm::cl::opt<std::string>
// 0     outputFilename("o", llvm::cl::desc("Output filename"),
// 0                    llvm::cl::value_desc("filename"), llvm::cl::init("-"));
// 0
// 0 static llvm::cl::opt<bool> splitInputFile(
// 0     "split-input-file",
// 0     llvm::cl::desc("Split the input file into pieces and process each "
// 0                    "chunk independently"),
// 0     llvm::cl::init(false));
// 0
// 0 static llvm::cl::opt<bool> verifyDiagnostics(
// 0     "verify-diagnostics",
// 0     llvm::cl::desc("Check that emitted diagnostics match "
// 0                    "expected-* lines on the corresponding line"),
// 0     llvm::cl::init(false));
// 0
// 0 static llvm::cl::opt<bool> verifyPasses(
// 0     "verify-each",
// 0     llvm::cl::desc("Run the verifier after each transformation pass"),
// 0     llvm::cl::init(true));
// 0
// 0 static llvm::cl::opt<bool> allowUnregisteredDialects(
// 0     "allow-unregistered-dialect",
// 0     llvm::cl::desc("Allow operation with no registered dialects"),
// 0     llvm::cl::init(false));
// 0
// 0 static llvm::cl::opt<bool>
// 0     showDialects("show-dialects",
// 0                  llvm::cl::desc("Print the list of registered dialects"),
// 0                  llvm::cl::init(false));
// 0

using namespace llvm;
using namespace llvm::orc;
ExitOnError ExitOnErr;

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

  for(int i = 0; i < 3; ++i) {
      llvm::errs() << "=====Module: simplyfing [" << i << "]=====\n";

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
  }

  // Lowering code to standard (?) Do I even need to (?)
  // Can I directly generate LLVM?

  if (!lowerToStandard) {
    module->print(llvm::outs());
    return 0;
  }
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
    } else {
      llvm::errs() << "===Lowering succeeded!===\n";
      llvm::errs() << "===Module  lowered to Standard+SCF:===\n";
      module->print(llvm::errs());
      llvm::errs() << "\n===\n";
    }
  }

  if (!lowerToLLVM) {
    module->print(llvm::outs());
    return 0;
  }

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

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  llvm::errs() << "===Lowering MLIR-LLVM module to LLVM===\n";

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, *llvmContext);
  llvm::errs() << *llvmModule << "\n===\n";

  if (!jit) {
    llvm::errs() << "===Printing LLVM module to stdout...===\n";
    llvm::outs() << *llvmModule;
    llvm::errs() << "\n===\n";
    return 0;
  }

  // Lower MLIR-LLVM all the way down to "real LLVM"
  // https://github.com/llvm/llvm-project/blob/670063eb220663b5a42fd4e9bd63f51d379c9aa0/mlir/examples/toy/Ch6/toyc.cpp#L193
  const bool nativeTargetInitialized = llvm::InitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
  assert(nativeTargetInitialized == false);

  llvm::errs() << "===Executing MLIR-LLVM in JIT===\n";
  // Now we create the JIT.
  llvm::orc::LLJITBuilder JITbuilder;

  llvm::errs() << "main:  " << __LINE__ << "\n";
  ExitOnErr(JITbuilder.prepareForConstruction());

  llvm::errs() << "main:  " << __LINE__ << "\n";
  std::unique_ptr<llvm::orc::LLJIT> J = ExitOnErr(JITbuilder.create());

  llvm::errs() << "main:  " << __LINE__ << "\n";
  llvm::orc::JITDylib *JD = J->getExecutionSession().getJITDylibByName("main");

  llvm::errs() << "main:  " << __LINE__ << "\n";
  assert(JD);
  llvm::errs() << "main:  " << __LINE__ << "\n";

  llvm::JITTargetAddress putsAddr = llvm::pointerToJITTargetAddress(&puts);
  llvm::errs() << "main:  " << __LINE__ << "\n";

  llvm::orc::MangleAndInterner Mangle(J->getExecutionSession(),
                                      J->getDataLayout());

  llvm::errs() << "main:  " << __LINE__ << "\n";

  llvm::DenseMap<llvm::orc::SymbolStringPtr, llvm::JITEvaluatedSymbol>
      name2symbol;

  llvm::errs() << "main:  " << __LINE__ << "\n";
  name2symbol.insert(
      {Mangle("puts"),
       llvm::JITEvaluatedSymbol(putsAddr, llvm::JITSymbolFlags::Callable)});
  name2symbol.insert(
      {Mangle("mkClosure_capture0_args2"),
       llvm::JITEvaluatedSymbol(
           llvm::pointerToJITTargetAddress(&mkClosure_capture0_args2),
           llvm::JITSymbolFlags::Callable)});
  name2symbol.insert(
      {Mangle("mkClosure_capture0_args1"),
       llvm::JITEvaluatedSymbol(
           llvm::pointerToJITTargetAddress(&mkClosure_capture0_args1),
           llvm::JITSymbolFlags::Callable)});

  name2symbol.insert(
      {Mangle("mkClosure_capture0_args0"),
       llvm::JITEvaluatedSymbol(
           llvm::pointerToJITTargetAddress(&mkClosure_capture0_args0),
           llvm::JITSymbolFlags::Callable)});
  name2symbol.insert({Mangle("mkClosure_thunkify"),
                      llvm::JITEvaluatedSymbol(
                          llvm::pointerToJITTargetAddress(&mkClosure_thunkify),
                          llvm::JITSymbolFlags::Callable)});
  name2symbol.insert(
      {Mangle("malloc"),
       llvm::JITEvaluatedSymbol(llvm::pointerToJITTargetAddress(&malloc),
                                llvm::JITSymbolFlags::Callable)});

  name2symbol.insert(
      {Mangle("evalClosure"),
       llvm::JITEvaluatedSymbol(llvm::pointerToJITTargetAddress(&evalClosure),
                                llvm::JITSymbolFlags::Callable)});

  name2symbol.insert(
      {Mangle("extractConstructorArg"),
       llvm::JITEvaluatedSymbol(
           llvm::pointerToJITTargetAddress(&extractConstructorArg),
           llvm::JITSymbolFlags::Callable)});

  name2symbol.insert({Mangle("mkConstructor0"),
                      llvm::JITEvaluatedSymbol(
                          llvm::pointerToJITTargetAddress(&mkConstructor0),
                          llvm::JITSymbolFlags::Callable)});

  name2symbol.insert({Mangle("mkConstructor1"),
                      llvm::JITEvaluatedSymbol(
                          llvm::pointerToJITTargetAddress(&mkConstructor1),
                          llvm::JITSymbolFlags::Callable)});
  name2symbol.insert({Mangle("mkConstructor2"),
                      llvm::JITEvaluatedSymbol(
                          llvm::pointerToJITTargetAddress(&mkConstructor2),
                          llvm::JITSymbolFlags::Callable)});

  name2symbol.insert({Mangle("isConstructorTagEq"),
                      llvm::JITEvaluatedSymbol(
                          llvm::pointerToJITTargetAddress(&isConstructorTagEq),
                          llvm::JITSymbolFlags::Callable)});

  llvm::errs() << "main:  " << __LINE__ << "\n";
  ExitOnErr(JD->define(llvm::orc::absoluteSymbols(name2symbol)));
  //  llvm::orc::absoluteSymbols({
  //    { sspool.intern("puts"), llvm::pointerToJITTargetAddress(&puts)},
  //    { sspool.intern("mkClosure_capture0_args2"),
  //    llvm::pointerToJITTargetAddress(&mkClosure_capture0_args2)}
  //  }));

  llvm::errs() << "main:  " << __LINE__ << "\n";
  ExitOnErr(J->addIRModule(llvm::orc::ThreadSafeModule(
      std::move(llvmModule), std::move(llvmContext))));

  llvm::errs() << "main:  " << __LINE__ << "\n";
  // https://llvm.org/docs/ORCv2.html#how-to-add-process-and-library-symbols-to-the-jitdylibs

  // Look up the JIT'd function, cast it to a function pointer, then call it.
  auto mainfnSym = ExitOnErr(J->lookup("main"));
  void *(*mainfn)(void *) = (void *(*)(void *))mainfnSym.getAddress();

  llvm::errs() << "main:  " << __LINE__ << "\n";
  void *result = mainfn(NULL);
  llvm::errs() << "main:  " << __LINE__ << "\n";
  llvm::errs() << "(void*)main(nullptr) = " << (size_t)result << "\n";
  // answer =
  const size_t result2int = (size_t)(((Constructor *)result)->args[0]);
  llvm::errs() << "(Constructor 1*)main(nullptr) = " << result2int << "\n";

  printf("%d\n", result2int);
  return 0;
}

