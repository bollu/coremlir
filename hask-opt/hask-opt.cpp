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

extern "C" {

static int DEBUG_STACK_DEPTH = 0;
void DEBUG_INDENT() {
  for (int i = 0; i < DEBUG_STACK_DEPTH; ++i) {
    fputs("  ⋮", stderr);
  }
}

#define DEBUG_LOG                                                              \
  if (1) {                                                                     \
    \ 
    DEBUG_INDENT();                                                            \
    fprintf(stderr, "%s ", __FUNCTION__);                                      \
  }
void DEBUG_PUSH_STACK() { DEBUG_STACK_DEPTH++; }
void DEBUG_POP_STACK() { DEBUG_STACK_DEPTH--; }

static const int MAX_CLOSURE_ARGS = 10;
struct Closure {
  int n;
  void *fn;
  void *args[MAX_CLOSURE_ARGS];
};

char *getPronouncableNum(size_t N) {
  const char *cs = "bcdfghjklmnpqrstvwxzy";
  const char *vs = "aeiou";

  size_t ncs = strlen(cs);
  size_t nvs = strlen(vs);

  char buf[1024];
  char *out = buf;
  int i = 0;
  while (N > 0) {
    const size_t icur = N % (ncs * nvs);
    *out++ = cs[icur % ncs];
    *out++ = vs[(icur / ncs) % nvs];
    N /= ncs * nvs;
    if (N > 0 && !(++i % 2)) {
      *out++ = '-';
    }
  }
  *out = 0;
  return strdup(buf);
};

char *getPronouncablePtr(void *N) { return getPronouncableNum((size_t)N); }

void *__attribute__((used))
mkClosure_capture0_args2(void *fn, void *a, void *b) {
  Closure *data = (Closure *)malloc(sizeof(Closure));
  DEBUG_LOG;
  fprintf(stderr, "(%p:%s, %p:%s, %p:%s) -> %10p:%s\n", fn,
          getPronouncablePtr(fn), a, getPronouncablePtr(a), b,
          getPronouncablePtr(b), data, getPronouncablePtr(data));
  data->n = 2;
  data->fn = fn;
  data->args[0] = a;
  data->args[1] = b;
  return (void *)data;
}

void *__attribute__((used)) mkClosure_capture0_args1(void *fn, void *a) {
  Closure *data = (Closure *)malloc(sizeof(Closure));
  DEBUG_LOG;
  fprintf(stderr, "(%p:%s, %p:%s) -> %10p:%s\n", fn, getPronouncablePtr(fn), a,
          getPronouncablePtr(a), data, getPronouncablePtr(data));
  data->n = 1;
  data->fn = fn;
  data->args[0] = a;
  return (void *)data;
}

void *__attribute__((used)) mkClosure_capture0_args0(void *fn) {
  Closure *data = (Closure *)malloc(sizeof(Closure));
  DEBUG_LOG;
  fprintf(stderr, "(%p:%s) -> %p:%s\n", fn, getPronouncablePtr(fn), data,
          getPronouncablePtr(data));
  data->n = 0;
  data->fn = fn;
  return (void *)data;
}

void *identity(void *v) { return v; }

void *__attribute__((used)) mkClosure_thunkify(void *v) {
  Closure *data = (Closure *)malloc(sizeof(Closure));
  DEBUG_LOG;
  fprintf(stderr, "(%p) -> %p\n", v, data);
  data->n = 1;
  data->fn = (void *)identity;
  data->args[0] = v;
  return (void *)data;
}

typedef void *(*FnZeroArgs)();
typedef void *(*FnOneArg)(void *);
typedef void *(*FnTwoArgs)(void *, void *);

void *__attribute__((used)) evalClosure(void *closure_voidptr) {
  DEBUG_LOG;
  fprintf(stderr, "(%p:%s)\n", closure_voidptr,
          getPronouncablePtr(closure_voidptr));
  DEBUG_PUSH_STACK();
  Closure *c = (Closure *)closure_voidptr;
  assert(c->n >= 0 && c->n <= 3);
  void *ret = NULL;
  if (c->n == 0) {
    FnZeroArgs f = (FnZeroArgs)(c->fn);
    ret = f();
  } else if (c->n == 1) {
    FnOneArg f = (FnOneArg)(c->fn);
    ret = f(c->args[0]);
  } else if (c->n == 2) {
    FnTwoArgs f = (FnTwoArgs)(c->fn);
    ret = f(c->args[0], c->args[1]);
  } else {
    assert(false && "unhandled function arity");
  }
  DEBUG_POP_STACK();
  DEBUG_INDENT();
  fprintf(stderr, "=>%10p:%s\n", ret, getPronouncablePtr(ret));
  return ret;
};

static const int MAX_CONSTRUCTOR_ARGS = 2;
struct Constructor {
  const char *tag; // inefficient!
  int n;
  void *args[MAX_CONSTRUCTOR_ARGS];
};

void *__attribute__((used)) mkConstructor0(const char *tag) {
  Constructor *c = (Constructor *)malloc(sizeof(Constructor));
  DEBUG_LOG;
  fprintf(stderr, "(%s) -> %p:%s\n", tag, c, getPronouncablePtr(c));
  c->n = 0;
  c->tag = tag;
  return c;
};

void *__attribute__((used)) mkConstructor1(const char *tag, void *a) {
  Constructor *c = (Constructor *)malloc(sizeof(Constructor));
  DEBUG_LOG;
  fprintf(stderr, "(%s, %p) -> %p:%s\n", tag, a, c, getPronouncablePtr(c));
  c->tag = tag;
  c->n = 1;
  c->args[0] = a;
  return c;
};

void *__attribute__((used)) mkConstructor2(const char *tag, void *a, void *b) {
  Constructor *c = (Constructor *)malloc(sizeof(Constructor));
  DEBUG_LOG;
  fprintf(stderr, "(%s, %p, %p) -> %p\n", tag, a, b, c);
  c->tag = tag;
  c->n = 2;
  c->args[0] = a;
  c->args[1] = b;
  return c;
};

void *extractConstructorArg(void *cptr, int i) {
  Constructor *c = (Constructor *)cptr;
  void *v = c->args[i];
  assert(i < c->n);
  DEBUG_LOG;
  fprintf(stderr, "%s %d -> %p:%s\n", cptr, i, v, getPronouncablePtr(v));
  return v;
}

bool isConstructorTagEq(const void *cptr, const char *tag) {
  Constructor *c = (Constructor *)cptr;
  const bool eq = !strcmp(c->tag, tag);
  DEBUG_LOG;
  fprintf(stderr, "(%p:%s, %s) -> %d\n", cptr, c->tag, tag, eq);
  return eq;
}
} // end extern C

namespace Example {
using namespace llvm;
using namespace llvm::orc;
ExitOnError ExitOnErr;

ThreadSafeModule createDemoModule() {
  auto Context = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("test", *Context);

  // Create the add1 function entry and insert this entry into module M.  The
  // function will have a return type of "int" and take an argument of "int".
  Function *Add1F =
      Function::Create(FunctionType::get(Type::getInt32Ty(*Context),
                                         {Type::getInt32Ty(*Context)}, false),
                       Function::ExternalLinkage, "add1", M.get());

  // Add a basic block to the function. As before, it automatically inserts
  // because of the last argument.
  BasicBlock *BB = BasicBlock::Create(*Context, "EntryBlock", Add1F);

  // Create a basic block builder with default parameters.  The builder will
  // automatically append instructions to the basic block `BB'.
  IRBuilder<> builder(BB);

  // Get pointers to the constant `1'.
  Value *One = builder.getInt32(1);

  // Get pointers to the integer argument of the add1 function...
  assert(Add1F->arg_begin() != Add1F->arg_end()); // Make sure there's an arg
  Argument *ArgX = &*Add1F->arg_begin();          // Get the arg
  ArgX->setName("AnArg"); // Give it a nice symbolic name for fun.

  // Create the add instruction, inserting it into the end of BB.
  Value *Add = builder.CreateAdd(One, ArgX);

  // Create the return instruction and add it to the basic block
  builder.CreateRet(Add);

  return ThreadSafeModule(std::move(M), std::move(Context));
}

} // namespace Example
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
  Example::ExitOnErr(JITbuilder.prepareForConstruction());

  llvm::errs() << "main:  " << __LINE__ << "\n";
  std::unique_ptr<llvm::orc::LLJIT> J = Example::ExitOnErr(JITbuilder.create());

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
  Example::ExitOnErr(JD->define(llvm::orc::absoluteSymbols(name2symbol)));
  //  llvm::orc::absoluteSymbols({
  //    { sspool.intern("puts"), llvm::pointerToJITTargetAddress(&puts)},
  //    { sspool.intern("mkClosure_capture0_args2"),
  //    llvm::pointerToJITTargetAddress(&mkClosure_capture0_args2)}
  //  }));

  llvm::errs() << "main:  " << __LINE__ << "\n";
  Example::ExitOnErr(J->addIRModule(llvm::orc::ThreadSafeModule(
      std::move(llvmModule), std::move(llvmContext))));

  llvm::errs() << "main:  " << __LINE__ << "\n";
  // https://llvm.org/docs/ORCv2.html#how-to-add-process-and-library-symbols-to-the-jitdylibs

  // Look up the JIT'd function, cast it to a function pointer, then call it.
  auto mainfnSym = Example::ExitOnErr(J->lookup("main"));
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

