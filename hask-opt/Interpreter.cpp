#pragma once

#include "Interpreter.h"
#include <map>

using namespace mlir;
using namespace standalone;

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, InterpValue v) {
  switch (v.type) {
  case InterpValueType::Closure: {
    o << "closure(";
    o << v.closureFn() << ", ";
    for (int i = 0; i < v.closureNumArgs(); ++i) {
      o << v.closureArg(i) << " ";
    }
    o << ")";
    break;
  };
  case InterpValueType::Ref: {
    o << v.ref();
    break;
  }
  case InterpValueType::I64: {
    o << v.i();
    break;
  }
  case InterpValueType::ThunkifiedValue: {
    o << "thunk(" << v.thunkifiedValue() << ")";
    break;
  }

  case InterpValueType::Constructor: {
    o << "constructor(" << v.constructorTag() << " ";
    for (int i = 0; i < v.constructorNumArgs(); ++i) {
      o << v.constructorArg(i) << " ";
    }
    break;
  }
  };
  return o;
}

struct InterpreterError {
  InFlightDiagnostic diag;

  InterpreterError(mlir::Location loc)
      : diag(loc.getContext()->getDiagEngine().emit(
      loc, DiagnosticSeverity::Error)) {}

  ~InterpreterError() {
    diag.report();
    exit(1);
  }
};

InterpreterError &operator<<(InterpreterError &err, std::string s) {
  err.diag << s;
  return err;
}

InterpreterError &operator<<(InterpreterError &err, int i) {
  err.diag << i;
  return err;
}

InterpreterError &operator<<(InterpreterError &err, Operation &op) {
  err.diag << op;
  return err;
}

struct Env {
public:
  void addNew(mlir::Value k, InterpValue v) {
    assert(!find(k));
    insert(k, v);
  }

  InterpValue lookup(mlir::Location loc, mlir::Value k) {
    DiagnosticEngine &diagEngine = k.getContext()->getDiagEngine();
    auto it = find(k);
    if (!it) {
      InterpreterError err(loc);
      llvm::errs() << "unable to find key: |";
      k.print(llvm::errs());
      llvm::errs() << "|\n";
      err << "unable to find key";
    }
    return it.getValue();
  }

private:
  Optional<InterpValue> find(mlir::Value k) {
    for (auto it : env) {
      if (it.first == k) {
        return {it.second};
      }
    }
    return {};
  }
  void insert(mlir::Value k, InterpValue v) {
    env.push_back(std::make_pair(k, v));
  }
  // you have got to be fucking kidding me. Value doesn't have a < operator?
  std::vector<std::pair<mlir::Value, InterpValue>> env;
};

enum class TerminatorResultType {
  Branch,
  Return
};
struct TerminatorResult {
  const TerminatorResultType type;

  TerminatorResult(Block *bbNext) : type(TerminatorResultType::Branch), bbNext_(bbNext) {}
  TerminatorResult(InterpValue returnValue) : type(TerminatorResultType::Return), returnValue_(returnValue) {}

  InterpValue returnValue() {
    assert(type == TerminatorResultType::Return);
    assert(returnValue_);
    return returnValue_.getValue();

  }
  Block *bbNext() {
    assert(type == TerminatorResultType::Branch);
    return bbNext_;
  }

private:
  Block *bbNext_;
  Optional<InterpValue> returnValue_;
};

struct Interpreter {
  // dispatch the correct interpret function.
  void interpretOperation(Operation &op, Env &env) {
    if (MakeI64Op mi64 = dyn_cast<MakeI64Op>(op)) {
      env.addNew(mi64.getResult(), InterpValue::i(mi64.getValue().getInt()));
    } else if (HaskConstructOp cons = dyn_cast<HaskConstructOp>(op)) {
      std::vector<InterpValue> vs;
      for (int i = 0; i < cons.getNumOperands(); ++i) {
        vs.push_back(env.lookup(cons.getLoc(), cons.getOperand(i)));
      }
      env.addNew(cons.getResult(), InterpValue::constructor(cons.getDataConstructorName().str(), vs));
    } else if (TransmuteOp transmute = dyn_cast<TransmuteOp>(op)) {
      env.addNew(transmute.getResult(), env.lookup(transmute.getLoc(), transmute.getOperand()));
    } else if (ThunkifyOp thunkify = dyn_cast<ThunkifyOp>(op)) {
      env.addNew(thunkify.getResult(),
                 InterpValue::thunkifiedValue(env.lookup(thunkify.getLoc(), thunkify.getOperand())));
    } else if (HaskRefOp ref = dyn_cast<HaskRefOp>(op)) {
      env.addNew(ref.getResult(),
                 InterpValue::ref(ref.getRef().str()));
    } else if (ApOp ap = dyn_cast<ApOp>(op)) {
      InterpValue fn = env.lookup(ap.getLoc(), ap.getFn());
      std::vector<InterpValue> args;
      for (int i = 0; i < ap.getNumFnArguments(); ++i) {
        args.push_back(env.lookup(ap.getLoc(), ap.getFnArgument(i)));
      }
      env.addNew(ap.getResult(), InterpValue::closure(fn, args));
    } else if (ForceOp force = dyn_cast<ForceOp>(op)) {
      InterpValue scrutinee = env.lookup(force.getLoc(), force.getScrutinee());
      assert(scrutinee.type == InterpValueType::ThunkifiedValue ||
          scrutinee.type == InterpValueType::Closure);
      if (scrutinee.type == InterpValueType::ThunkifiedValue) {
        env.addNew(force.getResult(), scrutinee.thunkifiedValue());
      } else {
        assert(scrutinee.type == InterpValueType::Closure);
        InterpValue scrutineefn = scrutinee.closureFn();
        assert(scrutineefn.type == InterpValueType::Ref);
        HaskFuncOp func = module.lookupSymbol<HaskFuncOp>(scrutineefn.ref());
        assert(func && "unable to find function");
        std::vector<InterpValue> args(scrutinee.closureArgBegin(), scrutinee.closureArgEnd());
        env.addNew(force.getResult(), interpretFunction(func, args));
      }
    } else {
      InterpreterError err(op.getLoc());
      err << "unknown operation: |" << op << "|\n";
    }
  };

  TerminatorResult interpretTerminator(Operation &op, Env &env) {
    if (HaskReturnOp ret = dyn_cast<HaskReturnOp>(op)) {
      return TerminatorResult(env.lookup(ret.getLoc(), ret.getOperand()));
    } else {
      InterpreterError err(op.getLoc());
      err << "unknown terminator: |" << op << "|\n";
    }
    assert(false && "unreachable");
  }

  TerminatorResult interpretBlock(Block &block, Env &env) {
    for (Operation &op : block) {
      if (op.isKnownNonTerminator()) {
        interpretOperation(op, env);
      } else if (op.isKnownTerminator()) {
        return interpretTerminator(op, env);
      }
    }
  }

  InterpValue interpretFunction(HaskFuncOp func, ArrayRef<InterpValue> args) {
    llvm::errs() << "interpreting function |" << func.getName() << "|\n";
    DiagnosticEngine &diagEngine = func.getContext()->getDiagEngine();
    if (func.getLambda().getNumInputs() != args.size()) {
      InFlightDiagnostic diag =
          diagEngine.emit(func.getLoc(), DiagnosticSeverity::Error);
      diag << "incorrect number of arguments. Given: |" << args.size() << "|\n";
      diag.report();
      exit(1);
    }

    Env env;
    for (int i = 0; i < func.getLambda().getNumInputs(); ++i) {
      env.addNew(func.getLambda().getInput(i), args[i]);
    }

    Block *bbCur = &func.getLambda().getBody().front();
    while (1) {
      TerminatorResult term = interpretBlock(*bbCur, env);
      switch (term.type) {
      case TerminatorResultType::Return: return term.returnValue();
      case TerminatorResultType::Branch: bbCur = term.bbNext();
      }
    }
  }

  Interpreter(ModuleOp module) : module(module) {};
private:
  ModuleOp module;
};

// interpret a module, and interpret the result as an integer. print it out.
int interpretModule(ModuleOp module) {
  standalone::HaskFuncOp main =
      module.lookupSymbol<standalone::HaskFuncOp>("main");
  assert(main && "unable to find main!");
  Interpreter I(module);
  I.interpretFunction(main, {});
  return 5;
};

