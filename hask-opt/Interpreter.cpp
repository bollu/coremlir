#pragma once

#include "Interpreter.h"
#include <map>

using namespace mlir;
using namespace standalone;
llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const InterpStats &s) {
  o << "num_thunkify_calls(" << s.num_thunkify_calls <<  ")\n";
  o << "num_force_calls(" << s.num_force_calls << ")\n";
  o << "num_construct_calls(" << s.num_construct_calls << ")\n";
  return o;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const InterpValue &v) {
  switch (v.type) {
  case InterpValueType::Closure: {
    o << "closure(";
    o << v.closureFn() << ", ";
    for (int i = 0; i < v.closureNumArgs(); ++i) {
      o << v.closureArg(i);
      if (i + 1 < v.closureNumArgs()) {
        o << " ";
      }
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
      o << v.constructorArg(i);
      if (i + 1 < v.constructorNumArgs()) {
        o << " ";
      }
    }
    o << ")";
    break;
  }

  } // end switch
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

enum class TerminatorResultType { Branch, Return };
struct TerminatorResult {
  const TerminatorResultType type;

  TerminatorResult(Block *bbNext)
      : type(TerminatorResultType::Branch), bbNext_(bbNext) {}
  TerminatorResult(InterpValue returnValue)
      : type(TerminatorResultType::Return), returnValue_(returnValue) {}

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
      return;
    }

    if (HaskConstructOp cons = dyn_cast<HaskConstructOp>(op)) {
      stats.num_construct_calls++;
      std::vector<InterpValue> vs;
      for (int i = 0; i < cons.getNumOperands(); ++i) {
        vs.push_back(env.lookup(cons.getLoc(), cons.getOperand(i)));
      }
      env.addNew(
          cons.getResult(),
          InterpValue::constructor(cons.getDataConstructorName().str(), vs));
      return;
    }

    if (TransmuteOp transmute = dyn_cast<TransmuteOp>(op)) {
      env.addNew(transmute.getResult(),
                 env.lookup(transmute.getLoc(), transmute.getOperand()));
      return;
    }
    if (ThunkifyOp thunkify = dyn_cast<ThunkifyOp>(op)) {
      stats.num_thunkify_calls++;
      env.addNew(thunkify.getResult(),
                 InterpValue::thunkifiedValue(
                     env.lookup(thunkify.getLoc(), thunkify.getOperand())));
      return;
    }

    if (HaskRefOp ref = dyn_cast<HaskRefOp>(op)) {
      env.addNew(ref.getResult(), InterpValue::ref(ref.getRef().str()));
      return;
    }
    if (ApOp ap = dyn_cast<ApOp>(op)) {
      InterpValue fn = env.lookup(ap.getLoc(), ap.getFn());
      std::vector<InterpValue> args;
      for (int i = 0; i < ap.getNumFnArguments(); ++i) {
        args.push_back(env.lookup(ap.getLoc(), ap.getFnArgument(i)));
      }
      env.addNew(ap.getResult(), InterpValue::closure(fn, args));
      return;
    }
    if (ForceOp force = dyn_cast<ForceOp>(op)) {
      stats.num_force_calls++;
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
        std::vector<InterpValue> args(scrutinee.closureArgBegin(),
                                      scrutinee.closureArgEnd());
        env.addNew(force.getResult(), interpretFunction(func, args));
      }
      return;
    }
    if (CaseOp case_ = dyn_cast<CaseOp>(op)) {
      InterpValue scrutinee = env.lookup(case_.getLoc(), case_.getScrutinee());
      assert(scrutinee.type == InterpValueType::Constructor);

      for (int i = 0; i < case_.getNumAlts(); ++i) {
        // no match
        if (case_.getAltLHS(i).getValue().str() != scrutinee.constructorTag()) {
          continue;
        }

        // skip default case
        if (case_.getDefaultAltIndex().getValueOr(-1) == i) {
          continue;
        }

        // matched!
        env.addNew(case_.getResult(),
                   interpretRegion(case_.getAltRHS(i),
                                   scrutinee.constructorArgs(), env));
        return;
      }

      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
      assert(case_.getDefaultAltIndex() && "neither match, nor default");
      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
      env.addNew(case_,
                 interpretRegion(case_.getAltRHS(*case_.getDefaultAltIndex()),
                                 {}, env));
      return;
    }

    if (CaseIntOp caseInt = dyn_cast<CaseIntOp>(op)) {
      InterpValue scrutinee =
          env.lookup(caseInt.getLoc(), caseInt.getScrutinee());
      assert(scrutinee.type == InterpValueType::I64);

      bool matched = false;
      for (int i = 0; i < caseInt.getNumAlts(); ++i) {

        // skip default case
        if (caseInt.getDefaultAltIndex().getValueOr(-1) == i) {
          continue;
        }
        // no match
        if (caseInt.getAltLHS(i)->getInt() != scrutinee.i()) {
          continue;
        }
        // matched!
        env.addNew(caseInt, interpretRegion(caseInt.getAltRHS(i), {}, env));
        return;
      }

      assert(caseInt.getDefaultAltIndex() && "neither match, nor default");
      env.addNew(caseInt, interpretRegion(caseInt.getDefaultRHS(), {}, env));
      return;
    }

    if (DefaultCaseOp default_ = dyn_cast<DefaultCaseOp>(op)) {
      // TODO: check that this has only 1 constructor!
      InterpValue scrutinee = env.lookup(default_.getLoc(), default_.getScrutinee());
      assert(scrutinee.type == InterpValueType::Constructor);
      assert(scrutinee.constructorTag() == default_.getConstructorTag());
      assert(scrutinee.constructorNumArgs() == 1);
      env.addNew(default_.getResult(), scrutinee.constructorArg(0));
      return;
    }

    if (HaskPrimopSubOp sub = dyn_cast<HaskPrimopSubOp>(op)) {
      InterpValue a = env.lookup(sub.getLoc(), sub.getOperand(0));
      InterpValue b = env.lookup(sub.getLoc(), sub.getOperand(1));
      assert(a.type == InterpValueType::I64);
      assert(b.type == InterpValueType::I64);
      env.addNew(sub.getResult(), InterpValue::i(a.i() - b.i()));
      return;
    }

    if (HaskPrimopAddOp add = dyn_cast<HaskPrimopAddOp>(op)) {
      InterpValue a = env.lookup(add.getLoc(), add.getOperand(0));
      InterpValue b = env.lookup(add.getLoc(), add.getOperand(1));
      assert(a.type == InterpValueType::I64);
      assert(b.type == InterpValueType::I64);
      env.addNew(add.getResult(), InterpValue::i(a.i() + b.i()));
      return;
    }

    if (ApEagerOp ap = dyn_cast<ApEagerOp>(op)) {
      InterpValue fnval = env.lookup(ap.getLoc(), ap.getFn());
      assert(fnval.type == InterpValueType::Ref);
      HaskFuncOp func = module.lookupSymbol<HaskFuncOp>(fnval.ref());
      assert(func && "unable to find function");

      std::vector<InterpValue> args;
      for (int i = 0; i < ap.getNumFnArguments(); ++i) {
        args.push_back(env.lookup(ap.getLoc(), ap.getFnArgument(i)));
      }
      env.addNew(ap.getResult(), interpretFunction(func, args));
      return;
    }


    InterpreterError err(op.getLoc());
    err << "INTERPRETER ERROR: unknown operation: |" << op << "|\n";

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

  InterpValue interpretRegion(Region &r, ArrayRef<InterpValue> args, Env env) {
    Env regionEnv(env);

    if (r.getNumArguments() != args.size()) {
      InFlightDiagnostic diag = r.getContext()->getDiagEngine().emit(
          r.getLoc(), DiagnosticSeverity::Error);
      diag << "incorrect number of arguments. Given: |" << args.size() << "|\n";
      diag.report();
      exit(1);
    }

    for (int i = 0; i < args.size(); ++i) {
      env.addNew(r.getArgument(i), args[i]);
    }

    Block *bbCur = &r.front();

    while (1) {
      TerminatorResult term = interpretBlock(*bbCur, env);
      switch (term.type) {
      case TerminatorResultType::Return:
        return term.returnValue();
      case TerminatorResultType::Branch:
        bbCur = term.bbNext();
      }
    }
  }

  InterpValue interpretFunction(HaskFuncOp func, ArrayRef<InterpValue> args) {
    // functions are isolated from above; create a fresh environment.
    return interpretRegion(func.getRegion(), args, Env());
  }

  Interpreter(ModuleOp module) : module(module){};

  InterpStats getStats() const { return stats; }
private:
  ModuleOp module;
  InterpStats stats;
};


// interpret a module, and interpret the result as an integer. print it out.
std::pair<InterpValue, InterpStats> interpretModule(ModuleOp module) {
  standalone::HaskFuncOp main =
      module.lookupSymbol<standalone::HaskFuncOp>("main");
  assert(main && "unable to find main!");
  Interpreter I(module);
  InterpValue val = I.interpretFunction(main, {});
  return { val, I.getStats() };
};

