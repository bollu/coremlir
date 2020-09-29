#pragma once
#include "./Runtime.h"
#include "Hask/HaskDialect.h"
#include "Hask/HaskOps.h"


// interpret a module, and interpret the result as an integer. print it out.
int interpretModule(mlir::ModuleOp module);

enum class InterpValueType {
  I64,
  Closure,
  Constructor,
  Ref,
  ThunkifiedValue,
};
struct InterpValue {
  InterpValueType type;

  int i() {
    assert(type == InterpValueType::I64);
    return i_;
  }

  std::string ref() {
    assert(type == InterpValueType::Ref);
    return s_;
  }

  static InterpValue i(int i) {
    InterpValue v(InterpValueType::I64);
    v.i_ = i;
    return v;
  }

  static InterpValue thunkifiedValue(InterpValue v) {
    InterpValue thunk(InterpValueType::ThunkifiedValue);
    thunk.vs_.push_back(v);
    return thunk;
  }

  InterpValue thunkifiedValue() const {
    assert(type == InterpValueType::ThunkifiedValue);
    return vs_[0];
  }


  static InterpValue ref(std::string tag) {
      InterpValue ref(InterpValueType::Ref);
      ref.s_ = tag;
      return ref;
  }
  static InterpValue closure(InterpValue fn, std::vector<InterpValue> args) {
      InterpValue closure(InterpValueType::Closure);
      closure.vs_.push_back(fn);
      closure.vs_.insert(closure.vs_.end(), args.begin(), args.end());
      return closure;
  }

  InterpValue closureFn() const {
    assert(type == InterpValueType::Closure);
    return vs_[0];
  }

  int closureNumArgs() const {
    assert(type == InterpValueType::Closure);
    return vs_.size() - 1;
  }

  InterpValue closureArg(int i) const {
    assert(type == InterpValueType::Closure);
    assert(i >= 0);
    assert(i < closureNumArgs());
    return vs_[1 + i];
  }

  std::vector<InterpValue>::const_iterator closureArgBegin() const {
    assert(type == InterpValueType::Closure);
    return vs_.begin() + 1;
  }

  std::vector<InterpValue>::const_iterator closureArgEnd() const {
    assert(type == InterpValueType::Closure);
    return vs_.end();
  }

  static InterpValue constructor(std::string tag, std::vector<InterpValue> vs) {
    InterpValue cons(InterpValueType::Constructor);
    cons.s_ = tag;
    cons.vs_ = vs;
    return cons;
  }


  int constructorNumArgs() const {
    assert(type == InterpValueType::Constructor);
    return vs_.size();
  }

  InterpValue constructorArg(int i) const {
    assert(type == InterpValueType::Constructor);
    assert(i >= 0);
    assert(i < constructorNumArgs());
    return vs_[i];
  }

  std::vector<InterpValue>::const_iterator constructorArgBegin() const {
    assert(type == InterpValueType::Constructor);
    return vs_.begin();
  }

  std::vector<InterpValue>::const_iterator constructorArgEnd() const {
    assert(type == InterpValueType::Constructor);
    return vs_.end();
  }

  std::string constructorTag() {
    assert(type == InterpValueType::Constructor);
    return s_;
  }



  int i_;
  std::vector<InterpValue> vs_;
  std::string s_;
private:
  InterpValue(InterpValueType type) : type(type){};
};

llvm::raw_ostream &operator << (llvm::raw_ostream &o, InterpValue v);

