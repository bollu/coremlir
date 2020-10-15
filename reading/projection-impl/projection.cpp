#include "sexpr.h"
#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

extern "C" {
const char *__asan_default_options() { return "detect_leaks=0"; }
};

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &o, const std::pair<T1, T2> &p) {
  return o << "(" << p.first << ", " << p.second << ")";
}

// ===PARSING AST ===
// ===PARSING AST ===
// ===PARSING AST ===
// ===PARSING AST ===
// ===PARSING AST ===
// ===PARSING AST ===

void print_indent(std::ostream &o, int indent) {
  for (int i = 0; i < indent; ++i) {
    o << "  ";
  }
}
struct Newline {
  int indent;
  Newline(int indent) : indent(indent) {}
};

std::ostream &operator<<(std::ostream &o, Newline nl) {
  o << "\n";
  print_indent(o, nl.indent);
  return o;
}
struct Expr {
  virtual void print(std::ostream &o, int indent = 0) const = 0;
};
struct FnDefinition : public Expr {
  std::string name;
  // TOO get multi argument functions.
  std::vector<std::string> args;
  Expr *body;

  void print(std::ostream &o, int indent) const {
    Newline nl(indent);
    o << nl << "(" << name << " ";
    o << "[";
    for (int i = 0; i < args.size(); ++i) {
      o << args[i];
      if (i + 1 < args.size()) {
        o << " ";
      }
    }
    o << "] ";
    body->print(o, indent + 1);
    o << ")";
  }
};

struct Variable : public Expr {
  std::string name;
  Variable(std::string name) : name(name){};
  virtual void print(std::ostream &o, int indent) const { o << name; }
};
struct ConstNumber : public Expr {
  ll i;
  ConstNumber(ll i) : i(i){};
  virtual void print(std::ostream &o, int indent) const { o << i; }
};
struct IfThenElse : public Expr {
  Expr *i;
  Expr *t;
  Expr *e;

  virtual void print(std::ostream &o, int indent) const {
    o << "(if ";
    i->print(o, indent);
    o << " ";
    t->print(o, indent);
    o << " ";
    e->print(o, indent);
    o << ")";
  }
};
struct Case : public Expr {
  Expr *scrutinee;
  Expr *rhsNil;
  std::pair<std::string, std::string> nameCons;
  Expr *rhsCons;

  virtual void print(std::ostream &o, int indent) const {
    o << "(case ";
    scrutinee->print(o, indent);
    o << Newline(indent + 1);
    rhsNil->print(o, indent + 1);
    o << Newline(indent + 1);
    o << nameCons.first << " " << nameCons.second << " ";
    rhsCons->print(o, indent);
    o << ")";
  }
};

struct FnApplication : public Expr {
  std::string fnname;
  std::vector<Expr *> args;
  virtual void print(std::ostream &o, int indent) const {
    o << "(";
    o << fnname << " ";
    for (int i = 0; i < args.size(); ++i) {
      args[i]->print(o, indent);
      if (i < args.size() - 1) {
        o << " ";
      }
    }
    o << ")";
  }
};

std::ostream &operator<<(std::ostream &o, const Expr &e) {
  e.print(o, 0);
  return o;
}

Expr *parseExpr(Parser &p) {
  if (std::optional<Identifier> id = p.parseOptionalIdentifier()) {
    return new Variable(id->name);
  } else if (std::optional<std::pair<Span, ll>> i = p.parseOptionalInteger()) {
    return new ConstNumber(i->second);
  } else {
    // special forms
    Span open = p.parseOpenRoundBracket();
    Identifier fst = p.parseIdentifier();
    if (fst.name == "if") {
      IfThenElse *ite = new IfThenElse;
      ite->i = parseExpr(p);
      ite->t = parseExpr(p);
      ite->e = parseExpr(p);
      p.parseCloseRoundBracket(open);
      return ite;
    } else if (fst.name == "case") {
      Case *c = new Case;
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__ << "\n";
      c->scrutinee = parseExpr(p);
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__
                << "| scrutinee:" << *c->scrutinee << "\n";
      c->rhsNil = parseExpr(p);
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__
                << "| rhsNil: " << *c->rhsNil << "|\n";
      c->nameCons.first = p.parseIdentifier().name;
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__
                << "| nameCons.first: " << c->nameCons.first << "\n";
      c->nameCons.second = p.parseIdentifier().name;
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__
                << "| nameCons.second: " << c->nameCons.second << "\n";
      c->rhsCons = parseExpr(p);
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__
                << "| rhsCons: " << *c->rhsCons << "\n";
      p.parseCloseRoundBracket(open);
      std::cerr << __PRETTY_FUNCTION__ << ":" << __LINE__ << "\n";
      return c;
    } else {
      // function application
      // TODO we assume that the function name is an identifier. It doesn't have
      // to be it can be an expression
      FnApplication *fnap = new FnApplication;
      fnap->fnname = fst.name;
      while (!p.parseOptionalCloseRoundBracket(open)) {
        fnap->args.push_back(parseExpr(p));
      }
      return fnap;
    } // end inner else: <if>, <case>, <fn>

  } // end outer else: ident, int, (
  return nullptr;
}

FnDefinition *parseFn(Parser &p) {
  Span openBody = p.parseOpenRoundBracket();
  FnDefinition *fndefn = new FnDefinition;
  fndefn->name = p.parseIdentifier().name;

  Span openArgs = p.parseOpenSquareBracket();
  while (!p.parseOptionalCloseSquareBracket(openArgs)) {
    fndefn->args.push_back(p.parseIdentifier().name);
  }
  fndefn->body = parseExpr(p);
  p.parseCloseRoundBracket(openBody);
  return fndefn;
}

// === VALUES ===
// === VALUES ===
// === VALUES ===
// === VALUES ===
// === VALUES ===
// === VALUES ===
// === VALUES ===

// the types of values we have in our language.
struct Value {
  virtual void print(std::ostream &o) const = 0;
};

struct Bottom : public Value {
  void print(std::ostream &o) const override { o << "⊥ "; }
};

struct Halt : public Value {
  void print(std::ostream &o) const override { o << "⑂"; }
};

struct Int : public Value {
  int i;
  Int(int i){};
  void print(std::ostream &o) const override { o << i; }
};

struct Nil : public Value {
  void print(std::ostream &o) const override { o << "NIL"; }
};

struct Cons : public Value {
  Value *head, *tail;
  void print(std::ostream &o) const override {
    o << "(CONS ";
    head->print(o);
    o << " ";
    tail->print(o);
    o << ")";
  }
};

// === DOMAINS ===
// === DOMAINS ===
// === DOMAINS ===
// === DOMAINS ===
// === DOMAINS ===
// === DOMAINS ===
// === DOMAINS ===

enum class FlatProjType { FAIL, ID, ABS, STR };

std::ostream &operator<<(std::ostream &o, FlatProjType f) {
  switch (f) {
  case FlatProjType::FAIL:
    return o << "FAIL";
  case FlatProjType::ID:
    return o << "ID";
  case FlatProjType::ABS:
    return o << "ABS";
  case FlatProjType::STR:
    return o << "STR";
  }
}

struct Proj {
  virtual void print(std::ostream &o) const = 0;
};

std::ostream &operator<<(std::ostream &o, const Proj &p) {
  p.print(o);
  return o;
}

struct FlatProj : public Proj {
  FlatProjType ty;
  FlatProj(FlatProjType ty) : ty(ty) {}
  FlatProj(const FlatProj &other) : ty(other.ty){};

  bool operator==(const FlatProj &other) { return other.ty == ty; }

  bool operator<=(const FlatProj &other) {
    // everything is equal to itself.
    if (ty == other.ty) {
      return true;
    }
    // everything is greater than fail.
    if (ty == FlatProjType::FAIL) {
      return true;
    }
    // everything is less than id.
    if (other.ty == FlatProjType::ID) {
      return true;
    }
    // no other case exists.
    return false;
  }
  void print(std::ostream &o) const { o << ty; }
};

FlatProj *cupFlatProj(FlatProj *a, FlatProj *b) {
  // x U x = x
  if (a->ty == b->ty) {
    return b;
  }
  // GIVEN: this != other
  // fail U <anything> = <anything>
  if (a->ty == FlatProjType::FAIL) {
    return b;
  }
  // bot U <anything other than bot> = id
  // str U <anything other than str> = id
  // id U <anything> = id
  return new FlatProj(FlatProjType::ID);
}

struct ListCupProj;

struct ListProj : public Proj {
  virtual void print(std::ostream &o) const = 0;
};
struct NilProj : public ListProj {
  void print(std::ostream &o) const override { o << "πNIL"; }
};
struct ConsProj : public ListProj {
  FlatProj *headProj;
  ListProj *tailProj;

  ConsProj(Proj *head, Proj *tail) {
    if (FlatProj *flatHead = dynamic_cast<FlatProj *>(head)) {
      headProj = flatHead;
    } else {
      head->print(std::cerr);
      std::cerr << "|, |";
      tail->print(std::cerr);
      std::cerr << "|)\n";
      assert(false && "incorrect projection given for cons head.");
    }

    if (ListProj *listTail = dynamic_cast<ListProj *>(tail)) {
      tailProj = listTail;
    } else {
      std::cerr << "incorrect Cons(|";
      head->print(std::cerr);
      std::cerr << "|, |";
      tail->print(std::cerr);
      std::cerr << "|)\n";
      assert(false && "incorrect projection given for cons tail");
    }
  }
  void print(std::ostream &o) const override {
    o << "πCONS(";
    headProj->print(o);
    o << ", ";
    tailProj->print(o);
    o << ")";
  }
};

struct ListCupProj : public ListProj {
  NilProj *nil;
  ConsProj *cons;
  ListCupProj(NilProj *nil, ConsProj *cons) : nil(nil), cons(cons){};
  void print(std::ostream &o) const override {
    o << "(";
    nil->print(o);
    o << " U ";
    cons->print(o);
    o << ")";
  }
};

ListProj *cupListProj(ListProj *a, ListProj *b) {
  if (NilProj *na = dynamic_cast<NilProj *>(a)) {
    if (NilProj *nb = dynamic_cast<NilProj *>(b)) {
      return new NilProj();
    } else {
      ConsProj *cb = dynamic_cast<ConsProj *>(b);
      assert(cb);
      return new ListCupProj(na, cb);
    }
  } else {
    ConsProj *ca = dynamic_cast<ConsProj *>(a);
    assert(ca);

    if (NilProj *nb = dynamic_cast<NilProj *>(b)) {
      return new ListCupProj(nb, ca);
    } else {
      ConsProj *cb = dynamic_cast<ConsProj *>(b);
      assert(cb);
      return new ConsProj(cupFlatProj(ca->headProj, cb->headProj),
                          cupListProj(ca->tailProj, cb->tailProj));
    }
  }
};

// === ENVIRONMENTS ===

template <typename K, typename T> struct Env {
  void add(K k, T v) {
    assert(!env.count(k));
    env[k] = v;
  }

  void replace(K k, T v) {
    assert(env.count(k));
    env[k] = v;
  }

  T getOrNull(K name) {
    if (env.count(name)) {
      return env[name];
    }
    return nullptr;
  }

  T getOrFail(K name) {
    T e = getOrNull(name);
    if (!e) {
      std::cerr << "===ERROR: unable to find key |" << name << "|===\n";
      assert(false && "unable to find key");
    }
    return e;
  }

private:
  std::map<K, T> env;
};

Proj *unionProj(Proj *a, Proj *b) {
  if (FlatProj *fa = dynamic_cast<FlatProj *>(a)) {
    FlatProj *fb = dynamic_cast<FlatProj *>(fb);
    if (!fb) {
      std::cerr << "trying to union flat|";
      a->print(std::cerr);
      std::cerr << "| with non flat|";
      b->print(std::cerr);
      std::cerr << "|\n";
      assert(false && "union does not type check");
    }
    return cupFlatProj(fa, fb);
  } else {
    ListProj *la = dynamic_cast<ListProj *>(a);
    assert(la && "projection must be either flat or list");
    ListProj *lb = dynamic_cast<ListProj *>(b);
    assert(lb && "b must have same type as a");
    return cupListProj(la, lb);
  }
  return nullptr;
}

bool isFail(Proj *a) {
  if (FlatProj *flat = dynamic_cast<FlatProj *>(a)) {
    return flat->ty == FlatProjType::FAIL;
  }
  return false;
}

Proj *unionBangProj(Proj *a, Proj *b) {
  if (isFail(a) || isFail(b)) {
    return new FlatProj(FlatProjType::FAIL);
  }
  return unionProj(a, b);
}

// === DEMAND ANALYSIS ===
// === DEMAND ANALYSIS ===
// === DEMAND ANALYSIS ===
// === DEMAND ANALYSIS ===
// === DEMAND ANALYSIS ===
// === DEMAND ANALYSIS ===
using FnAndArg = std::pair<std::string, int>;
Proj *calculateDemandForExprAtVar(Env<FnAndArg, Proj *> env, Expr *e,
                                  std::string x, Proj *alpha);
Proj *calculateDemandForFnAtArg(Env<FnAndArg, Proj *> env, FnApplication *f,
                                int i, Proj *alpha);

// f^i(α)
Proj *calculateDemandForFnAtArg(Env<FnAndArg, Proj *> env, FnApplication *f,
                                int i, Proj *alpha) {
  // 6.2 Projection transformer
  // Definitions of `f^i` for primitive `f` appear in Section 6.7
  // f x1 ... xn = 3
  // f^i (α) = e^(x_i) (α)
  // return calculateDemandForExprAtVar(env,
  return env.getOrFail({f->fnname, i});
}

// e^x(α)
Proj *calculateDemandForExprAtVar(Env<FnAndArg, Proj *> env, Expr *e,
                                  std::string x, Proj *alpha) {

  std::cerr << "{";
  e->print(std::cerr);
  std::cerr << "}(" << x << ", ";
  alpha->print(std::cerr);
  std::cerr << ")\n";

  // x^x (a)
  if (Variable *v = dynamic_cast<Variable *>(e)) {
    if (v->name == x) {
      return alpha;
    } else {
      return new FlatProj(FlatProjType::ABS);
    }
  } else if (dynamic_cast<ConstNumber *>(e)) {
    return new FlatProj(FlatProjType::ABS);
  } else if (FnApplication *ap = dynamic_cast<FnApplication *>(e)) {
    assert(ap->args.size() >= 1);
    Proj *p = calculateDemandForFnAtArg(env, ap, 0, alpha);
    for (int i = 1; i < ap->args.size(); ++i) {
      p = unionBangProj(p, calculateDemandForFnAtArg(env, ap, i, alpha));
    }
    return p;
  } else if (IfThenElse *ite = dynamic_cast<IfThenElse *>(e)) {
    Proj *pi = calculateDemandForExprAtVar(env, ite->i, x,
                                           new FlatProj(FlatProjType::STR));
    Proj *pt = calculateDemandForExprAtVar(env, ite->t, x, alpha);
    Proj *pe = calculateDemandForExprAtVar(env, ite->e, x, alpha);
    return unionBangProj(pi, unionProj(pt, pe));
  } else if (Case *c = dynamic_cast<Case *>(e)) {
    Proj *scrutineeNil =
        calculateDemandForExprAtVar(env, c->scrutinee, x, new NilProj);
    Proj *rhsNil = calculateDemandForExprAtVar(env, c->rhsNil, x, alpha);

    // y:ys
    Proj *projy =
        calculateDemandForExprAtVar(env, c->rhsCons, c->nameCons.first, alpha);
    Proj *projys =
        calculateDemandForExprAtVar(env, c->rhsCons, c->nameCons.second, alpha);

    Proj *scrutineeCons = calculateDemandForExprAtVar(
        env, c->scrutinee, x, new ConsProj(projy, projys));
    Proj *rhsCons = calculateDemandForExprAtVar(env, c->rhsCons, x, alpha);
    return unionProj(unionBangProj(scrutineeNil, rhsNil),
                     unionBangProj(scrutineeCons, rhsCons));
  } else {
    assert(false && "unknown");
  }
}

int main(int argc, char *argv[]) {
  assert(argc == 2 && "usage: <program-name> <path-to-input-program>");
  Parser p = parserFromPath(argv[1]);
  FnDefinition *fn = parseFn(p);
  fn->print(std::cout, 0);
  std::cout << "\n";
  {
    Env<FnAndArg, Proj *> env;
    // is this correct? o_O
    env.add({"+", 0}, new FlatProj(FlatProjType::STR));
    env.add({"+", 1}, new FlatProj(FlatProjType::STR));
    const int NITER = 1;
    for (int n = 0; n < NITER; ++n) {

      for (int i = 0; i < fn->args.size(); ++i) {
        env.add({fn->name, i}, new FlatProj(FlatProjType::FAIL));
      }

      std::vector<Proj *> newps;
      for (int i = 0; i < fn->args.size(); ++i) {
        newps.push_back(calculateDemandForExprAtVar(
            env, fn->body, fn->args[i], new FlatProj(FlatProjType::STR)));
      }

      for (int i = 0; i < newps.size(); ++i) {
        env.replace({fn->name, i}, newps[i]);
      }

      std::cout << "===\n";
      std::cout << "projections after iteration |" << n + 1 << "|:\n";
      for (int i = 0; i < fn->args.size(); ++i) {
        std::cout << fn->name << "[i]"
                  << ":\n";
        std::cout << "    ";
        env.getOrFail({fn->name, i})->print(std::cout);
        std::cout << "\n";
      }
    }
  }

  return 0;
}
