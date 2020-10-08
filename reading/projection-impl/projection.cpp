#include "sexpr.h"
#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

extern "C" {
const char *__asan_default_options() { return "detect_leaks=0"; }
};

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
  o << "\n"; print_indent(o, nl.indent); return o;
}
struct Expr {
  virtual void print(std::ostream &o, int indent = 0) const = 0;
};
struct FnDefinition : public Expr {
  std::string name;
  // TOO get multi argument functions.
  std::string arg;
  Expr *body;

  void print(std::ostream &o, int indent) const {
    Newline nl(indent);
    o << nl << "(" << name << " ";
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
    o << Newline(indent+1);
    rhsNil->print(o, indent+1);
    o << Newline(indent+1);
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

enum class FlatDomain { FAIL, ID, ABS, STR };

std::ostream &operator<<(std::ostream &o, FlatDomain f) {
  switch (f) {
  case FlatDomain::FAIL:
    return o << "FAIL";
  case FlatDomain::ID:
    return o << "ID";
  case FlatDomain::ABS:
    return o << "ABS";
  case FlatDomain::STR:
    return o << "STR";
  }
}

struct Env {
  void add(std::string s, Expr *e) {
    assert(!env.count(s));
    env[s] = e;
  }

  Expr *getOrNull(std::string name) {
    if (env.count(name)) {
      return env[name];
    }
    return nullptr;
  }

  Expr *getOrFail(std::string name) {
    Expr *e = getOrNull(name);
    if (!e) {
      std::cerr << "unable to find |" << name << "\n";
      assert(false && "unable to find key");
    }
    return e;
  }

private:
  std::map<std::string, Expr *> env;
};

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
  Span open = p.parseOpenRoundBracket();
  FnDefinition *fndefn = new FnDefinition;
  fndefn->name = p.parseIdentifier().name;
  fndefn->arg = p.parseIdentifier().name;
  fndefn->body = parseExpr(p);

  p.parseCloseRoundBracket(open);
  return fndefn;
}

int main(int argc, char *argv[]) {
  assert(argc == 2 && "usage: <program-name> <path-to-input-program>");
  Parser p = parserFromPath(argv[1]);
  FnDefinition *fn = parseFn(p);
  fn->print(std::cout, 0);
  std::cout << "\n";
  return 0;
}
