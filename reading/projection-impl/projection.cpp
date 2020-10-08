#include "sexpr.h"
#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

void print_indent(std::ostream &o, int indent) {
  for (int i = 0; i < indent; ++i) {
    o << "  ";
  }
}
struct NewlineIndent {
  int i;
  NewlineIndent(int i) : i(i) {}
};


std::ostream &operator<<(std::ostream &o, NewlineIndent i) {
  o << "\n";
  print_indent(o, i.i);
}
struct Expr {
  virtual void print(std::ostream &o, int indent = 0) const = 0;
};
struct FnDefinition : public Expr {
  std::string name;
  Expr *body;
  FnDefinition(std::string name, Expr *body) : name(name), body(body) {};

  void print(std::ostream &o, int indent) const {
      NewlineIndent nl(indent);
      o << nl << "(" << name << " ";
      body->print(o, indent+1);
      o << nl << ")";
  }
};

struct Variable : public Expr {
  std::string name;
  Variable(std::string name) : name(name) {};
  virtual void print(std::ostream &o, int indent) const { o << name; }
};
struct ConstNumber : public Expr {
  ll i;             
  ConstNumber(ll i) : i(i) {};
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
  Expr *lhs;
  Expr *rhsNil;
  std::pair<std::string, std::string> nameCons;
  Expr *rhsCons;


  virtual void print(std::ostream &o, int indent) const { 
      o << "(case ";
      lhs->print(o, indent);
      o << " ";
      rhsNil->print(o, indent);
      o << " ";
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
      o << fnname;
      for(int i = 0; i < args.size(); ++i) {
          args[i]->print(o, indent);
          if (i < args.size() - 1) { o << " "; }
      }
      o << ")";
  }

};

std::ostream &operator<<(std::ostream &o, Expr *e) {
    e->print(o, 0); return o;
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

Expr *parseExpr(Parser p) {
    if (std::optional<Identifier> id = p.parseOptionalIdentifier()) {
        return new Variable(id->name);
    } else if (std::optional<std::pair<Span, ll>> i = p.parseOptionalInteger()) {
        return new ConstNumber(i->second);
    }
    else {
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

            p.parseCloseRoundBracket(open);
            return c;
        } else {
            // function application
	    // TODO we assume that the function name is an identifier. It doesn't have to be
	    // it can be an expression
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

FnDefinition *parseFn(Parser p) {
    Span open = p.parseOpenRoundBracket();
    Identifier name = p.parseIdentifier();
    Expr *body = parseExpr(p);

    p.parseCloseRoundBracket(open);
    return new FnDefinition(name.name, body);
}

int main(int argc, char *argv[]) {
  assert(argc == 2 && "usage: <program-name> <path-to-input-program>");
  Parser p = parserFromPath(argv[1]);
  FnDefinition *fn = parseFn(p);
  fn->print(std::cout, 0);
  return 0;
}
