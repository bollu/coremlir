#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "sexpr.h"
using namespace std;

using namespace std;
struct Expr {};
struct Variable : public Expr {
  string name;
};
struct ConstNumber : public Expr {
  int i;
};
struct FnApplication : public Expr {
  Expr *f;
  vector<Expr *> xs;
};
struct IfThenElse : public Expr {
  Expr *i;
  Expr *t;
  Expr *e;
};
struct Case : public Expr {
  Expr *lhs;
  Expr *rhsNil;
  pair<string, string> nameCons;
  Expr *rhsCons;
};


ostream &operator<<(ostream &o, Expr *e) {}

enum class FlatDomain { FAIL, ID, ABS, STR };

ostream &operator<<(ostream &o, FlatDomain f) {
    switch(f) {
        case FlatDomain::FAIL: return o << "FAIL"; 
        case FlatDomain::ID: return o << "ID"; 
        case FlatDomain::ABS: return o << "ABS"; 
        case FlatDomain::STR: return o << "STR"; 
    }
}

struct Env {
  void add(string s, Expr *e) {
    assert(!env.count(s));
    env[s] = e;
  }

  Expr *getOrNull(string name) {
    if (env.count(name)) {
      return env[name];
    }
    return nullptr;
  }

  Expr *getOrFail(string name) {
    Expr *e = getOrNull(name);
    if (!e) {
      cerr << "unable to find |" << name << "\n";
      assert(false && "unable to find key");
    }
    return e;
  }

private:
  map<string, Expr *> env;
};

/*
void eatWhitespace(FILE *f) {
    while(!feof(f)) {
        char c = fgetc(f);
        if (c == ' ' || c == '\t' || c== '\n' ) { continue; }
        // if ; then eat till newline.
        if (c == ';') {
            while(!feof(f)) {
                char d = fgetc(f);
                if (c == '\n') { break; }
            }
        }
        // not a whitespace char, put back into stream.
        ungetc(c, f);
        return;
    }
}

string nextToken(FILE *f) {
}
*/

int main(int argc, char *argv[]) {
    assert(argc == 2 && "usage: <program-name> <path-to-input-program>");
    // AST *ast = parsePath(argv[1]);
    // assert(ast && "unable to parse file!");
}
