#pragma once
#include <assert.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <optional>
#include <stdarg.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define GIVE
#define TAKE
#define KEEP
using namespace std;
using ll = long long;

struct Span;

// L for location
struct Loc {
  const char *filename;
  ll si, line, col;

  Loc(const char *filename, ll si, ll line, ll col)
      : filename(filename), si(si), line(line), col(col){};

  Loc nextline() const { return Loc(filename, si + 1, line + 1, 1); }

  Loc nextc(char c) const {
    if (c == '\n') {
      return nextline();
    } else {
      return nextcol();
    }
  }

  Loc prev(char c) const {
    if (c == '\n') {
      assert(false && "don't know how to walk back newline");
    } else {
      return prevcol();
    }
  }

  Loc prev(const char *s) const {
    Loc l = *this;
    for (int i = strlen(s) - 1; i >= 0; --i) {
      l = l.prev(s[i]);
    }
    return l;
  }

  bool operator==(const Loc &other) const {
    return si == other.si && line == other.line && col == other.col;
  }

  bool operator!=(const Loc &other) const { return !(*this == other); }

  // move the current location to the location Next, returning
  // the distance spanned.
  Span moveMut(Loc next);

private:
  Loc nextcol() const { return Loc(filename, si + 1, line, col + 1); }

  Loc prevcol() const {
    assert(col - 1 >= 1);
    return Loc(filename, si - 1, line, col - 1);
  }
};

ostream &operator<<(ostream &o, const Loc &l) {
  return cout << ":" << l.line << ":" << l.col;
}

// half open [...)
// substr := str[span.begin...span.end-1];
struct Span {
  Loc begin, end;

  Span(Loc begin, Loc end) : begin(begin), end(end) {
    assert(end.si >= begin.si);
    assert(!strcmp(begin.filename, end.filename));
  };

  ll nchars() const { return end.si - begin.si; }
};

Span Loc::moveMut(Loc next) {
  assert(next.si >= this->si);
  Span s(*this, next);
  *this = next;
  return s;
};

ostream &operator<<(ostream &o, const Span &s) {
  return cout << s.begin << " - " << s.end;
}

// TODO: upgrade this to take a space, not just a location.
void vprintfspan(Span span, const char *raw_input, const char *fmt,
                 va_list args) {
  char *outstr = nullptr;
  vasprintf(&outstr, fmt, args);
  assert(outstr);
  cerr << "===\n";
  cerr << span.begin << ":" << span.end << "\n";

  cerr << "===\n";
  cerr << span << "\t" << outstr << "\n";
  for (ll i = span.begin.si; i < span.end.si; ++i) {
    cerr << raw_input[i];
  }
  cerr << "\n===\n";
}

void printfspan(Span span, const char *raw_input, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintfspan(span, raw_input, fmt, args);
  va_end(args);
}

void vprintferr(Loc loc, const char *raw_input, const char *fmt, va_list args) {
  char *outstr = nullptr;
  vasprintf(&outstr, fmt, args);
  assert(outstr);

  const int LINELEN = 80;
  const int CONTEXTLEN = 25;
  char line_buf[2 * LINELEN];
  char pointer_buf[2 * LINELEN];

  // find the previous newline character, or some number of characters back.
  // Keep a one window lookahead.
  ll nchars_back = 0;
  for (; loc.si - nchars_back >= 1 &&
         raw_input[loc.si - (nchars_back + 1)] != '\n';
       nchars_back++) {
  }

  int outix = 0;
  if (nchars_back > CONTEXTLEN) {
    nchars_back = CONTEXTLEN;
    for (int i = 0; i < 3; ++i, ++outix) {
      line_buf[outix] = '.';
      pointer_buf[outix] = ' ';
    }
  }

  {
    int inix = loc.si - nchars_back;
    for (; inix - loc.si <= CONTEXTLEN; ++inix, ++outix) {
      if (raw_input[inix] == '\0') {
        break;
      }
      if (raw_input[inix] == '\n') {
        break;
      }
      line_buf[outix] = raw_input[inix];
      pointer_buf[outix] = (inix == loc.si) ? '^' : ' ';
    }

    if (raw_input[inix] != '\0' && raw_input[inix] != '\n') {
      for (int i = 0; i < 3; ++i, ++outix) {
        line_buf[outix] = '.';
        pointer_buf[outix] = ' ';
      }
    }
    line_buf[outix] = pointer_buf[outix] = '\0';
  }

  cerr << "\n==\n"
       << outstr << "\n"
       << loc.filename << loc << "\n"
       << line_buf << "\n"
       << pointer_buf << "\n==\n";
  free(outstr);
}

void printferr(Loc loc, const char *raw_input, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintferr(loc, raw_input, fmt, args);
  va_end(args);
}

bool isWhitespace(char c) { return c == ' ' || c == '\n' || c == '\t'; }
bool isReservedSigil(char c) {
  return c == '(' || c == ')' || c == '{' || c == '}' || c == ',' || c == ';' ||
         c == '[' || c == ']' || c == ':';
}

struct Error {
  string errmsg;
  Loc loc;

  Error(Loc loc, string errmsg) : errmsg(errmsg), loc(loc){};
};

struct Identifier {
  const Span span;
  const string name;
  Identifier(const Identifier &other) = default;
  Identifier(Span span, string name) : span(span), name(name){};

  Identifier operator=(const Identifier &other) { return Identifier(other); }

  void print(ostream &f) const { f << name; }
};

struct Parser {
  Parser(const char *filename, const char *raw)
      : s(string(filename)), l(filename, 0, 1, 1){};

  void parseOpenCurly() { parseSigil(string("{")); }
  void parseCloseCurly() { parseSigil(string("}")); }
  void parseFatArrow() { parseSigil(string("=>")); }
  bool parseOptionalCloseCurly() {
    return bool(parseOptionalSigil(string("}")));
  }
  void parseOpenRoundBracket() { parseSigil(string("(")); }
  bool parseOptionalOpenRoundBracket() {
    return bool(parseOptionalSigil(string("(")));
  }
  bool parseCloseRoundBracket() {
    return bool(parseOptionalSigil(string(")")));
  }
  bool parseOptionalCloseRoundBracket() {
    return bool(parseOptionalSigil(string(")")));
  }
  void parseColon() { parseSigil(string(":")); }
  bool parseOptionalComma() { return bool(parseOptionalSigil(string(","))); }
  void parseComma() { parseSigil(string(",")); }
  void parseSemicolon() { parseSigil(string(";")); }
  bool parseOptionalSemicolon() {
    return bool(parseOptionalSigil(string(";")));
  }
  void parseThinArrow() { parseSigil(string("->")); }

  pair<Span, ll> parseInteger() {
    optional<pair<Span, ll>> out = parseOptionalInteger();
    if (!out) {
      this->addErr(Error(l, string("unble to find integer")));
      exit(1);
    }
    return *out;
  }

  // [-][0-9]+
  optional<pair<Span, ll>> parseOptionalInteger() {
    eatWhitespace();
    bool negate = false;
    optional<char> ccur; // peeking character
    Loc lcur = l;

    ccur = this->at(lcur);
    if (!ccur) {
      return {};
    }
    if (*ccur == '-') {
      negate = true;
      lcur = lcur.nextc(*ccur);
    }

    ll number = 0;
    while (1) {
      ccur = this->at(lcur);
      if (!ccur) {
        break;
      }
      if (!isdigit(*ccur)) {
        break;
      }
      number = number * 10 + (*ccur - '0');
      lcur = lcur.nextc(*ccur);
    }
    Span span = l.moveMut(lcur);
    if (span.nchars() == 0) {
      return {};
    }
    if (negate) {
      number *= -1;
    };

    return {{span, number}};
  }

  Span parseSigil(const string sigil) {
    optional<Span> span = parseOptionalSigil(sigil);
    if (span) {
      return *span;
    }

    addErr(Error(l, "expected sigil: |" + sigil + "|"));
    exit(1);
  }

  // difference is that a sigil needs no whitespace after it, unlike
  // a keyword.
  optional<Span> parseOptionalSigil(const string sigil) {
    cerr << __FUNCTION__ << "|" << sigil.c_str() << "|\n";
    optional<char> ccur;
    eatWhitespace();
    Loc lcur = l;
    // <sigil>

    for (ll i = 0; i < sigil.size(); ++i) {
      ccur = this->at(lcur);
      if (!ccur || *ccur != sigil[i]) {
        return {};
      }
      lcur = lcur.nextc(*ccur);
    }

    Span span = l.moveMut(lcur);
    return span;
  }

  Identifier parseIdentifier() {
    optional<Identifier> ms = parseOptionalIdentifier();
    if (ms.has_value()) {
      return *ms;
    }
    addErr(Error(l, string("expected identifier")));
    exit(1);
  }

  optional<Identifier> parseOptionalIdentifier() {
    eatWhitespace();
    Loc lcur = l;

    optional<char> fst = this->at(lcur);
    if (!fst) {
      return {};
    }
    if (!isalpha(*fst)) {
      return {};
    }
    lcur = lcur.nextc(*fst);

    while (1) {
      optional<char> cchar = this->at(lcur);
      if (!cchar) {
        return {};
      }
      if (isWhitespace(*cchar) || isReservedSigil(*cchar)) {
        break;
      }
      lcur = lcur.nextc(s[lcur.si]);
    }

    const Span span = l.moveMut(lcur);
    return Identifier(span, s.substr(span.begin.si, span.nchars()));
  }

  optional<Span> parseOptionalKeyword(const string keyword) {
    eatWhitespace();
    // <keyword><non-alpha-numeric>
    Loc lcur = l;
    for (int i = 0; i < keyword.size(); ++i) {
      optional<char> c = this->at(lcur);
      if (!c) {
        return {};
      }
      if (c != keyword[i]) {
        return {};
      }
      lcur = lcur.nextc(*c);
    }
    optional<char> c = this->at(lcur);
    if (!c) {
      return {};
    }
    if (isalnum(*c)) {
      return {};
    };

    return l.moveMut(lcur);
  };

  Span parseKeyword(const string keyword) {
    optional<Span> ms = parseOptionalKeyword(keyword);
    if (ms) {
      return *ms;
    }

    const std::string err = "expected keyword |" + keyword + "|\n";
    addErr(Error(l, err));
    exit(1);
  }

  void addErr(Error e) {
    errs.push_back(e);
    printferr(e.loc, s.c_str(), e.errmsg.c_str());
  }

  void addErrAtCurrentLoc(string err) { addErr(Error(l, err)); }

  bool eof() {
    eatWhitespace();
    return l.si == s.size();
  }

  Loc getCurrentLoc() {
    eatWhitespace();
    return l;
  }

private:
  const string s;
  Loc l;
  vector<Error> errs;

  optional<char> at(Loc loc) {
    if (loc.si >= s.size()) {
      return optional<char>();
    }
    return s[loc.si];
  }

  void eatWhitespace() {
    while (1) {
      optional<char> ccur = this->at(l);
      if (!ccur) {
        return;
      }
      if (!isWhitespace(*ccur)) {
        return;
      }
      l = l.nextc(*ccur);
    }
  }
};
