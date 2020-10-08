#pragma once
#include <assert.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdarg.h>
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

std::ostream &operator<<(std::ostream &o, const Loc &l) {
  return o << ":" << l.line << ":" << l.col;
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

std::ostream &operator<<(std::ostream &o, const Span &s) {
  return o << s.begin << " - " << s.end;
}

// TODO: upgrade this to take a space, not just a location.
void vprintfspan(Span span, const char *raw_input, const char *fmt,
                 va_list args) {
  char *outstr = nullptr;
  vasprintf(&outstr, fmt, args);
  assert(outstr);
  std::cerr << "===\n";
  std::cerr << span.begin << ":" << span.end << "\n";

  std::cerr << "===\n";
  std::cerr << span << "\t" << outstr << "\n";
  for (ll i = span.begin.si; i < span.end.si; ++i) {
    std::cerr << raw_input[i];
  }
  std::cerr << "\n===\n";
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

  std::cerr << "\n==\n"
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
  std::string errmsg;
  Loc loc;

  Error(Loc loc, std::string errmsg) : errmsg(errmsg), loc(loc){};
};

struct Identifier {
  const Span span;
  const std::string name;
  Identifier(const Identifier &other) = default;
  Identifier(Span span, std::string name) : span(span), name(name){};

  Identifier operator=(const Identifier &other) { return Identifier(other); }

  void print(std::ostream &f) const { f << name; }
};

struct Parser {
  Parser(const char *filename, const char *raw)
      : s(std::string(raw)), l(filename, 0, 1, 1){};

  Span parseOpenRoundBracket() { return parseSigil(std::string("(")); }

  std::optional<Span> parseOptionalOpenRoundBracket() {
    return parseOptionalSigil(std::string("("));
  }

  void parseCloseRoundBracket(Span open) {
    parseMatchingSigil(open, std::string(")"));
  }
  bool parseOptionalCloseRoundBracket(Span open) {
    return bool(parseOptionalMatchingSigil(open, std::string(")")));
  }

  std::pair<Span, ll> parseInteger() {
    std::optional<std::pair<Span, ll>> out = parseOptionalInteger();
    if (!out) {
      this->addErr(Error(l, std::string("unble to find integer")));
      exit(1);
    }
    return *out;
  }

  // [-][0-9]+
  std::optional<std::pair<Span, ll>> parseOptionalInteger() {
    eatWhitespace();
    bool negate = false;
    std::optional<char> ccur; // peeking character
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

  Span parseSigil(const std::string sigil) {
    std::optional<Span> span = parseOptionalSigil(sigil);
    if (span) {
      return *span;
    }

    addErr(Error(l, "expected sigil: |" + sigil + "|"));
    exit(1);
  }

  std::optional<Span> parseOptionalMatchingSigil(Span open,
                                                 const std::string sigil) {
    std::optional<Span> span = parseOptionalSigil(sigil);
    if (span) {
      return span;
    } else if (this->eof()) {
      addErr(Error(l, "found end of file!"));
      addErr(Error(l, "expected closing sigil: |" + sigil + "|"));
      addErr(Error(open.begin, "unmatched sigil opened here"));
      exit(1);
    }
    return {};
  }

  Span parseMatchingSigil(Span open, const std::string sigil) {
    std::optional<Span> span = parseOptionalMatchingSigil(open, sigil);
    if (span) {
      return *span;
    }
    addErr(Error(l, "expected closing sigil: |" + sigil + "|"));
    addErr(Error(open.begin, "unmatched sigil opened here"));

    exit(1);
  }

  // difference is that a sigil needs no whitespace after it, unlike
  // a keyword.
  std::optional<Span> parseOptionalSigil(const std::string sigil) {
    // std::cerr << __FUNCTION__ << "|" << sigil.c_str() << "|\n";
    std::optional<char> ccur;
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
    std::optional<Identifier> ms = parseOptionalIdentifier();
    if (ms.has_value()) {
      return *ms;
    }
    addErr(Error(l, std::string("expected identifier")));
    exit(1);
  }

  std::optional<Identifier> parseOptionalIdentifier() {
    eatWhitespace();
    Loc lcur = l;

    std::optional<char> fst = this->at(lcur);
    if (!fst || isReservedSigil(*fst)) { return {}; }
    lcur = lcur.nextc(*fst);

    while (1) {
      std::optional<char> c = this->at(lcur);
      if (!c || isWhitespace(*c) || isReservedSigil(*c)) {
        break;
      }
      lcur = lcur.nextc(s[lcur.si]);
    }

    const Span span = l.moveMut(lcur);
    return Identifier(span, s.substr(span.begin.si, span.nchars()));
  }

  std::optional<Span> parseOptionalKeyword(const std::string keyword) {
    eatWhitespace();
    // <keyword><non-alpha-numeric>
    Loc lcur = l;
    for (int i = 0; i < keyword.size(); ++i) {
      std::optional<char> c = this->at(lcur);
      if (!c) {
        return {};
      }
      if (c != keyword[i]) {
        return {};
      }
      lcur = lcur.nextc(*c);
    }
    std::optional<char> c = this->at(lcur);
    if (!c) {
      return {};
    }
    if (isalnum(*c)) {
      return {};
    };

    return l.moveMut(lcur);
  };

  Span parseKeyword(const std::string keyword) {
    std::optional<Span> ms = parseOptionalKeyword(keyword);
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

  void addErrAtCurrentLoc(std::string err) { addErr(Error(l, err)); }

  bool eof() {
    eatWhitespace();
    return l.si == s.size();
  }

  Loc getCurrentLoc() {
    eatWhitespace();
    return l;
  }

private:
  const std::string s;
  Loc l;
  std::vector<Error> errs;

  std::optional<char> at(Loc loc) {
    if (loc.si >= s.size()) {
      return std::optional<char>();
    }
    return s[loc.si];
  }

  void eatWhitespace() {
    while (1) {
      std::optional<char> ccur = this->at(l);
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

Parser parserFromPath(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    std::cerr << "unable to open file |" << path << "|\n";
    assert(false && "unable to open file.");
  }

  fseek(f, 0, SEEK_END);
  ll len = ftell(f);
  rewind(f);
  char *buf = (char *)malloc(len + 1);
  ll nread = fread(buf, 1, len, f);
  assert(nread == len);
  buf[nread] = 0;
  fclose(f);

  return Parser(path, buf);
}
