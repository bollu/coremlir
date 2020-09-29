#pragma once
#include <stdio.h>
#include <string.h>
#include <assert.h>

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
