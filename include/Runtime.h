#pragma once
#include <assert.h>
#include <stdio.h>
#include <string.h>

extern "C" {

static int DEBUG_STACK_DEPTH = 0;
void DEBUG_INDENT();

#define DEBUG_LOG                                                              \
  if (1) {                                                                     \
    \ 
    DEBUG_INDENT();                                                            \
    fprintf(stderr, "%s ", __FUNCTION__);                                      \
  }
void DEBUG_PUSH_STACK();
void DEBUG_POP_STACK();

static const int MAX_CLOSURE_ARGS = 10;
struct Closure {
  int n;
  void *fn;
  void *args[MAX_CLOSURE_ARGS];
};

char *getPronouncableNum(size_t N);
char *getPronouncablePtr(void *N);

void *__attribute__((used))
mkClosure_capture0_args2(void *fn, void *a, void *b);
void *__attribute__((used)) mkClosure_capture0_args1(void *fn, void *a);
void *__attribute__((used)) mkClosure_capture0_args0(void *fn);
void *identity(void *v);
void *__attribute__((used)) mkClosure_thunkify(void *v);

typedef void *(*FnZeroArgs)();
typedef void *(*FnOneArg)(void *);
typedef void *(*FnTwoArgs)(void *, void *);

void *__attribute__((used)) evalClosure(void *closure_voidptr);

static const int MAX_CONSTRUCTOR_ARGS = 2;
struct Constructor {
  const char *tag; // inefficient!
  int n;
  void *args[MAX_CONSTRUCTOR_ARGS];
};

void *__attribute__((used)) mkConstructor0(const char *tag);
void *__attribute__((used)) mkConstructor1(const char *tag, void *a);
void *__attribute__((used)) mkConstructor2(const char *tag, void *a, void *b);
void *extractConstructorArg(void *cptr, int i);
bool isConstructorTagEq(const void *cptr, const char *tag);
} // end extern C
