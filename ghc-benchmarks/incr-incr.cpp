// Try to optimize map(+1) . map (+1) into a fused loop

#include <assert.h>
#include <stdio.h>

#include <functional>
#include <optional>

int g_count = 0;

struct SimpleInt {
  int v;
  SimpleInt(int v) : v(v) {
    printf("- building SimpleInt(%d): #%d\n", v, ++g_count);
  };

  operator int() { return v; }
};
int casedefault(SimpleInt s) {
  static int count = 0;
  return s.v;
}

template <typename T> struct Thunk {
  Thunk(std::function<T()> lzf) : lzf(lzf) {
    printf("- thunking: #%d\n", ++g_count);
  }

  T v() {
    if (!cached) {
      printf("- forcing: #%d\n", ++g_count);
      cached = lzf();
    }
    assert(cached);
    return *cached;
  }

private:
  std::optional<T> cached;
  std::function<T()> lzf;
};

template <typename T> T undefThunkFunction() {
  printf("EVALUATING UNDEF THUNK!");
  exit(1);
}

template <typename T> Thunk<T> undefThunk() {
  return Thunk<T>(undefThunkFunction<T>);
}

template <typename T> T force(Thunk<T> thnk) { return thnk.v(); }

template <typename T> Thunk<T> thunkify(T v) {
  return Thunk<T>([v]() { return v; });
}

template <typename R, typename... Args>
Thunk<R> ap(std::function<R(Args...)> f, Args... args) {
  return Thunk<R>([=]() { return f(args...); });
}

template <typename R, typename... Args>
Thunk<R> ap(R (*f)(Args...), Args... args) {
  return Thunk<R>([=]() { return f(args...); });
}

template <typename R, typename... Args>
R apStrict(std::function<R(Args...)> f, Args... args) {
  return f(args...);
}

// function arguments and real arguments can be mismatched,
// since function can take a Force instead of a Thunk ?
template <typename R, typename... FArgs, typename... Args>
R apStrict(R (*f)(FArgs...), Args... args) {
  return f(args...);
}

template <typename T> struct Force {
  T v;
  Force(T v) : v(v){};
  Force(Thunk<T> thnk) : v(thnk.v()){};
  operator T() { return T(v); }
};

template <typename T> T force(Force<T> forcedv) { return forcedv.v; }

template <typename Outer, typename Inner> struct Unwrap {
  Inner v;
  Unwrap(Outer outer) : v(outer){};
  Unwrap(Inner inner) : v(inner){};
  operator Inner() { return v; }
};

// specialize to allow implicit construction of a
// Unwrap<Force<Outer>> from an Outer
template <typename Outer, typename Inner> struct Unwrap<Force<Outer>, Inner> {
  Inner v;
  Unwrap(Outer outer) : v(outer.v){};
  Unwrap(Thunk<Outer> outer) : v(outer.v()){};
  Unwrap(Inner inner) : v(inner){};
  operator Inner() { return v; }
};

// denotes that we are wrapping a value into another value.
template <typename Inner, typename Outer> struct Wrap {
  Inner wi;

  Wrap(Inner i) : wi(i) {}

  operator Outer() { return Outer(wi); }
  operator Inner() { return wi; }
};

enum class ListTy { Nil, Cons };
struct List {
  ListTy ty;
  List(ListTy ty) : ty(ty) {}
};

struct ListNil : public List {
  ListNil() : List(ListTy::Nil) {
    printf("- building ListNil: #%d\n", ++g_count);
  };
};
struct ListCons : public List {
  ListCons(Thunk<SimpleInt> yt, Thunk<List *> yst)
      : List(ListTy::Cons), yt(yt), yst(yst) {
    printf("- building LitsCons: #%d\n", ++g_count);
  };
  Thunk<SimpleInt> yt;
  Thunk<List *> yst;
};

List *mkList(int i) {
  if (i == 0) {
    return new ListNil();
  } else {
    return new ListCons(thunkify(SimpleInt(i)), thunkify(mkList(i - 1)));
  }
}

void printList(List *xs) {
  switch (xs->ty) {
  case ListTy::Nil: {
    printf("Nil");
    return;
  }
  case ListTy::Cons: {
    ListCons *lc = (ListCons *)xs;
    SimpleInt si = force(lc->yt);
    printf("{Cons %d,\n", si.v);
    printList(force(lc->yst));
    printf("}");
  }
  }
}

// target
namespace ideal {
List *incr2(Thunk<List *> xst) {
  List *xsv = force(xst);
  switch (xsv->ty) {
  case ListTy::Nil: {
    return new ListNil();
  }
  case ListTy::Cons: {
    ListCons *xsv_cons = (ListCons *)(xsv);
    SimpleInt y = force(xsv_cons->yt);
    SimpleInt yincr = SimpleInt(y.v + 2);
    List *incr_ys = incr2(xsv_cons->yst);
    return new ListCons(thunkify(yincr), thunkify(incr_ys));
  }
  }
}

int main() {
  g_count = 0;
  printf("\n===ideal===\n");
  List *xs = mkList(3);
  List *ys = incr2(thunkify(xs));
  printList(ys);
  printf("\n");
  return 0;
}
} // namespace ideal

namespace f0 {
List *incr(Thunk<List *> xst) {
  List *xsv = force(xst);
  switch (xsv->ty) {
  case ListTy::Nil: {
    return new ListNil();
  }
  case ListTy::Cons: {
    ListCons *xsv_cons = (ListCons *)(xsv);
    SimpleInt y = force(xsv_cons->yt);
    SimpleInt yincr = SimpleInt(y.v + 1);
    List *incr_ys = incr(xsv_cons->yst);
    return new ListCons(thunkify(yincr), thunkify(incr_ys));
  }
  };
}

List *incr2(Thunk<List *> xst) {
  List *xst2 = incr(xst);
  List *xst3 = incr(thunkify(xst2));
  return xst3;
}

int main() {
  g_count = 0;
  printf("\n===f0===\n");
  List *xs = mkList(3);
  List *ys = incr2(thunkify(xs));
  printList(ys);
  printf("\n");
  return 0;
}
} // namespace f0

// 1. Outline List * since it's being forced
namespace f1 {

List *incr(Thunk<List *> xst);

List *incrv(List *xsv) {
  switch (xsv->ty) {
  case ListTy::Nil: {
    return new ListNil();
  }
  case ListTy::Cons: {
    ListCons *xsv_cons = (ListCons *)(xsv);
    SimpleInt y = force(xsv_cons->yt);
    SimpleInt yincr = SimpleInt(y.v + 1);
    List *incr_ys = incr(xsv_cons->yst);
    return new ListCons(thunkify(yincr), thunkify(incr_ys));
  }
  };
}
List *incr(Thunk<List *> xst) {
  List *xsv = force(xst);
  return incrv(xsv);
}

List *incr2(Thunk<List *> xst) {
  List *xst2 = incr(xst);
  List *xst3 = incr(thunkify(xst2));
  return xst3;
}

int main() {
  g_count = 0;
  printf("\n===f1===\n");
  List *xs = mkList(3);
  List *ys = incr2(thunkify(xs));
  printList(ys);
  printf("\n");
  return 0;
}
} // namespace f1

// 1. Outline List * since it's being forced
// 2. outline/Explode pattern match into two functions
// 3. outline the forcing of `yt`
namespace f2 {

List *incr(Thunk<List *> xst);

List *incrv_explode_nil() { return new ListNil(); }

List *incrv_explode_cons(Thunk<SimpleInt> yt, Thunk<List *> yst) {
  // ListCons *xsv_cons = (ListCons *)(xsv);
  // SimpleInt y = force(xsv_cons->yt);
  SimpleInt y = force(yt);
  SimpleInt yincr = SimpleInt(y.v + 1);
  // List *incr_ys = incr(xsv_cons->yst);
  List *incr_ys = incr(yst);
  return new ListCons(thunkify(yincr), thunkify(incr_ys));
}

List *incrv(List *xsv) {
  switch (xsv->ty) {
  case ListTy::Nil: {
    return incrv_explode_nil();
  }
  case ListTy::Cons: {
    ListCons *xsv_cons = (ListCons *)(xsv);
    return incrv_explode_cons(xsv_cons->yt, xsv_cons->yst);
  }
  };
}
List *incr(Thunk<List *> xst) {
  List *xsv = force(xst);
  return incrv(xsv);
}

List *incr2(Thunk<List *> xst) {
  List *xst2 = incr(xst);
  List *xst3 = incr(thunkify(xst2));
  return xst3;
}

int main() {
  g_count = 0;
  printf("\n===f2===\n");
  List *xs = mkList(3);
  List *ys = incr2(thunkify(xs));
  printList(ys);
  printf("\n");
  return 0;
}
} // namespace f2

// 1. Outline List * since it's being forced
// 2. outline/Explode pattern match into two functions
// 3. outline forced yt
namespace f3 {

List *incr(Thunk<List *> xst);

List *incrv_explode_nil() { return new ListNil(); }

List *incrv_explode_cons_forced_yt(SimpleInt y, Thunk<List *> yst) {
  SimpleInt yincr = SimpleInt(y.v + 1);
  // List *incr_ys = incr(xsv_cons->yst);
  List *incr_ys = incr(yst);
  return new ListCons(thunkify(yincr), thunkify(incr_ys));
}

List *incrv_explode_cons(Thunk<SimpleInt> yt, Thunk<List *> yst) {
  // ListCons *xsv_cons = (ListCons *)(xsv);
  // SimpleInt y = force(xsv_cons->yt);
  SimpleInt y = force(yt);
  return incrv_explode_cons_forced_yt(y, yst);
}

List *incrv(List *xsv) {
  switch (xsv->ty) {
  case ListTy::Nil: {
    return incrv_explode_nil();
  }
  case ListTy::Cons: {
    ListCons *xsv_cons = (ListCons *)(xsv);
    return incrv_explode_cons(xsv_cons->yt, xsv_cons->yst);
  }
  };
}
List *incr(Thunk<List *> xst) {
  List *xsv = force(xst);
  return incrv(xsv);
}

List *incr2(Thunk<List *> xst) {
  List *xst2 = incr(xst);
  List *xst3 = incr(thunkify(xst2));
  return xst3;
}

int main() {
  g_count = 0;
  printf("\n===f3===\n");
  List *xs = mkList(3);
  List *ys = incr2(thunkify(xs));
  printList(ys);
  printf("\n");
  return 0;
}
} // namespace f3

int main() {
  f0::main();
  f1::main();
  f2::main();
  f3::main();
  ideal::main();
  return 0;
}

