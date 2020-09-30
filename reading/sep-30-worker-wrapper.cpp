#include <stdio.h>
#include <functional>
#include <optional>
#include <assert.h>

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

template<typename T>
struct Thunk;



template<typename T>
struct Thunk {
  Thunk(std::function<T()> lzf) : lzf(lzf) {
    printf("- thunking: #%d\n", ++g_count);
  }

  T v() {
    if(!cached) {
      printf("- forcing: #%d\n", ++g_count);
      cached = lzf();
    }
    assert(cached);
    return *cached;
  }

private:
  std::optional<T> cached;
  std::function<T()> lzf;
  // operator Force<T>() {
  //   return Force<T>(lzf());
  // };

  // template<typename Inner>
  // operator Unwrap<Force<T>, Inner> () {
  //   return Unwrap<Force<T>, Inner>(lzf().v);
  // };
};

template<typename T>
struct Force {
  T v;
  Force(T v): v(v) {};
  Force(Thunk<T> thnk) : v(thnk.v()) {
  };
  operator T() { return T(v);}

};


// template<typename T>
// Force<T>::Force (Thunk<T> thnk) : v (thnk.lzf()) {}

template<typename T>
T force(Thunk<T> thnk) { return thnk.v(); }

template<typename T>
T force(Force<T> forcedv) { return forcedv.v; }

template<typename T>
Thunk<T> thunkify(T v) { return Thunk<T>([v]() { return v; }); }

template<typename R, typename... Args> 
Thunk<R> ap(std::function<R(Args...)> f, Args... args) { 
  return Thunk<R>([=]() { return f(args...); });
}

template<typename R, typename... Args> 
Thunk<R> ap(R(*f)(Args...), Args... args) { 
  return Thunk<R>([=]() { return f(args...); });
}

template<typename R, typename... Args> 
R apStrict(std::function<R(Args...)> f, Args... args) { 
  return f(args...); 
}

// function arguments and real arguments can be mismatched,
// since function can take a Force instead of a Thunk ?
template<typename R, typename... FArgs, typename... Args> 
R apStrict(R(*f)(FArgs...), Args... args) { 
  return f(args...);
}

namespace f0 {
  SimpleInt f(Thunk<SimpleInt> i) {
    SimpleInt icons = force(i);
    int ihash = casedefault(icons);
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt f_prev_v = apStrict(f, siprev_t);
      return f_prev_v;
    }
  }

  int main() {
    printf("mainf0: %d\n", f(thunkify(SimpleInt(1))).v);
  }
}

namespace f1{
  SimpleInt f(Force<SimpleInt> i) {
    // SimpleInt icons = force(i);
    SimpleInt icons = i;
    int ihash = casedefault(icons);
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt f_prev_v = apStrict(f, siprev_t);
      return f_prev_v;
    }
  }

  int main() {
    printf("mainf1: %d\n", f(thunkify(SimpleInt(1))).v);
  }
}

namespace f2{
  SimpleInt f(Force<SimpleInt> i) {
    // SimpleInt icons = force(i);
    SimpleInt icons = i;
    int ihash = casedefault(icons);
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt f_prev_v = apStrict(f, siprev);
      return f_prev_v;
    }
  }

  int main() {
    printf("mainf2: %d\n", f(thunkify(SimpleInt(1))).v);
  }
}

namespace g0{
  SimpleInt g(Thunk<SimpleInt> i) {
    SimpleInt icons = force(i);
    int ihash = casedefault(icons);
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, siprev_t);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      SimpleInt ret = SimpleInt(rethash);
      return ret;
    }
  }

  int main() {
    g_count = 0;
    printf("\n===maing0===\n");
    printf("out: %d\n", g(thunkify(SimpleInt(3))).v);
  }
}

namespace g1 {
  SimpleInt g(Force<SimpleInt> i) {
    // SimpleInt icons = force(i);
    SimpleInt icons = i;
    int ihash = casedefault(icons);
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, siprev_t);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      SimpleInt ret = SimpleInt(rethash);
      return ret;
    }
  }

  int main() {
    g_count = 0;
    printf("\n===maing1===\n");
    printf("out: %d\n", g(thunkify(SimpleInt(3))).v);
  }
}

namespace g2{
  SimpleInt g(Force<SimpleInt> i) {
    // SimpleInt icons = force(i);
    SimpleInt icons = i;
    int ihash = casedefault(icons);
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, siprev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      SimpleInt ret = SimpleInt(rethash);
      return ret;
    }
  }

  int main() {
    g_count = 0;
    printf("\n===maing2===\n");
    printf("out: %d\n", g(thunkify(SimpleInt(3))).v);
  }
}

namespace g3 {
  SimpleInt g(Force<SimpleInt> i) {
    // SimpleInt icons = force(i);
    SimpleInt icons = i;
    int ihash = casedefault(icons);

    int retval;
    if (ihash <= 0) {
      // return SimpleInt(42);
      retval = 42;
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, siprev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      retval = rethash;
    }
    return SimpleInt(retval);
  }

  int main() {
    g_count = 0;
    printf("\n===maing3===\n");
    printf("out: %d\n", g(thunkify(SimpleInt(3))).v);
  }
};


// ===g4====

// Declare that we will unwrap the SimpleInt to produce an `int`.
template<typename Outer, typename Inner>
struct Unwrap {
  Inner v;
  Unwrap(Outer outer) : v(outer) {};
  Unwrap(Inner inner) : v(inner) {};
  operator Inner() { return v; }
};

// specialize to allow implicit construction of a
// Unwrap<Force<Outer>> from an Outer
template<typename Outer, typename Inner>
struct Unwrap<Force<Outer>, Inner> {
  Inner v;
  Unwrap(Outer outer) : v(outer.v) {};
  Unwrap(Thunk<Outer> outer) : v(outer.v()) {};
  Unwrap(Inner inner) : v(inner) {};
  operator Inner() { return v; }
};


namespace g4 {
  SimpleInt g(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;

    int retval;
    if (ihash <= 0) {
      // return SimpleInt(42);
      retval = 42;
    } else {
      int prev = ihash - 1;
      SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, siprev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      retval = rethash;
    }
    return SimpleInt(retval);
  }

  int main() {
    g_count = 0;
    printf("\n===maing4===\n");
    printf("out: %d\n", g(thunkify(SimpleInt(3))).v);
  }
};

// convert ~apStrict(g, SimpleInt(prev))~ into ~apStrict(g, prev)~
namespace g5 {
  SimpleInt g(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;

    int retval;
    if (ihash <= 0) {
      // return SimpleInt(42);
      retval = 42;
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, prev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      retval = rethash;
    }
    return SimpleInt(retval);
  }

  int main() {
    g_count = 0;
    printf("\n===maing5===\n");
    printf("out: %d\n", g(thunkify(SimpleInt(3))).v);
  }
};

// ===g6===

// denotes that we are wrapping a value into another value.
template <typename Inner, typename Outer>
struct Wrap {
  Inner wv;

  Wrap(Inner v) : wv(v) {}
  Wrap(Outer v) : wv(v.v) {
    printf("unwrapping (%d)\n", ++g_count);
  }

  operator Outer () { return Outer(wv);}
  operator Inner () {return wv; }
};

namespace g6 {
  Wrap<int, SimpleInt> g(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;

    int retval;
    if (ihash <= 0) {
      // return SimpleInt(42);
      retval = 42;
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      SimpleInt g_prev_v = apStrict(g, prev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      retval = rethash;
    }
    return retval;
  }

  int main() {
    g_count = 0;
    printf("\n===maing7===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    printf("\n===maing7===\n");
  }
};

// ===g7===

// denotes that we are wrapping a value into another value.

// Eliminate the unwrap to `SimpleInt` by calling `casedefault`.
namespace g7 {
  Wrap<int, SimpleInt> g(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;

    int retval;
    if (ihash <= 0) {
      // return SimpleInt(42);
      retval = 42;
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      // SimpleInt g_prev_v = apStrict(g, prev);
      // int g_prev_v_hash = casedefault(g_prev_v);

      int g_prev_v_hash = apStrict(g, prev);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      retval = rethash;
    }
    return retval;
  }

  int main() {
    g_count = 0;
    printf("\n===maing7===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    printf("\n===maing7===\n");
  }
};

int main() {
  f0::main();
  f1::main();
  f2::main();
  g0::main();
  g1::main();
  g2::main();
  g3::main();
  g4::main();
  g5::main();
  g6::main();
  g7::main();
}
