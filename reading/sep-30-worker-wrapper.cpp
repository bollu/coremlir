// [Godbolt link](https://godbolt.org/z/rao3Ee)

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
};


template<typename T>
T force(Thunk<T> thnk) { return thnk.v(); }

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
    g_count = 0;
    printf("\n===mainf0===\n");
    SimpleInt out = f(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }

  }

template<typename T>
struct Force {
  T v;
  Force(T v): v(v) {};
  Force(Thunk<T> thnk) : v(thnk.v()) {};
  operator T() { return T(v);}

};

template<typename T>
T force(Force<T> forcedv) { return forcedv.v; }


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
    g_count = 0;
    printf("\n===mainf1===\n");
    SimpleInt out = f(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
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
    g_count = 0;
    printf("\n===mainf2===\n");
    SimpleInt out = f(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

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

namespace f3{
  SimpleInt f(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      //SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      // SimpleInt f_prev_v = apStrict(f, siprev);
      SimpleInt f_prev_v = apStrict(f, prev);
      return f_prev_v;
    }
  }

  int main() {
    g_count = 0;
    printf("\n===mainf3===\n");
    SimpleInt out = f(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
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
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
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
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
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
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

namespace g3{
  SimpleInt g(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;
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
    printf("\n===maing3===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

namespace g4{
  SimpleInt g(Unwrap<Force<SimpleInt>, int> i) {
    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;
    if (ihash <= 0) {
      return SimpleInt(42);
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      // SimpleInt g_prev_v = apStrict(g, siprev);
      SimpleInt g_prev_v = apStrict(g, prev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      SimpleInt ret = SimpleInt(rethash);
      return ret;
    }
  }

  int main() {
    g_count = 0;
    printf("\n===maing4===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

namespace g5{
  SimpleInt g(Unwrap<Force<SimpleInt>, int> i) {
    std::optional<int> iret; // return value

    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;
    if (ihash <= 0) {
      // return 42
      iret = 42;
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      // SimpleInt g_prev_v = apStrict(g, siprev);
      SimpleInt g_prev_v = apStrict(g, prev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      iret = rethash;
    }
    assert(iret);
    return SimpleInt(*iret);
  }

  int main() {
    g_count = 0;
    printf("\n===maing5===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

// denotes that we are wrapping a value into another value.
template <typename Inner, typename Outer>
struct Wrap {
  Inner wi;

  Wrap(Inner i) : wi(i) {}

  operator Outer () {
    return Outer(wi);
  }
  operator Inner () {
    return wi;
  }
};

namespace g6{
  Wrap<int, SimpleInt> g(Unwrap<Force<SimpleInt>, int> i) {
    std::optional<int> iret; // return value

    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;
    if (ihash <= 0) {
      // return 42
      iret = 42;
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      // SimpleInt g_prev_v = apStrict(g, siprev);
      SimpleInt g_prev_v = apStrict(g, prev);
      int g_prev_v_hash = casedefault(g_prev_v);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      iret = rethash;
    }
    assert(iret);
    return *iret;
  }

  int main() {
    g_count = 0;
    printf("\n===maing6===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

namespace g7{
  Wrap<int, SimpleInt> g(Unwrap<Force<SimpleInt>, int> i) {
    std::optional<int> iret; // return value

    // SimpleInt icons = force(i);
    // SimpleInt icons = i;
    // int ihash = casedefault(icons);
    int ihash = i;
    if (ihash <= 0) {
      // return 42
      iret = 42;
    } else {
      int prev = ihash - 1;
      // SimpleInt siprev = SimpleInt(prev);
      // Thunk<SimpleInt> siprev_t = thunkify(siprev);
      // SimpleInt g_prev_v = apStrict(g, siprev);
      // SimpleInt g_prev_v = apStrict(g, prev);
      // int g_prev_v_hash = casedefault(g_prev_v);
      int g_prev_v_hash = apStrict(g, prev);
      int rethash = g_prev_v_hash + 2;
      // SimpleInt ret = SimpleInt(rethash);
      // return ret;
      iret = rethash;
    }

    assert(iret);
    return *iret;
  }

  int main() {
    g_count = 0;
    printf("\n===maing7===\n");
    SimpleInt out = g(thunkify(SimpleInt(3)));
    printf("out: %d\n", out.v);
    return 0;
  }
}

int main() {
  f0::main();
  f1::main();
  f2::main();
  f3::main();
  g0::main();
  g1::main();
  g2::main();
  g3::main();
  g4::main();
  g5::main();
  g6::main();
  g7::main();
}
