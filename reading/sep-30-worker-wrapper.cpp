
#include <stdio.h>
#include <functional>
struct SimpleInt { int v; SimpleInt(int v) : v(v) {} };
int casedefault(SimpleInt s) { return s.v; }

template<typename T>
struct Thunk;

template<typename T>
struct Force {
  T val;
  Force(T val): val(val) {};
  operator T() { return val;}
};


template<typename T>
struct Thunk {
  std::function<T()> lzf;
  operator Force<T>() {
    return Force<T>(lzf());
  };
};

// template<typename T>
// Force<T>::Force (Thunk<T> thnk) : val (thnk.lzf()) {}

template<typename T>
T force(Thunk<T> thnk) { return thnk.lzf(); }

template<typename T>
T force(Force<T> forcedv) { return forcedv.val; }

template<typename T>
Thunk<T> thunkify(T val) { return Thunk<T>{ .lzf=[val]() { return val; } }; }

template<typename R, typename... Args> 
Thunk<R> ap(std::function<R(Args...)> f, Args... args) { 
  return Thunk<R>{ .lzf=[=]() { return f(args...); } };
}

template<typename R, typename... Args> 
Thunk<R> ap(R(*f)(Args...), Args... args) { 
  return Thunk<R>{ .lzf=[=]() { return f(args...); } };
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

SimpleInt f0(Thunk<SimpleInt> i) {
  SimpleInt icons = force(i);
  int ihash = casedefault(icons);
  if (ihash <= 0) {
    return SimpleInt(42);
  } else {
    int prev = ihash - 1;
    SimpleInt siprev = SimpleInt(prev);
    Thunk<SimpleInt> siprev_t = thunkify(siprev);
    SimpleInt f_prev_v = apStrict(f0, siprev_t);
    return f_prev_v;
  }
}

int mainf0() {
  printf("mainf0: %d\n", f0(thunkify(SimpleInt(1))).v);
}

SimpleInt f1(Force<SimpleInt> i) {
  // SimpleInt icons = force(i);
  SimpleInt icons = i;
  int ihash = casedefault(icons);
  if (ihash <= 0) {
    return SimpleInt(42);
  } else {
    int prev = ihash - 1;
    SimpleInt siprev = SimpleInt(prev);
    Thunk<SimpleInt> siprev_t = thunkify(siprev);
    SimpleInt f_prev_v = apStrict(f1, siprev_t);
    return f_prev_v;
  }
}

int mainf1() {
  printf("mainf1: %d\n", f1(thunkify(SimpleInt(1))).v);
}

SimpleInt f2(Force<SimpleInt> i) {
  // SimpleInt icons = force(i);
  SimpleInt icons = i;
  int ihash = casedefault(icons);
  if (ihash <= 0) {
    return SimpleInt(42);
  } else {
    int prev = ihash - 1;
    SimpleInt siprev = SimpleInt(prev);
    // Thunk<SimpleInt> siprev_t = thunkify(siprev);
    SimpleInt f_prev_v = apStrict(f2, siprev);
    return f_prev_v;
  }
}

int mainf2() {
  printf("mainf2: %d\n", f2(thunkify(SimpleInt(1))).v);
}

SimpleInt g0(Thunk<SimpleInt> i) {
    SimpleInt icons = force(i);
    int ihash = casedefault(icons);
    if (ihash <= 0) {
        return SimpleInt(42);
    } else {
        int prev = ihash - 1;
        SimpleInt siprev = SimpleInt(prev);
        Thunk<SimpleInt> siprev_t = thunkify(siprev);
        SimpleInt g_prev_v = apStrict(g0, siprev_t);
        int g_prev_v_hash = casedefault(g_prev_v);
        int rethash = g_prev_v_hash + 2;
        SimpleInt ret = SimpleInt(rethash);
        return ret;
    }
}

int maing0() {
    printf("maing0: %d\n", g0(thunkify(SimpleInt(3))).v);
}

int main() {
  mainf0();
  mainf1();
  mainf2();
  maing0();
}
