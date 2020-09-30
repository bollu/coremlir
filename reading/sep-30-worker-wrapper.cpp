
#include <stdio.h>
#include <functional>
struct SimpleInt { int v; SimpleInt(int v) : v(v) {} };
int casedefault(SimpleInt s) { return s.v; }

template<typename T>
struct Thunk { std::function<T()> lzf; };
template<typename T>
T force(Thunk<T> thnk) { return thnk.lzf(); }

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

template<typename R, typename... Args> 
R apStrict(R(*f)(Args...), Args... args) { 
  return f(args...);
}

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
  printf("%d\n", f(thunkify(SimpleInt(1))).v);
}
