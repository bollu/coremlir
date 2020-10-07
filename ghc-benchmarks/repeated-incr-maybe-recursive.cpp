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



template<typename T>
struct Thunk {
  Thunk(std::function<T()> lzf) : lzf(lzf) {
    printf("- thunking: #%d\n", ++g_count);
  }

  static Thunk<T> thunkify(T value) {
    return Thunk([=]() { return value; });
  }

  T force() {
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



template<typename R, typename... Args> 
Thunk<R> ap(std::function<R(Args...)> f, Args... args) { 
  return Thunk<R>([=]() { return f(args...); });
}

template<typename R, typename... Args> 
Thunk<R> ap(R(*f)(Args...), Args... args) { 
  return Thunk<R>([=]() { return f(args...); });
}

struct MaybeInt {
  MaybeInt(Thunk<SimpleInt> v) : mv(v) {};
  explicit MaybeInt() : mv() {}

  bool isJust() { return bool(mv); }
  Thunk<SimpleInt> just() { assert(mv.has_value()); return *mv; }

private:
  std::optional<Thunk<SimpleInt>> mv;
};

namespace f0 {
  // incrm1 :: Maybe Int -> Maybe Int
  // incrm1 mx = case mx of Nothing -> Nothing; Just x -> Just (x+1)
  MaybeInt incrm1(Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    if(mv.isJust()) {
      Thunk<SimpleInt> sit = mv.just();
      SimpleInt si = sit.force();
      int i2 = si.v + 1;
      SimpleInt si2 = SimpleInt(i2);
      Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
      MaybeInt ret = MaybeInt(si2t);
      return ret;
    } else {
      MaybeInt ret = MaybeInt();
      return ret;
    }
  }

  // incrm3 :: Maybe Int -> Maybe Int
  // incrm3 mx = incrm1 (incrm1(incrm1(mx)))
  MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    if(i == 0)  {
      MaybeInt mv = mt.force();
      return mv;
    } else {
      MaybeInt mv2 = incrm1(mt);
      Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
      return incrmN(i-1, mt2); 
    }
  }

  void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("===f0===\n%d\n", output.just().force().v);
  }
} // end namespace f0

int main() {
  f0::main();
  return 0;
}
