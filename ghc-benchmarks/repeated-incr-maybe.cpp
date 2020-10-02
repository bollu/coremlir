#include <stdio.h>
#include <functional>
#include <optional>
#include <assert.h>

// incrm1 :: Maybe Int -> Maybe Int
// incrm1 mx = case mx of Nothing -> Nothing; Just x -> Just (x+1)
// 
// incrm3 :: Maybe Int -> Maybe Int
// incrm3 mx = incrm1 (incrm1(incrm1(mx)))


int g_count = 0;

struct SimpleInt {
  int v;
  SimpleInt(int v) : v(v) {
    printf("- building SimpleInt(%d): #%d\n", v, ++g_count);
  };

  operator int() { return v; }
};
int casedefault(SimpleInt s) {
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

template<typename T>
struct Force {
  T v;
  Force(T v): v(v) {};
  Force(Thunk<T> thnk) : v(thnk.v()) {};
  operator T() { return T(v);}

};

template<typename T>
T force(Force<T> forcedv) { return forcedv.v; }


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


enum class MaybeTy { Nothing, Just };
struct Maybe {
  MaybeTy ty;
  Maybe(MaybeTy ty) : ty(ty) {}
};

struct MNothing : public Maybe {
  MNothing() : Maybe(MaybeTy::Nothing) {
    printf("- building MNothing: #%d\n", ++g_count);
  };
};
struct MJust : public Maybe {
  // thunk of value inside MJust
  MJust(Thunk<SimpleInt> tin) 
      : Maybe(MaybeTy::Just), tin(tin) {
    printf("- building MJust: #%d\n", ++g_count);
  };
  Thunk<SimpleInt> tin;
};

void printMaybe(Maybe *mx) {
    if (mx->ty == MaybeTy::Nothing) {
        printf("Nothing");
    } else {
        printf("Some");
    }
}

namespace f0 {
  Maybe *incrm1(Thunk<Maybe *> tm) {
    Maybe *m = force(tm);
    if (m->ty == MaybeTy::Nothing) {
        return new MNothing();
    } else {
        MJust *mj = (MJust *)(m);
        Thunk<SimpleInt> tin = mj->tin;
        SimpleInt x = force(tin);
        int xnexthash = x.v + 1;
        SimpleInt xnext(xnexthash);
        return (Maybe *)(new MJust(thunkify(xnext)));
    }
  }

  Maybe *incrm3(Thunk<Maybe *>tm0) {
      Maybe *tm1 = incrm1(tm0);
      Maybe *tm2 = incrm1(thunkify(tm1));
      Maybe *tm3 = incrm1(thunkify(tm2));
      return tm3;
  }

  int main() {
    g_count = 0;
    printf("\n===mainf0===\n");
    Maybe *out = incrm3(thunkify((Maybe *)new MJust(thunkify(SimpleInt(39)))));
    MJust *jout = (MJust *)out;
    Thunk<SimpleInt> jout_val_thunk = jout->tin;
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
  }
}

int main() {
  f0::main();
}
