#include <assert.h>
#include <stdio.h>

#include <functional>
#include <optional>

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
int casedefault(SimpleInt s) { return s.v; }

template <typename T>
struct Thunk {
    Thunk(std::function<T()> lzf) : lzf(lzf) {
        printf("- thunking: #%d\n", ++g_count);
    }

    Thunk(T val) : lzf([=]() { return val; }) {
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

template <typename T>
T force(Thunk<T> thnk) {
    return thnk.v();
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

template <typename T>
struct Force {
    T v;
    Force(T v) : v(v){};

    template<typename TLIKE>
    Force(Thunk<TLIKE> t) : v(t.v()) { }

    template<typename TLIKE>
    Force(TLIKE vlike) : v(vlike) { }

    operator T() { return T(v); }

    template<typename R>
    R as() { return (R)v; }
};

template <typename T>
T force(Force<T> forcedv) {
    return forcedv.v;
}

template <typename Outer, typename Inner>
struct Unwrap {
    Inner v;
    Unwrap(Outer outer) : v(outer){};
    Unwrap(Inner inner) : v(inner){};
    operator Inner() { return v; }
};

// specialize to allow implicit construction of a
// Unwrap<Force<Outer>> from an Outer
template <typename Outer, typename Inner>
struct Unwrap<Force<Outer>, Inner> {
    Inner v;
    Unwrap(Outer outer) : v(outer.v){};
    Unwrap(Thunk<Outer> outer) : v(outer.v()){};
    Unwrap(Inner inner) : v(inner){};
    operator Inner() { return v; }
};

// denotes that we are wrapping a value into another value.
template <typename Inner, typename Outer>
struct Wrap {
    Inner wi;

    Wrap(Inner i) : wi(i) {}

    operator Outer() { return Outer(wi); }
    operator Inner() { return wi; }
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
    MJust(Thunk<SimpleInt> tin) : Maybe(MaybeTy::Just), tin(tin) {
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
        return (Maybe *)(new MJust(Thunk<SimpleInt>(xnext)));
    }
}
Maybe *incrm3(Thunk<Maybe *> tm0) {
    Maybe *tm1 = incrm1(tm0);
    Maybe *tm2 = incrm1(Thunk<Maybe *>(tm1));
    Maybe *tm3 = incrm1(Thunk<Maybe *>(tm2));
    return tm3;
}
int main() {
    g_count = 0;
    printf("\n===mainf0===\n");
    Maybe *out = incrm3(Thunk<Maybe *>((Maybe *)new MJust(Thunk<SimpleInt>(SimpleInt(39)))));
    MJust *jout = (MJust *)out;
    Thunk<SimpleInt> jout_val_thunk = jout->tin;
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace f0

// 1. outline forced functions
namespace f1 {
Maybe *incrm1_nil() { return new MNothing(); }
Maybe *incrm1_just(Thunk<SimpleInt> tin) {
    SimpleInt x = force(tin);
    int xnexthash = x.v + 1;
    SimpleInt xnext(xnexthash);
    return (Maybe *)(new MJust(Thunk<SimpleInt>(xnext)));
}

Maybe *incrm1(Thunk<Maybe *> tm) {
    Maybe *m = force(tm);
    if (m->ty == MaybeTy::Nothing) {
        return incrm1_nil();
    } else {
        MJust *mj = (MJust *)(m);
        Thunk<SimpleInt> tin = mj->tin;
        return incrm1_just(tin);
    }
}
Maybe *incrm3(Thunk<Maybe *> tm0) {
    Maybe *tm1 = incrm1(tm0);
    Maybe *tm2 = incrm1(Thunk<Maybe *>(tm1));
    Maybe *tm3 = incrm1(Thunk<Maybe *>(tm2));
    return tm3;
}
int main() {
    g_count = 0;
    printf("\n===mainf1===\n");
    Maybe *out = incrm3(Thunk<Maybe *>((Maybe *)new MJust(Thunk<SimpleInt>(39))));
    MJust *jout = (MJust *)out;
    Thunk<SimpleInt> jout_val_thunk = jout->tin;
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace f1

// 1. outline forced functions
// 2. inline incrm1 into incrm3 once.
namespace f2 {
Maybe *incrm1_nil() { return new MNothing(); }
Maybe *incrm1_just(Thunk<SimpleInt> tin) {
    SimpleInt x = force(tin);
    int xnexthash = x.v + 1;
    SimpleInt xnext(xnexthash);
    return (Maybe *)(new MJust(Thunk<SimpleInt>(xnext)));
}

Maybe *incrm1(Thunk<Maybe *> tm) {
    Maybe *m = force(tm);
    if (m->ty == MaybeTy::Nothing) {
        return incrm1_nil();
    } else {
        MJust *mj = (MJust *)(m);
        Thunk<SimpleInt> tin = mj->tin;
        return incrm1_just(tin);
    }
}
Maybe *incrm3(Thunk<Maybe *> tm0) {
    // Maybe *tm1 = incrm1(tm0);
    Maybe *tm1 = nullptr;
    {
        Maybe *m = force(tm0);
        if (m->ty == MaybeTy::Nothing) {
            tm1 = incrm1_nil();
        } else {
            MJust *mj = (MJust *)(m);
            Thunk<SimpleInt> tin = mj->tin;
            tm1 = incrm1_just(tin);
        }
    }

    Maybe *tm2 = incrm1(tm1);
    Maybe *tm3 = incrm1(tm2);
    return tm3;
}
int main() {
    g_count = 0;
    printf("\n===mainf2===\n");
    Maybe *out = incrm3(((Maybe *)new MJust((SimpleInt(39)))));
    MJust *jout = (MJust *)out;
    Thunk<SimpleInt> jout_val_thunk = jout->tin;
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace f2

// 1. outline forced functions
// 2. inline incrm1 into incrm3 once.
// 3. duplicate the code *after* the if/else into the two branches of the
// if/else.
namespace f3 {
Maybe *incrm1_nil() { return new MNothing(); }
Maybe *incrm1_just(Thunk<SimpleInt> tin) {
    SimpleInt x = force(tin);
    int xnexthash = x.v + 1;
    SimpleInt xnext(xnexthash);
    return (Maybe *)(new MJust((xnext)));
}

Maybe *incrm1(Thunk<Maybe *> tm) {
    Maybe *m = force(tm);
    if (m->ty == MaybeTy::Nothing) {
        return incrm1_nil();
    } else {
        MJust *mj = (MJust *)(m);
        Thunk<SimpleInt> tin = mj->tin;
        return incrm1_just(tin);
    }
}
Maybe *incrm3(Thunk<Maybe *> tm0) {
    // Maybe *tm1 = incrm1(tm0);
    Maybe *tm1 = nullptr;
    {
        Maybe *m = force(tm0);
        if (m->ty == MaybeTy::Nothing) {
            tm1 = incrm1_nil();
            Maybe *tm2 = incrm1((tm1));
            Maybe *tm3 = incrm1((tm2));
            return tm3;
        } else {
            MJust *mj = (MJust *)(m);
            Thunk<SimpleInt> tin = mj->tin;
            tm1 = incrm1_just(tin);
            Maybe *tm2 = incrm1((tm1));
            Maybe *tm3 = incrm1((tm2));
            return tm3;
        }
    }
}
int main() {
    g_count = 0;
    printf("\n===mainf3===\n");
    Maybe *out = incrm3(((Maybe *)new MJust((SimpleInt(39)))));
    MJust *jout = (MJust *)out;
    Thunk<SimpleInt> jout_val_thunk = jout->tin;
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace f3

// 1. outline forced functions
// 2. inline incrm1 into incrm3 once.
// 3. duplicate the code *after* the if/else into the two branches of the
// if/else
// 4. inline incrm3 once again.
namespace f4 {
Maybe *incrm1_nil() { return new MNothing(); }
Maybe *incrm1_just(Thunk<SimpleInt> tin) {
    SimpleInt x = force(tin);
    int xnexthash = x.v + 1;
    SimpleInt xnext(xnexthash);
    return (Maybe *)(new MJust((xnext)));
}

Maybe *incrm1(Thunk<Maybe *> tm) {
    Maybe *m = force(tm);
    if (m->ty == MaybeTy::Nothing) {
        return incrm1_nil();
    } else {
        MJust *mj = (MJust *)(m);
        Thunk<SimpleInt> tin = mj->tin;
        return incrm1_just(tin);
    }
}
Maybe *incrm3(Thunk<Maybe *> tm0) {
    // Maybe *tm1 = incrm1(tm0);
    Maybe *tm1 = nullptr;
    {
        Maybe *m = force(tm0);
        if (m->ty == MaybeTy::Nothing) {
            tm1 = incrm1_nil();
            // incrm1((tm1));
            Thunk<Maybe *> thunk_tm1 = (tm1);
            Maybe *tm2 = nullptr;
            {
                Maybe *m = force(thunk_tm1);
                if (m->ty == MaybeTy::Nothing) {
                    tm2 = incrm1_nil();
                } else {
                    MJust *mj = (MJust *)(m);
                    Thunk<SimpleInt> tin = mj->tin;
                    tm2 = incrm1_just(tin);
                }
            }
            Maybe *tm3 = incrm1((tm2));
            return tm3;
        } else {
            MJust *mj = (MJust *)(m);
            Thunk<SimpleInt> tin = mj->tin;
            tm1 = incrm1_just(tin);
            // Maybe *tm2 = incrm1((tm1));
            Thunk<Maybe *> thunk_tm1 = (tm1);
            Maybe *tm2 = nullptr;
            {
                Maybe *m = force(thunk_tm1);
                if (m->ty == MaybeTy::Nothing) {
                    tm2 = incrm1_nil();
                } else {
                    MJust *mj = (MJust *)(m);
                    Thunk<SimpleInt> tin = mj->tin;
                    tm2 = incrm1_just(tin);
                }
            }
            Maybe *tm3 = incrm1((tm2));
            return tm3;
        }
    }
}
int main() {
    g_count = 0;
    printf("\n===mainf4===\n");
    Maybe *out = incrm3(((Maybe *)new MJust((SimpleInt(39)))));
    MJust *jout = (MJust *)out;
    Thunk<SimpleInt> jout_val_thunk = jout->tin;
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace f4

template <typename L, typename R>
struct Either {
   public:
    Either(L l) : ol_(l){};
    Either(R r) : or_(r){};

    template <typename LLIKE, typename RLIKE>
    Either(const Either<LLIKE, RLIKE> &e) {
        if (e.isl()) {
            this->ol_ = L(e.l());
        } else {
            this->or_ = R(e.r());
        }
    }

    static Either left(L l) { return Either(l); }

    template <typename RLIKE>
    static Either right(RLIKE r) {
        return Either(R(r));
    }
    bool isl() const { return ol_.has_value(); }
    R r() const {
        assert(or_);
        return *or_;
    }
    L l() const { assert(ol_); return *ol_; }

   private:
    std::optional<L> ol_;
    std::optional<R> or_;
};

struct Unit {};

using MaybeADT = Either<Unit, Thunk<SimpleInt>>;

namespace g0 {
// Maybe *incrm1(Thunk<Maybe *> tm) {
MaybeADT incrm1(Thunk<MaybeADT> tm) {
    MaybeADT m = force(tm);
    if (m.isl()) {
        return MaybeADT::left(Unit());
    } else {
        Thunk<SimpleInt> tin = m.r();
        SimpleInt x = force(tin);
        int xnexthash = x.v + 1;
        SimpleInt xnext(xnexthash);
        return MaybeADT::right((xnext));
    }
}
MaybeADT incrm3(Thunk<MaybeADT> tm0) {
    MaybeADT tm1 = incrm1(tm0);
    MaybeADT tm2 = incrm1((tm1));
    MaybeADT tm3 = incrm1((tm2));
    return tm3;
}

int main() {
    g_count = 0;
    printf("\n===maing1===\n");
    MaybeADT out = incrm3((MaybeADT::right((SimpleInt(39)))));
    Thunk<SimpleInt> jout_val_thunk = out.r();
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace g0

// 1. replace Thunk with Force
namespace g1 {
// Maybe *incrm1(Thunk<Maybe *> tm) {
MaybeADT incrm1(Force<MaybeADT> tm) {
    MaybeADT m = tm;
    if (m.isl()) {
        return MaybeADT::left(Unit());
    } else {
        Thunk<SimpleInt> tin = m.r();
        SimpleInt x = force(tin);
        int xnexthash = x.v + 1;
        SimpleInt xnext(xnexthash);
        return MaybeADT::right((xnext));
    }
}
MaybeADT incrm3(Thunk<MaybeADT> tm0) {
    MaybeADT tm1 = incrm1(tm0);
    MaybeADT tm2 = incrm1((tm1));
    MaybeADT tm3 = incrm1((tm2));
    return tm3;
}

int main() {
    g_count = 0;
    printf("\n===maing1===\n");
    MaybeADT out = incrm3((MaybeADT::right((SimpleInt(39)))));
    Thunk<SimpleInt> jout_val_thunk = out.r();
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace g0

// 1. replace Thunk with Force
// 2. remove  using Force
namespace g2 {
// Maybe *incrm1(Thunk<Maybe *> tm) {
MaybeADT incrm1(Force<MaybeADT> tm) {
    MaybeADT m = tm;
    if (m.isl()) {
        return MaybeADT::left(Unit());
    } else {
        Thunk<SimpleInt> tin = m.r();
        SimpleInt x = force(tin);
        int xnexthash = x.v + 1;
        SimpleInt xnext(xnexthash);
        return MaybeADT::right((xnext));
    }
}
MaybeADT incrm3(Thunk<MaybeADT> tm0) {
    MaybeADT tm1 = incrm1(tm0);
    // MaybeADT tm2 = incrm1((tm1));
    MaybeADT tm2 = incrm1(tm1);
    // MaybeADT tm3 = incrm1((tm2));
    MaybeADT tm3 = incrm1(tm2);
    return tm3;
}

int main() {
    g_count = 0;
    printf("\n===maing2===\n");
    MaybeADT out = incrm3((MaybeADT::right((SimpleInt(39)))));
    Thunk<SimpleInt> jout_val_thunk = out.r();
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace g0



// 1. replace Thunk with Force
// 2. remove  using Force
// 3. Change the right component of MaybeADT to be a Force<SimpleInt>
namespace g3 {
using MaybeADTStrictR = Either<Unit, Force<SimpleInt>>;
// Maybe *incrm1(Thunk<Maybe *> tm) {
// MaybeADT incrm1(Force<MaybeADT> tm) {
MaybeADT incrm1(Force<MaybeADTStrictR> tm) {
    MaybeADT m = tm.v;
    if (m.isl()) {
        return MaybeADT::left(Unit());
    } else {
        Thunk<SimpleInt> tin = m.r();
        SimpleInt x = force(tin);
        int xnexthash = x.v + 1;
        SimpleInt xnext(xnexthash);
        return MaybeADT::right((xnext));
    }
}
MaybeADT incrm3(Thunk<MaybeADT> tm0) {
    MaybeADT tm1 = incrm1((tm0));
    // MaybeADT tm2 = incrm1((tm1));
    MaybeADT tm2 = incrm1((tm1));
    // MaybeADT tm3 = incrm1((tm2));
    MaybeADT tm3 = incrm1((tm2));
    return tm3;
}

int main() {
    g_count = 0;
    printf("\n===maing3===\n");
    MaybeADT out = incrm3((MaybeADT::right((SimpleInt(39)))));
    Thunk<SimpleInt> jout_val_thunk = out.r();
    SimpleInt jout_val_val = force(jout_val_thunk);
    printf("out: %d\n", jout_val_val.v);
    return 0;
}
}  // namespace g0



int main() {
    f0::main();
    f1::main();
    f2::main();
    f3::main();
    f4::main();
    g0::main();
    g1::main();
    g2::main();
    g3::main();
}
