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

template <typename T>
struct Thunk {
    Thunk(std::function<T()> lzf) : lzf(lzf) {
        printf("- thunking: #%d\n", ++g_count);
    }

    static Thunk<T> thunkify(T value) {
        return Thunk([=]() { return value; });
    }

    T force() {
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

template <typename R, typename... Args>
Thunk<R> ap(std::function<R(Args...)> f, Args... args) {
    return Thunk<R>([=]() { return f(args...); });
}

template <typename R, typename... Args>
Thunk<R> ap(R (*f)(Args...), Args... args) {
    return Thunk<R>([=]() { return f(args...); });
}

struct MaybeInt {
    MaybeInt(Thunk<SimpleInt> v) : mv(v){};
    explicit MaybeInt() : mv() {}

    bool isJust() { return bool(mv); }
    Thunk<SimpleInt> just() {
        assert(mv.has_value());
        return *mv;
    }

   private:
    std::optional<Thunk<SimpleInt>> mv;
};

namespace f0 {
// incrm1 :: Maybe Int -> Maybe Int
// incrm1 mx = case mx of Nothing -> Nothing; Just x -> Just (x+1)
MaybeInt incrm1(Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    if (mv.isJust()) {
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
    if (i == 0) {
        MaybeInt mv = mt.force();
        return mv;
    } else {
        MaybeInt mv2 = incrm1(mt);
        Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        return incrmN(i - 1, mt2);
    }
}

void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    printf("===f0===\n");
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force().v);
}
}  // end namespace f0

// 1. Inline incrm1
namespace f1 {
// incrm1 :: Maybe Int -> Maybe Int
// incrm1 mx = case mx of Nothing -> Nothing; Just x -> Just (x+1)
// MaybeInt incrm1(Thunk<MaybeInt> mt) {
//     MaybeInt mv = mt.force();
//     if (mv.isJust()) {
//         Thunk<SimpleInt> sit = mv.just();
//         SimpleInt si = sit.force();
//         int i2 = si.v + 1;
//         SimpleInt si2 = SimpleInt(i2);
//         Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
//         MaybeInt ret = MaybeInt(si2t);
//         return ret;
//     } else {
//         MaybeInt ret = MaybeInt();
//         return ret;
//     }
// }

// incrm3 :: Maybe Int -> Maybe Int
// incrm3 mx = incrm1 (incrm1(incrm1(mx)))
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    if (i == 0) {
        MaybeInt mv = mt.force();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        MaybeInt mv = mt.force();
        if (mv.isJust()) {
            Thunk<SimpleInt> sit = mv.just();
            SimpleInt si = sit.force();
            int i2 = si.v + 1;
            SimpleInt si2 = SimpleInt(i2);
            Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
        }

        Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        return incrmN(i - 1, mt2);
    }
}

void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    printf("===f1===\n");
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force().v);
}
}  // namespace f1

// 2. hoist force to top of function.
namespace f2 {
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    if (i == 0) {
        // MaybeInt mv = mt.force();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        // MaybeInt mv = mt.force();
        if (mv.isJust()) {
            Thunk<SimpleInt> sit = mv.just();
            SimpleInt si = sit.force();
            int i2 = si.v + 1;
            SimpleInt si2 = SimpleInt(i2);
            Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
        }

        Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        return incrmN(i - 1, mt2);
    }
}

void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    printf("===f2===\n");
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force().v);
}
}  // namespace f2

// 3. outline forced computation
namespace f3 {
MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        // MaybeInt mv = mt.force();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        // MaybeInt mv = mt.force();
        if (mv.isJust()) {
            Thunk<SimpleInt> sit = mv.just();
            SimpleInt si = sit.force();
            int i2 = si.v + 1;
            SimpleInt si2 = SimpleInt(i2);
            Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
        }

        Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        return incrmN(i - 1, mt2);
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    printf("===f3===\n");
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force().v);
}
}  // end namespace f3

// - Remove a layer of thunkify):
//   Convert recursive call to use `incrmN_2` since we know that `incrmN`
//     immediately forces its argument.
namespace f4 {
MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        // MaybeInt mv = mt.force();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        // MaybeInt mv = mt.force();
        if (mv.isJust()) {
            Thunk<SimpleInt> sit = mv.just();
            SimpleInt si = sit.force();
            int i2 = si.v + 1;
            SimpleInt si2 = SimpleInt(i2);
            Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
        }

        // Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        // return incrmN(i - 1, mt2);
        return incrmN_2(i - 1, mv2);
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    printf("===f4===\n");
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force().v);
}
}  // end namespace f4

// - Remove a layer of MaybeInt(SimpleInt): since in the `if { .. }`
//   branch, we begin by forcing the the `SimpleInt`, we create a new
//   version of MaybeInt that has its argument pre-forced.
//   Convert recursive call to use `incrmN_2` since we know that `incrmN`
//     immediately forces its argument.
namespace f5 {

struct MaybeIntUnlifted {
    MaybeIntUnlifted(SimpleInt v) : mv(v){};
    explicit MaybeIntUnlifted() : mv() {}

    // TODO can this be auto generated? I think so, but check!
    MaybeInt toMaybeInt() {
        if (mv) {
            MaybeInt(Thunk<SimpleInt>::thunkify(*mv));
        } else {
            MaybeInt();
        }
    }

    bool isJust() { return bool(mv); }
    SimpleInt just() {
        assert(mv.has_value());
        return *mv;
    }

   private:
    std::optional<SimpleInt> mv;
};

MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv);

// only the just branch
MaybeInt incrmN_Just_3(int i, MaybeIntUnlifted mvu) {
    if (i == 0) {
        // MaybeInt mv = mt.force();
        // TODO impedance mismatch!
        MaybeInt mv = mvu.toMaybeInt();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        // MaybeInt mv = mt.force();
        MaybeInt mv = mvu.toMaybeInt();
        if (mv.isJust()) {
            Thunk<SimpleInt> sit = mv.just();
            SimpleInt si = sit.force();
            int i2 = si.v + 1;
            SimpleInt si2 = SimpleInt(i2);
            Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
        }

        // Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        // return incrmN(i - 1, mt2);
        return incrmN_2(i - 1, mv2);
    }
}

MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        // MaybeInt mv = mt.force();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        // MaybeInt mv = mt.force();
        if (mv.isJust()) {
            Thunk<SimpleInt> sit = mv.just();
            SimpleInt si = sit.force();
            int i2 = si.v + 1;
            SimpleInt si2 = SimpleInt(i2);
            Thunk<SimpleInt> si2t = Thunk<SimpleInt>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
        }

        // Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
        // return incrmN(i - 1, mt2);
        return incrmN_2(i - 1, mv2);
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    Thunk<SimpleInt> tsi = Thunk<SimpleInt>::thunkify(SimpleInt(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    printf("===f4===\n");
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force().v);
}
}  // namespace f5

int main() {
    f0::main();
    f1::main();
    f2::main();
    f3::main();
    f4::main();
    f5::main();
    return 0;
}
