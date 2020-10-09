#include <assert.h>
#include <stdio.h>

#include <functional>
#include <optional>

int g_count = 0;

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
    MaybeInt(Thunk<int> v) : mv(v){};
    explicit MaybeInt() : mv() {}

    bool isJust() { return bool(mv); }
    Thunk<int> just() {
        assert(mv.has_value());
        return *mv;
    }

   private:
    std::optional<Thunk<int>> mv;
};

namespace f0 {
// incrm1Raw# :: Maybe# -> Maybe#
// incrm1Raw# mx = case mx of Nothing# -> Nothing# ; Just# x -> Just# (x +# 1#)
MaybeInt incrm1(Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    if (mv.isJust()) {
        Thunk<int> sit = mv.just();
        int si = sit.force();
        int i2 = si + 1;
        int si2 = int(i2);
        Thunk<int> si2t = Thunk<int>::thunkify(si2);
        MaybeInt ret = MaybeInt(si2t);
        return ret;
    } else {
        MaybeInt ret = MaybeInt();
        return ret;
    }
}

// incrmNRaw# :: Int# -> Maybe# -> Maybe#
// incrmNRaw# n mx = 
//   case n of 
//     0# -> mx; _ -> incrmNRaw# (n -# 1#) (incrm1Raw# mx)
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
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // end namespace f0

namespace f1 {
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
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
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
    printf("===%s===\n", __PRETTY_FUNCTION__);
    g_count = 0;
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f1

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
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
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
    printf("===%s===\n", __PRETTY_FUNCTION__);
    g_count = 0;
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f2

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
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
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
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // end namespace f3

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
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
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
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // end namespace f4

// - Copy the =return incrmN_2(i - 1, mv2)= into both branches
//   to get more information from the local context.
// - For more, think about  the "compiling with continuations" paper
//   where they advocate  outlining the computation after the branches
//   into a function and then creating function calls.

namespace f5 {

MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv);

MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        // MaybeInt mv = mt.force();
        return mv;
    } else {
        // MaybeInt mv2 = incrm1(mt);
        MaybeInt mv2;
        // MaybeInt mv = mt.force();
        if (mv.isJust()) {
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            // return ret;
            mv2 = ret;
            // Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
            // return incrmN(i - 1, mt2);
            return incrmN_2(i - 1, mv2);
        } else {
            MaybeInt ret = MaybeInt();
            // return ret;
            mv2 = ret;
            // Thunk<MaybeInt> mt2 = Thunk<MaybeInt>::thunkify(mv2);
            // return incrmN(i - 1, mt2);
            return incrmN_2(i - 1, mv2);
        }
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f5

namespace f6 {

MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv);

MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        return mv;
    } else {
        if (mv.isJust()) {
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            return incrmN_2(i - 1, ret);
        } else {
            MaybeInt ret = MaybeInt();
            return incrmN_2(i - 1, ret);
        }
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f6

namespace f7 {

MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv);

MaybeInt incrmN_Just3(int i, int si) {
    int i2 = si + 1;
    int si2 = int(i2);
    Thunk<int> si2t = Thunk<int>::thunkify(si2);
    MaybeInt ret = MaybeInt(si2t);
    return incrmN_2(i - 1, ret);
}

MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        return mv;
    } else {
        if (mv.isJust()) {
            Thunk<int> sit = mv.just();
            int si = sit.force();
            return incrmN_Just3(i, si);
        } else {
            MaybeInt ret = MaybeInt();
            return incrmN_2(i - 1, ret);
        }
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f7

struct MaybeIntUnlifted {
    MaybeIntUnlifted(int v) : mv(v){};
    explicit MaybeIntUnlifted() : mv() {}

    // TODO can this be auto generated? I think so, but check!
    MaybeInt toMaybeInt() {
        if (mv) {
            return MaybeInt(Thunk<int>::thunkify(*mv));
        } else {
            return MaybeInt();
        }
    }

    bool isJust() { return bool(mv); }
    int just() {
        assert(mv.has_value());
        return *mv;
    }

   private:
    std::optional<int> mv;
};

namespace f8 {

MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv);

MaybeInt incrmN_3(int i, MaybeIntUnlifted mv) {
    if (i == 0) {
        return mv.toMaybeInt();
    } else {
        if (mv.isJust()) {
            // Thunk<int> sit = mv.just();
            // int si = sit.force();
            int si = mv.just();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            return incrmN_2(i - 1, ret);
        } else {
            MaybeInt ret = MaybeInt();
            return incrmN_2(i - 1, ret);
        }
    }
}

MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        return mv;
    } else {
        if (mv.isJust()) {
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            Thunk<int> si2t = Thunk<int>::thunkify(si2);
            MaybeInt ret = MaybeInt(si2t);
            return incrmN_2(i - 1, ret);
        } else {
            MaybeInt ret = MaybeInt();
            return incrmN_2(i - 1, ret);
        }
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f8

namespace f9 {

MaybeInt incrmN(int i, Thunk<MaybeInt> mt);
MaybeInt incrmN_2(int i, MaybeInt mv);

MaybeInt incrmN_3(int i, MaybeIntUnlifted mv) {
    if (i == 0) {
        return mv.toMaybeInt();
    } else {
        if (mv.isJust()) {
            // Thunk<int> sit = mv.just();
            // int si = sit.force();
            int si = mv.just();
            int i2 = si + 1;
            int si2 = int(i2);
            // Thunk<int> si2t = Thunk<int>::thunkify(si2);
            // MaybeInt ret = MaybeInt(si2t);
            MaybeIntUnlifted ret = MaybeIntUnlifted(si2);
            // return incrmN_2(i - 1, ret);
            return incrmN_3(i-1, ret);
        } else {
            MaybeInt ret = MaybeInt();
            return incrmN_2(i - 1, ret);
        }
    }
}

MaybeInt incrmN_2(int i, MaybeInt mv) {
    if (i == 0) {
        return mv;
    } else {
        if (mv.isJust()) {
            Thunk<int> sit = mv.just();
            int si = sit.force();
            int i2 = si + 1;
            int si2 = int(i2);
            // Thunk<int> si2t = Thunk<int>::thunkify(si2);
            // MaybeInt ret = MaybeInt(si2t);
            MaybeIntUnlifted ret = MaybeIntUnlifted(si2);
            // return incrmN_2(i - 1, ret);
            return incrmN_3(i-1, ret);
        } else {
            MaybeInt ret = MaybeInt();
            return incrmN_2(i - 1, ret);
        }
    }
}
MaybeInt incrmN(int i, Thunk<MaybeInt> mt) {
    MaybeInt mv = mt.force();
    return incrmN_2(i, mv);
}

void main() {
    g_count = 0;
    printf("===%s===\n", __PRETTY_FUNCTION__);
    Thunk<int> tsi = Thunk<int>::thunkify(int(10));
    Thunk<MaybeInt> input = Thunk<MaybeInt>::thunkify(MaybeInt(tsi));
    MaybeInt output = incrmN(4, input);
    printf("%d\n", output.just().force());
    assert(14 == output.just().force());
}
}  // namespace f9

int main() {
    f0::main();
    f1::main();
    f2::main();
    f3::main();
    f4::main();
    f5::main();
    f6::main();
    f7::main();
    f8::main();
    f9::main();
    return 0;
}
