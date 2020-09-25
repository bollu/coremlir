# Worker-wrapper

## Revision 2, Example 1: tail recursive function:

```
// A simple recursive function: f(Int 0) = Int 42; f(Int x) = f(x-1)
// We need to worker wrapper optimise this into:
// f(Int y) = Int (g# y)
// g# 0 = 1; g# x = g (x - 1) -- g# is strict.
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 42
module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]

  hask.func @f{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %ret = hask.case @SimpleInt %icons
               [@SimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseint %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                               %v = hask.make_i64(42)
                               %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                               hask.return (%boxed): !hask.adt<@SimpleInt>
                     }]
                     [@default -> { ^entry:
                                %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                                %onehash = hask.make_i64(1)
                                %prev = hask.primop_sub(%ihash, %onehash)
                                %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                                %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                                %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                    %box_prev_t) 
                                %prev_v = hask.force(%fprev_v): !hask.adt<@SimpleInt>
                                hask.return(%prev_v): !hask.adt<@SimpleInt>
                     }]
                     hask.return(%ret): !hask.adt<@SimpleInt>
               }]
        hask.return (%ret):!hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }


  hask.func@main {
    %lam = hask.lambda(%_: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %n = hask.make_i64(6)
      %box_n_v = hask.construct(@SimpleInt, %n: !hask.value): !hask.adt<@SimpleInt> 
      %box_n_t = hask.thunkify(%box_n_v: !hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
      %f = hask.ref(@f)  : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %out_t = hask.ap(%fib : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) ->  !hask.adt<@SimpleInt>>, %box_n_t)
      %out_v = hask.force(%out_t): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
}
```

#### Stage 1: Replace everything after `hask.force(%i)` into a separate function:

- We convert whatever comes after the `hask.force(%i)` into a `@f2`. We then
  eagerly call into `f2` [ie, no closure creation].

```
module {
  hask.func @f2{
        %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
            %ret = hask.case @SimpleInt %icons
                   [@SimpleInt -> { ^entry(%ihash: !hask.value):
                         %ret = hask.caseint %ihash 
                         [0 -> { ^entry(%_: !hask.value): 
                                   %v = hask.make_i64(42)
                                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                                   hask.return (%boxed): !hask.adt<@SimpleInt>
                         }]
                         [@default -> { ^entry:
                                    %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                                    %onehash = hask.make_i64(1)
                                    %prev = hask.primop_sub(%ihash, %onehash)
                                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                                    %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                                    %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                        %box_prev_t) 
                                    %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                                    hask.return(%prev_v): !hask.adt<@SimpleInt>
                         }]
                         hask.return(%ret): !hask.adt<@SimpleInt>
                   }]
            hask.return (%ret):!hask.adt<@SimpleInt>
        }
        hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @f{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %f2 = hask.ref(@f2)
        %ret = hask.apEager(%f2, %icons)
        hask.return(%ret) : !hask.adt<@SimpleInt>
    }
    hask.return(%lam):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>> 
  }
```

#### Stage 2: replace the recursive call of `f` to a call of `f2`

- We can replace any call of the form `f(thunkify(value))` to `f2(value)`.

```
hask.func @f2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ret = hask.case @SimpleInt %icons
               [@SimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseint %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                               %v = hask.make_i64(42)
                               %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                               hask.return (%boxed): !hask.adt<@SimpleInt>
                     }]
                     [@default -> { ^entry:
                                %f2 = hask.ref(@f2):  !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                                %onehash = hask.make_i64(1)
                                %prev = hask.primop_sub(%ihash, %onehash)
                                %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                                // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                                // OLD: %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                // OLD:     %box_prev_t) 
                                %fprev_t = hask.ap(%f2: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, 
                                    %box_prev_v) 
                                %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                                hask.return(%prev_v): !hask.adt<@SimpleInt>
                     }]
                     hask.return(%ret): !hask.adt<@SimpleInt>
               }]
        hask.return (%ret):!hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}
```

#### Stage 3: replace the case-of-single argument with special instruction `@hask.singlecase`

- Replace the case-of-single argument with a special instruction `@hask.singlecase`


```
hask.func @f2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.singlecase(@SimpleInt, %icons)
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %f2 = hask.ref(@f2):  !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    // OLD: %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %box_prev_t) 
                    %fprev_t = hask.ap(%f2: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, 
                        %box_prev_v) 
                    %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                    hask.return(%prev_v): !hask.adt<@SimpleInt>
         }]
         hask.return(%ret): !hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}
```

#### Stage 4: Replace everything after `hask.singlecase(%ihash)` into a separate function:

```
hask.func @f3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %f2 = hask.ref(@f2):  !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    // OLD: %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %box_prev_t) 
                    %fprev_t = hask.ap(%f2: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, 
                        %box_prev_v) 
                    %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                    hask.return(%prev_v): !hask.adt<@SimpleInt>
         }]
         hask.return(%ret): !hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}

hask.func @f2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.singlecase(@SimpleInt, %icons)
        %f3 = hask.ref(@f3) : !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>
        %ret = apEager(%f3, %ihash)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}
```

#### Stage 5: Replace the call of `f2` to a call of `f3`:

```
hask.func @f3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %f2 = hask.ref(@f2):  !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // OLD: %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    // OLD: %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %box_prev_t) 
                    // call changed to f3
                    %fprev_t = hask.ap(%f3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                        %prev) 
                    %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                    hask.return(%prev_v): !hask.adt<@SimpleInt>
         }]
         hask.return(%ret): !hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}

hask.func @f2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.singlecase(@SimpleInt, %icons)
        %f3 = hask.ref(@f3) : !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>
        %ret = apEager(%f3, %ihash)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}
```



#### Stage 6: Replace the call of `force(ap(f3))` with `apEager`

```
hask.func @f3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %f2 = hask.ref(@f2):  !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // OLD: %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    // OLD: %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %box_prev_t) 
                    // call changed to f3
                    // OLD: %fprev_t = hask.ap(%f3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %prev) 
                    // OLD: %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                    %fprev_v = hask.apEager(%f3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>,
                                    %prev): !hask.adt<@SimpleInt>
                    hask.return(%prev_v): !hask.adt<@SimpleInt>
         }]
         hask.return(%ret): !hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}

hask.func @f2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.singlecase(@SimpleInt, %icons)
        %f3 = hask.ref(@f3) : !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>
        %ret = apEager(%f3, %ihash)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}
```

#### Stage 7: Unwind: inline back the now non-recursive calls in `f1 -> f2 -> (f3 -> f3 -> ...)`

```
hask.func @f{
  %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %icons = hask.force(%i):!hask.adt<@SimpleInt>
      %f2 = hask.ref(@f2)
      %ret = hask.apEager(%f2, %icons)
      hask.return(%ret) : !hask.adt<@SimpleInt>
  }
  hask.return(%lam):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>> 
}
```

on inlining `f2` becomes:

```
hask.func @f{
  %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %icons = hask.force(%i):!hask.adt<@SimpleInt>
      %ihash = hask.singlecase(@SimpleInt, %icons)
      %f3 = hask.ref(@f3) : !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>
      %ret = apEager(%f3, %ihash)
      hask.return(%ret) : !hask.adt<@SimpleInt>
  }
  hask.return(%lam):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>> 
}
```

That's it. We should not inline `f3` because it's recursive. We thus get a clean
separation between the "wrapper" which forces the thunk, and the "worker"
which operates on integers. This example is kind of easy, because `f` is tail
recursive. We'll need try `fib`, which is *not* tail recursive, and should
thus be more challenging.

Our final worker `f3` is:

```
hask.func @f3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %f2 = hask.ref(@f2):  !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // OLD: %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    // OLD: %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %box_prev_t) 
                    // call changed to f3
                    // OLD: %fprev_t = hask.ap(%f3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                    // OLD:     %prev) 
                    // OLD: %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                    %fprev_v = hask.apEager(%f3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>,
                                    %prev): !hask.adt<@SimpleInt>
                    hask.return(%prev_v): !hask.adt<@SimpleInt>
         }]
         hask.return(%ret): !hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
}
```


## Revision 2, Example 2: `g`: non-tail-recursive function

```hs
-- g 0 = 42; g x = (g (x - 1)) + 2
g (Int 0) = Int 42
g (Int x) = case g (Int (x-1)) of
                Int y -> Int (y + 2)
```

```hs
-- elaborated version of previous in ANF
g ix = case ix of
  SimpleInt x -> case x of
    0 -> 42
    i# -> let prev# = i# - 1 
             in case g (SimpleInt prev#) of
                    gprev -> gprev + 10
```

##### Stage 0

```
// stupid MLIR encoding, using casedefault and `apEager` already wherever necessary.
module {
  hask.func @g{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %ihash = hask.casedefault(@SimpleInt, %icons) : i64
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                        %box_prev_t) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    hask.return(%ret_v): !hask.adt<@SimpleInt>
         }]
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
```

##### Stage 1: everything after `hask.force(%i)` is a new `g2`

```

module {
  hask.func @g2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.casedefault(@SimpleInt, %icons) : i64
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                        %box_prev_t) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    hask.return(%ret_v): !hask.adt<@SimpleInt>
         }]
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %g2 = hask.ref(@g2)
        %ret = hask.apEager(%g2, %icons)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
```

##### Stage 2: replace call of `apEager(g1, thunkify(x))` in `g2` to `apEager(g2, x)`

```
module {
  hask.func @g2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.casedefault(@SimpleInt, %icons) : i64
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g2: !hask.fn<(!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                        %box_prev_v) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    hask.return(%ret_v): !hask.adt<@SimpleInt>
         }]
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %g2 = hask.ref(@g2)
        %ret = hask.apEager(%g2, %icons)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
```

##### Stage 3: move everything after `hask.casedefault` to a `g3`

```
module {
  hask.func @g3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g2: !hask.fn<(!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                        %box_prev_v) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    hask.return(%ret_v): !hask.adt<@SimpleInt>
         }]
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.casedefault(@SimpleInt, %icons) : i64
        %g3 = hask.ref(@g3)
        %ret = hask.apEager(%g3, %icons)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %g2 = hask.ref(@g2)
        %ret = hask.apEager(%g2, %icons)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
```

##### Stage 4: replace the call of `apEager(g2, construct(x))` with `apEager(g3, x)`:

```
module {
  hask.func @g3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %ret = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   hask.return (%boxed): !hask.adt<@SimpleInt>
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                        %prev) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    hask.return(%ret_v): !hask.adt<@SimpleInt>
         }]
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g2{
    %lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
        %ihash = hask.casedefault(@SimpleInt, %icons) : i64
        %g3 = hask.ref(@g3)
        %ret = hask.apEager(%g3, %icons)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %g2 = hask.ref(@g2)
        %ret = hask.apEager(%g2, %icons)
        hask.return(%ret)
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
```

- Now comes the hard part: we need to remove the **wrapping of the return value**


##### Stage 5: move the call of `hask.construct(@SimpleInt, ...)` to be outside the `case` 

- The rationale is that we want to float wrapping out. That is, we want the
  "push the wrapping" as externally as possible. I don't see a nice generic
  way to motivate doing this, sadly.
  
```
module {
  hask.func @g3{
    %lam = hask.lambda(%ihash: !hask.value) {
         %rethash = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   // OLD: %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   // OLD: hask.return (%boxed): !hask.adt<@SimpleInt>
                    hask.return(%v): !hask.value
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                        %prev) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    // OLD: %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    // OLD: hask.return(%ret_v): !hask.adt<@SimpleInt>
                    hask.return(%rethash): !hask.value
         }]

        %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
        hask.return(%ret_v): !hask.value
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g2{
    ...
  }

  hask.func @g{
    ...
  }
```

#### Step 6: move everything *above* `hask.return(%ret_v): !hask.value` to a separate function

```
module {
  hask.func @g4{
    %lam = hask.lambda(%ihash: !hask.value) {
         %rethash = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   // OLD: %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   // OLD: hask.return (%boxed): !hask.adt<@SimpleInt>
                    hask.return(%v): !hask.value
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_v = hask.apEager(%g3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                        %prev) 
                    %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    // OLD: %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    // OLD: hask.return(%ret_v): !hask.adt<@SimpleInt>
                    hask.return(%rethash): !hask.value
         }]

        hask.return(%rethash): !hask.value
    }
    hask.return (%lam): !hask.fn<(!hask.value) -> !hask.value>
  }

  hask.func @g3{
    %lam = hask.lambda(%ihash: !hask.value) {
        %rethash = hask.apStrict(@g4, %ihash)
        %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
        hask.return(%ret_v): !hask.value
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }

  hask.func @g2{
    ...
  }

  hask.func @g{
    ...
  }
```

#### Step 7: replace the call `casedefault(apEager(g3, ...))` to `apEager(g4, ...)`

```
module {
  hask.func @g4{
    %lam = hask.lambda(%ihash: !hask.value) {
         %rethash = hask.caseint %ihash 
         [0 -> { ^entry(%_: !hask.value): 
                   %v = hask.make_i64(42)
                   // OLD: %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                   // OLD: hask.return (%boxed): !hask.adt<@SimpleInt>
                    hask.return(%v): !hask.value
         }]
         [@default -> { ^entry:
                    %g = hask.ref(@g):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                    %onehash = hask.make_i64(1)
                    %prev = hask.primop_sub(%ihash, %onehash)
                    // %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                    // OLD: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                    %gprev_vhash = hask.apEager(%g4: !hask.fn<(!hask.value) -> !hask.value>,
                        %prev) 
                    // OLD: %gprev_vhash = hask.casedefault(@SimpleInt, %gprev_v)
                    %tenhash = hask.make_i64(10)
                    %rethash = hask.primop_add(%gprev_vhash, %tenhash)
                    // OLD: %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
                    // OLD: hask.return(%ret_v): !hask.adt<@SimpleInt>
                    hask.return(%rethash): !hask.value
         }]

        hask.return(%rethash): !hask.value
    }
    hask.return (%lam): !hask.fn<(!hask.value) -> !hask.value>
  }

  hask.func @g3 {
      ...
  }

  hask.func @g2{
    ...
  }

  hask.func @g{
    ...
  }
```
#### Step 8: convert recursion into loop?

LLVM manages to optimize:

```c
int foo(int x) {
    if (x <= 0) { return 42; }
    return 10 + foo(x-2);
}
```

the corresponding LLVM:

```ll
define dso_local i32 @foo(i32 %x) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %cmp = icmp sle i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 42, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %1 = load i32, i32* %x.addr, align 4
  %sub = sub nsw i32 %1, 2
  %call = call i32 @foo(i32 %sub)
  %add = add nsw i32 10, %call
  store i32 %add, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, i32* %retval, align 4
  ret i32 %2
}
```


into:

```ll
; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @foo(i32 %x) local_unnamed_addr #0 {
entry:
  %cmp2 = icmp slt i32 %x, 1
  br i1 %cmp2, label %return, label %if.end.preheader

if.end.preheader:                                 ; preds = %entry
  %0 = xor i32 %x, -1
  %1 = icmp sgt i32 %0, -3
  %smax = select i1 %1, i32 %0, i32 -3
  %2 = add i32 %x, 2
  %3 = add i32 %2, %smax
  %4 = lshr i32 %3, 1
  %5 = mul i32 %4, 10
  %6 = add i32 %5, 52
  br label %return

return:                                           ; preds = %if.end.preheader, %entry
  %accumulator.tr.lcssa = phi i32 [ 42, %entry ], [ %6, %if.end.preheader ]
  ret i32 %accumulator.tr.lcssa
}                                                        
```


I'm not 100% sure how it converts the non-tail-call into an accumulator.
Will need to read how it does so. I'm hoping MLIR can do this (at the Standard level).
I'm not very hopeful, though. Will check soon.

#### Step 9: inline `g1 -> g2 -> g3 -> g4 -> g4 -> ...`

```
hask.func @g{
%lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
    %icons = hask.force(%i):!hask.adt<@SimpleInt>
    %g2 = hask.ref(@g2)
    %ret = hask.apEager(%g2, %icons)
    hask.return(%ret)
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```

- inline `g2`:

```
hask.func @g2{
%lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
    %ihash = hask.casedefault(@SimpleInt, %icons) : i64
    %g3 = hask.ref(@g3)
    %ret = hask.apEager(%g3, %icons)
    hask.return(%ret)
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```

- after inlining `g2` into `g`:

```
hask.func @g{
%lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
    %icons = hask.force(%i):!hask.adt<@SimpleInt>
    %ihash = hask.casedefault(@SimpleInt, %icons) : i64
    %g3 = hask.ref(@g3)
    %ret = hask.apEager(%g3, %icons)
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```

- inline `g3`:


```
hask.func @g3{
%lam = hask.lambda(%ihash: !hask.value) {
    %rethash = hask.apStrict(@g4, %ihash)
    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
    hask.return(%ret_v): !hask.value
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```

- after inlining `g3` into `g`:


```
hask.func @g{
%lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
    %icons = hask.force(%i):!hask.adt<@SimpleInt>
    %ihash = hask.casedefault(@SimpleInt, %icons) : i64
    %rethash = hask.apStrict(@g4, %ihash)
    %ret_v = hask.construct(@SimpleInt, %tenhash:!hask.value)
    hask.return(%ret_v): !hask.value
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```

- We now have the 'worker' `g4` and the wrapper `g`.

#### Thoughts

- Our major "optimization" seems to come from _outlining_, which is an exact
  _dual_ of the usual case, where we get performance from _inlining_.


## Revision 1

Consider the recursive function:


```
// A simple recursive function: f(Int 0) = Int 42; f(Int x) = f(x-1)
// We need to worker wrapper optimise this into:
// f(Int y) = Int (g# y)
// g# 0 = 1; g# x = g (x - 1) -- g# is strict.
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 42
module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]

  hask.func @f{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %ret = hask.case @SimpleInt %icons
               [@SimpleInt -> { ^entry(%ihash: !hask.value):
                     %ret = hask.caseint %ihash 
                     [0 -> { ^entry(%_: !hask.value): 
                               %v = hask.make_i64(42)
                               %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                               hask.return (%boxed): !hask.adt<@SimpleInt>
                     }]
                     }]
                     [@default -> { ^entry:
                                %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                                %onehash = hask.make_i64(1)
                                %prev = hask.primop_sub(%ihash, %onehash)
                                %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                                %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                                %fprev_t = hask.ap(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                    %box_prev_t) 
                                %fprev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                                hask.return(%fprev_v): !hask.adt<@SimpleInt>
                     }]
                     hask.return(%ret): !hask.adt<@SimpleInt>
               }]
        hask.return (%ret):!hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }


  hask.func@main {
    %lam = hask.lambda(%_: !hask.thunk<!hask.adt<@SimpleInt>>) {
      %n = hask.make_i64(6)
      %box_n_v = hask.construct(@SimpleInt, %n: !hask.value): !hask.adt<@SimpleInt> 
      %box_n_t = hask.thunkify(%box_n_v: !hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
      %f = hask.ref(@f)  : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %out_t = hask.ap(%fib : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) ->  !hask.adt<@SimpleInt>>, %box_n_t)
      %out_v = hask.force(%out_t): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
}
```

How should we worker/wrapper this? As far as I can tell, it should proceed
in these stages:

#### Stage 1: replace `force(ap(f, ...))` with `apEager(f, ...)`.

We can't actually inline the call, but we can at least improve the situation
by noting that we want to make the call eagerly.

```
hask.func @f{
%lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
    %icons = hask.force(%i):!hask.adt<@SimpleInt>
    %ret = hask.case @SimpleInt %icons
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
                 %ret = hask.caseint %ihash 
                 [0 -> { ^entry(%_: !hask.value): 
                           %v = hask.make_i64(42)
                           %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                           hask.return (%boxed): !hask.adt<@SimpleInt>
                 }]
                 }]
                 [@default -> { ^entry:
                            %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                            %onehash = hask.make_i64(1)
                            %prev = hask.primop_sub(%ihash, %onehash)
                            %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                            %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                            // changed to fprev_v 
                            %fprev_v = hask.apEager(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                %box_prev_t) 
                            // ELIMINATED: %prev_v = hask.force(%fprev_v): !hask.adt<@SimpleInt>
                            hask.return(%fprev_v): !hask.adt<@SimpleInt>
                 }]
                 hask.return(%ret): !hask.adt<@SimpleInt>
           }]
    hask.return (%ret):!hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```

#### Stage 2: clone `f` into `f2`: strict in its argument/takes a value, not a thunk

Since we have a `hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>)`
which instantly forces its arguments with a `%icons = hask.force(%i):!hask.adt<@SimpleInt>`,
we create a new `f2`:

```
hask.func @f2{
%lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
    %ret = hask.case @SimpleInt %icons
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
                 %ret = hask.caseint %ihash 
                 [0 -> { ^entry(%_: !hask.value): 
                           %v = hask.make_i64(42)
                           %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                           hask.return (%boxed): !hask.adt<@SimpleInt>
                 }]
                 }]
                 [@default -> { ^entry:
                            %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                            %onehash = hask.make_i64(1)
                            %prev = hask.primop_sub(%ihash, %onehash)
                            %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                            %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                            %fprev_v = hask.apEager(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                %box_prev_t) 
                            hask.return(%fprev_v): !hask.adt<@SimpleInt>
                 }]
                 hask.return(%ret): !hask.adt<@SimpleInt>
           }]
    hask.return (%ret):!hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}

hask.func @f{
%lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
    %icons = hask.force(%i):!hask.adt<@SimpleInt>
    %ret = hask.case @SimpleInt %icons
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
                 %ret = hask.caseint %ihash 
                 [0 -> { ^entry(%_: !hask.value): 
                           %v = hask.make_i64(42)
                           %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                           hask.return (%boxed): !hask.adt<@SimpleInt>
                 }]
                 }]
                 [@default -> { ^entry:
                            %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                            %onehash = hask.make_i64(1)
                            %prev = hask.primop_sub(%ihash, %onehash)
                            %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                            %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                            %fprev_v = hask.apEager(%f: !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
                                %box_prev_t) 
                            hask.return(%fprev_v): !hask.adt<@SimpleInt>
                 }]
                 hask.return(%ret): !hask.adt<@SimpleInt>
           }]
    hask.return (%ret):!hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```


#### Stage 3: Change the recursive call of `f` to  call of `f2`:

Since we have a call of the form `apEager(f(thunkify(constructor)))`, we 
can change this to `apEager(f_strict(constructor))`

```
hask.func @f2{
%lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
    %ret = hask.case @SimpleInt %icons
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
                 %ret = hask.caseint %ihash 
                 [0 -> { ^entry(%_: !hask.value): 
                           %v = hask.make_i64(42)
                           %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                           hask.return (%boxed): !hask.adt<@SimpleInt>
                 }]
                 }]
                 [@default -> { ^entry:
                            %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                            %onehash = hask.make_i64(1)
                            %prev = hask.primop_sub(%ihash, %onehash)
                            %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                            // ELIMINTATED: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                            // CALL CHANGED TO f2
                            %fprev_v = hask.apEager(%f2: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, 
                                %box_prev_t) 
                            hask.return(%fprev_v): !hask.adt<@SimpleInt>
                 }]
                 hask.return(%ret): !hask.adt<@SimpleInt>
           }]
    hask.return (%ret):!hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}

hask.func @f{
%lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
    %icons = hask.force(%i):!hask.adt<@SimpleInt>
    %ret = hask.case @SimpleInt %icons
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
                 %ret = hask.caseint %ihash 
                 [0 -> { ^entry(%_: !hask.value): 
                           %v = hask.make_i64(42)
                           %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                           hask.return (%boxed): !hask.adt<@SimpleInt>
                 }]
                 }]
                 [@default -> { ^entry:
                            %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                            %onehash = hask.make_i64(1)
                            %prev = hask.primop_sub(%ihash, %onehash)
                            %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                            // ELIMINTATED: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                            // changed to call to f2 
                            %fprev_v = hask.apEager(%f2: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, 
                                %box_prev_v) 
                            hask.return(%fprev_v): !hask.adt<@SimpleInt>
                 }]
                 hask.return(%ret): !hask.adt<@SimpleInt>
           }]
    hask.return (%ret):!hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}

```

#### Stage 4: since `f2` immediately `case`s its argument, unbox argument to function `f3`

```

hask.func @f3 {
%lam = hask.lambda(%ihash: !hask.value) {
     %ret = hask.caseint %ihash 
     [0 -> { ^entry(%_: !hask.value): 
               %v = hask.make_i64(42)
               %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
               hask.return (%boxed): !hask.adt<@SimpleInt>
     }]
     [@default -> { ^entry:
                %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                %onehash = hask.make_i64(1)
                %prev = hask.primop_sub(%ihash, %onehash)
                %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                %fprev_v = hask.apEager(%f2: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, 
                    %box_prev_t) 
                hask.return(%fprev_v): !hask.adt<@SimpleInt>
     }]
     hask.return(%ret): !hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@SimpleInt>>
}


hask.func @f2{
...
}

hask.func @f{
...
}
```

#### Stage 5: change calls of `f2(construct(@SimpleInt, int))` to `f3(int)`


```
hask.func @f3 {
%lam = hask.lambda(%ihash: !hask.value) {
     %ret = hask.caseint %ihash 
     [0 -> { ^entry(%_: !hask.value): 
               %v = hask.make_i64(42)
               %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
               hask.return (%boxed): !hask.adt<@SimpleInt>
     }]
     [@default -> { ^entry:
                %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                %onehash = hask.make_i64(1)
                %prev = hask.primop_sub(%ihash, %onehash)
                // ELIMINTATED: %box_prev_v = hask.construct(@SimpleInt, %prev: !hask.value) : !hask.adt<@SimpleInt>
                // CALL CHANGED TO f3
                %fprev_v = hask.apEager(%f3: !hask.value) -> !hask.adt<@SimpleInt>>, %prev) 
                hask.return(%fprev_v): !hask.adt<@SimpleInt>
     }]
     hask.return(%ret): !hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@SimpleInt>>
}

hask.func @f2{
%lam = hask.lambda(%icons: !hask.adt<@SimpleInt>) {
    %ret = hask.case @SimpleInt %icons
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
                 %ret = hask.caseint %ihash 
                 [0 -> { ^entry(%_: !hask.value): 
                           %v = hask.make_i64(42)
                           %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt> 
                           hask.return (%boxed): !hask.adt<@SimpleInt>
                 }]
                 }]
                 [@default -> { ^entry:
                            %f = hask.ref(@f):  !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                            %onehash = hask.make_i64(1)
                            %prev = hask.primop_sub(%ihash, %onehash)
                            // ELIMINTATED: %box_prev_t = hask.thunkify(%box_prev_v :!hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                            // CALL CHANGED TO f3
                            %fprev_v = hask.apEager(%f3: !hask.fn<(!hask.value) -> !hask.adt<@SimpleInt>>, 
                                %prev) 
                            hask.return(%prev_v): !hask.adt<@SimpleInt>
                 }]
                 hask.return(%ret): !hask.adt<@SimpleInt>
           }]
    hask.return (%ret):!hask.adt<@SimpleInt>
}
hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
}
```
