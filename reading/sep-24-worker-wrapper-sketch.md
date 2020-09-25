# Worker-wrapper

## Revision 2, Example 2: `fib`: non-tail-recursive function


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
