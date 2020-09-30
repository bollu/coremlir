// A simple recursive function: f(Int 0) = Int 42; f(Int x) = f(x-1)
// We need to worker wrapper optimise this into:
// f(Int y) = Int (g# y)
// g# 0 = 1; g# x = g (x - 1) -- g# is strict.
// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// CHECK: constructor(SimpleInt 42)
module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]

  hask.func @f{
    %lam = hask.lambda(%i: !hask.thunk<!hask.adt<@SimpleInt>>) {
        %icons = hask.force(%i):!hask.adt<@SimpleInt>
        %ihash = hask.defaultcase(@SimpleInt, %icons) : !hask.value
        %ret = hask.caseint %ihash 
            [0 -> { ^entry: 
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
                       %prev_v = hask.force(%fprev_t): !hask.adt<@SimpleInt>
                       hask.return(%prev_v): !hask.adt<@SimpleInt>
            }]
        hask.return (%ret):!hask.adt<@SimpleInt>
    }
    hask.return (%lam): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }


  hask.func@main {
    %lam = hask.lambda() {
      %n = hask.make_i64(6)
      %box_n_v = hask.construct(@SimpleInt, %n: !hask.value): !hask.adt<@SimpleInt> 
      %box_n_t = hask.thunkify(%box_n_v: !hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
      %f = hask.ref(@f)  : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %out_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) ->  !hask.adt<@SimpleInt>>, %box_n_t)
      %out_v = hask.force(%out_t): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
    hask.return (%lam) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
    
}
