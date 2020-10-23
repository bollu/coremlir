// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s  -worker-wrapper -interpret | FileCheck %s --check-prefix=CHECK-WW 
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true

// Check that @plus works with SimpleInt works.
// CHECK: constructor(SimpleInt 42)
// CHECK: num_thunkify_calls(6)
// CHECK: num_force_calls(12)
// CHECK: num_construct_calls(7)

// CHECK-WW: constructor(SimpleInt 42)
// CHECK-WW: num_thunkify_calls(0)
// CHECK-WW: num_force_calls(0)
// CHECK-WW: num_construct_calls(1)
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]

  // f :: SimpleInt -> SimpleInt
  // f i = case i of SimpleInt i# -> case i# of 0 -> SimpleInt 42; _ -> f ( SimpleInt(i# -# 1#))
  hask.func @f (%i : !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt> {
      %icons = hask.force(%i): !hask.adt<@SimpleInt>
      %reti = hask.case @SimpleInt %icons 
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
              %retj = hask.caseint %ihash
                  [0 -> {
                        %fortytwo = hask.make_i64(42)
                        %boxed = hask.construct(@SimpleInt, %fortytwo:!hask.value): !hask.adt<@SimpleInt>
                        hask.return(%boxed) : !hask.adt<@SimpleInt>
                  }]
                  [@default ->  {
                        %one = hask.make_i64(1)
                        %isub = hask.primop_sub(%ihash, %one)
                        %boxed_isub = hask.construct(@SimpleInt, %isub: !hask.value): !hask.adt<@SimpleInt>
                        %boxed_isub_t = hask.thunkify(%boxed_isub : !hask.adt<@SimpleInt>) : !hask.thunk<!hask.adt<@SimpleInt>>
                        %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
                        %rec_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>> , %boxed_isub_t)
                        %rec_v = hask.force(%rec_t): !hask.adt<@SimpleInt> 
                        hask.return(%rec_v): !hask.adt<@SimpleInt>
                  }]
              hask.return(%retj):!hask.adt<@SimpleInt>
           }]
      hask.return(%reti): !hask.adt<@SimpleInt>
    }

  // 1 + 2 = 3
  hask.func@main () -> !hask.adt<@SimpleInt> {
      %v = hask.make_i64(5)
      %v_box = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt>
      %v_thunk = hask.thunkify(%v_box: !hask.adt<@SimpleInt>): !hask.thunk<!hask.adt<@SimpleInt>>
      %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %out_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %v_thunk)
      %out_v = hask.force(%out_t): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
}

