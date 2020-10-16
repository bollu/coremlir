// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// Check that @plus works with SimpleInt works.
// CHECK: constructor(SimpleInt 42)
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]

  // f :: SimpleInt -> SimpleInt
  // f i = case i of SimpleInt i# -> 
  //          case i# of 
  //            0 -> SimpleInt 5; 
  //            _ -> case f ( SimpleInt(i# -# 1#)) of
  //                  SimpleInt j# -> SimpleInt (j# +# 1)
  hask.func @f (%i : !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt> {
      %icons = hask.force(%i): !hask.adt<@SimpleInt>
      %reti = hask.case @SimpleInt %icons 
           [@SimpleInt -> { ^entry(%ihash: !hask.value):
              %retj = hask.caseint %ihash
                  [0 -> {
                        %five = hask.make_i64(5)
                        %boxed = hask.construct(@SimpleInt, %five:!hask.value): !hask.adt<@SimpleInt>
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
                        // TODO: should `case` be a terminator?
                        %out = hask.case @SimpleInt %rec_v 
                          [@SimpleInt -> { ^entry(%jhash: !hask.value):
                              %one_j = hask.make_i64(1)
                              %jincr = hask.primop_add(%jhash, %one_j)
                              %boxed_jincr = hask.construct(@SimpleInt, %jincr: !hask.value): !hask.adt<@SimpleInt>
                              hask.return(%boxed_jincr): !hask.adt<@SimpleInt>
                          }]
                        hask.return(%out): !hask.adt<@SimpleInt>
                  }]
              hask.return(%retj):!hask.adt<@SimpleInt>
           }]
      hask.return(%reti): !hask.adt<@SimpleInt>
    }

  // 37 + 5 = 42
  hask.func@main () -> !hask.adt<@SimpleInt> {
      %v = hask.make_i64(37)
      %v_box = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt>
      %v_thunk = hask.thunkify(%v_box: !hask.adt<@SimpleInt>): !hask.thunk<!hask.adt<@SimpleInt>>
      %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %out_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %v_thunk)
      %out_v = hask.force(%out_t): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
}

