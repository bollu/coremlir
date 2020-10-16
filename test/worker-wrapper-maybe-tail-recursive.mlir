// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// Check that @plus works with Maybe works.
// CHECK: constructor(Just 42)
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @Maybe [#hask.data_constructor<@Just [@"Int#"]>, #hask.data_constructor<@Nothing []>]

  // f :: Maybe -> Maybe
  // f i = case i of Maybe i# -> case i# of 0 -> Maybe 42; _ -> f ( Maybe(i# -# 1#))
  hask.func @f (%i : !hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe> {
      %icons = hask.force(%i): !hask.adt<@Maybe>
      %reti = hask.case @Maybe %icons 
           [@Nothing -> {
              %nothing = hask.construct(@Nothing): !hask.adt<@Maybe>
              hask.return (%nothing):!hask.adt<@Maybe>
           }
           [@Just -> { ^entry(%ihash: !hask.value):
              %retj = hask.caseint %ihash
                  [0 -> {
                        %fortytwo = hask.make_i64(42)
                        %boxed = hask.construct(@Just, %fortytwo:!hask.value): !hask.adt<@Maybe>
                        hask.return(%boxed) : !hask.adt<@Maybe>
                  }]
                  [@default ->  {
                        %one = hask.make_i64(1)
                        %isub = hask.primop_sub(%ihash, %one)
                        %boxed_isub = hask.construct(@Just, %isub: !hask.value): !hask.adt<@Maybe>
                        %boxed_isub_t = hask.thunkify(%boxed_isub : !hask.adt<@Maybe>) : !hask.thunk<!hask.adt<@Maybe>>
                        %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>
                        %rec_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>> , %boxed_isub_t)
                        %rec_v = hask.force(%rec_t): !hask.adt<@Maybe> 
                        hask.return(%rec_v): !hask.adt<@Maybe>
                  }]
              hask.return(%retj):!hask.adt<@Maybe>
           }]
      hask.return(%reti): !hask.adt<@Maybe>
    }

  // 1 + 2 = 3
  hask.func@main () -> !hask.adt<@Maybe> {
      %v = hask.make_i64(5)
      %v_box = hask.construct(@Just, %v:!hask.value): !hask.adt<@Maybe>
      %v_thunk = hask.thunkify(%v_box: !hask.adt<@Maybe>): !hask.thunk<!hask.adt<@Maybe>>
      %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>
      %out_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>, %v_thunk)
      %out_v = hask.force(%out_t): !hask.adt<@Maybe>
      hask.return(%out_v) : !hask.adt<@Maybe>
    }
}

