// The common hask.construct(...) should be peeled out of  the `case`, 
// leaving us with the control flow to decide on the `int` followed by a boxing.
// TODO: really we want *anything* that is common in control flow to be 
//        pulled out.
// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s  -worker-wrapper | FileCheck %s -check-prefix='CHECK-WW-IR'
// RUN: ../build/bin/hask-opt %s  -interpret -worker-wrapper | FileCheck %s -check-prefix='CHECK-WW-OUTPUT'
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// Check that @plus works with Maybe works.
// CHECK-WW-IR: %1 = hask.case @Maybe %0 
// CHECK-WW-IR: %2 = hask.construct(@Just, %1 : !hask.adt<@Maybe>) : !hask.adt<@Maybe>
// CHECK-WW-OUTPUT: constructor(Just 0)
// CHECK: constructor(Just 0)



module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @Maybe [#hask.data_constructor<@Just [@"Int#"]>, #hask.data_constructor<@Nothing []>]

  // f :: Maybe -> Maybe
  // f mi = case i of Maybe _ -> Maybe 1; Nothing -> Maybe 0;
  hask.func @f (%i : !hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe> {
      %icons = hask.force(%i): !hask.adt<@Maybe>
      %reti = hask.case @Maybe %icons 
           [@Nothing -> {
              %zero = hask.make_i64(0)
              %just0 = hask.construct(@Just, %zero: !hask.value) : !hask.adt<@Maybe>
              hask.return (%just0):!hask.adt<@Maybe>
           }]
           [@Just -> { ^entry(%_: !hask.value):
              %one = hask.make_i64(1)
              %just1 = hask.construct(@Just, %one: !hask.value): !hask.adt<@Maybe>
              hask.return (%just1):!hask.adt<@Maybe>
           }]
      hask.return(%reti): !hask.adt<@Maybe>
    }


    hask.func @main() -> !hask.adt<@Maybe> {
        %n = hask.construct(@Nothing) : !hask.adt<@Maybe>
        %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>
        %nt = hask.thunkify(%n : !hask.adt<@Maybe>) :!hask.thunk<!hask.adt<@Maybe>>
        %out = hask.apEager(%f: !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>, %nt)
        hask.return(%out) : !hask.adt<@Maybe>
    }
}

