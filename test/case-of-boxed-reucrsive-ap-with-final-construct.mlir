// The common hask.construct(...) should be peeled out of  the `case`, 
// leaving us with the control flow to decide on the `int` followed by a boxing.
// TODO: really we want *anything* that is common in control flow to be 
//        pulled out.
// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s  -worker-wrapper | FileCheck %s -check-prefix='CHECK-WW-IR'
// RUN: ../build/bin/hask-opt %s  -interpret -worker-wrapper | FileCheck %s -check-prefix='CHECK-WW-OUTPUT'
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// Check that @plus works with SimpleInt works.
// CHECK-WW-IR: %1 = hask.case @SimpleInt %0 
// CHECK-WW-IR: %2 = hask.construct(@Just, %1 : !hask.adt<@SimpleInt>) : !hask.adt<@SimpleInt>
// CHECK-WW-OUTPUT: constructor(Just 0)
// CHECK: constructor(Just 0)



module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @SimpleInt [#hask.data_constructor<@Just [@"Int#"]>, #hask.data_constructor<@Nothing []>]

  // f :: SimpleInt -> SimpleInt
  // f mi = case si of SimpleInt n -> case n of 0 -> SimpleInt 42; _ -> f (SimpleInt (ihash - 1))
  hask.func @f (%sn: !hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt> {
      %reti = hask.case @SimpleInt %sn 
                  [@SimpleInt ->  {
                     ^entry(%n: !hask.value):

                     %v = hask.caseint %n
                      [0 : i64 -> {
                        %forty_two = hask.make_i64(42)
                        %forty_two_si = hask.construct(@SimpleInt, %forty_two: !hask.value) : !hask.adt<@SimpleInt>
                        hask.return(%forty_two_si): !hask.adt<@SimpleInt>
                      }]

                      [@default ->  {
                         %one = hask.make_i64(1)
                         %n_minus_1 = hask.primop_sub(%n, %one)
                         %si_n_minus_1 = hask.construct(@SimpleInt, %n_minus_1: !hask.value) : !hask.adt<@SimpleInt>
                         %f = hask.ref(@f): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
                         %ret = hask.apEager(%f : !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>> , %si_n_minus_1)
                         hask.return(%ret):!hask.adt<@SimpleInt>
                      }]

                     hask.return(%v) : !hask.adt<@SimpleInt>
                  }]

      hask.return (%reti): !hask.adt<@SimpleInt>
    }


    hask.func @main() -> !hask.adt<@SimpleInt> {
        %three = hask.make_i64(3)
        %n = hask.construct(@SimpleInt, %three:!hask.value) : !hask.adt<@SimpleInt>
        %f = hask.ref(@f): !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>
        %out = hask.apEager(%f: !hask.fn<(!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>>, %n)
        hask.return(%out) : !hask.adt<@SimpleInt>
    }
}

