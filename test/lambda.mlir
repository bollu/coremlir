// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// Check that @plus works with SimpleInt works.
// CHECK: constructor(SimpleInt 3)
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]

  // plus :: SimpleInt -> SimpleInt -> SimpleInt
  // plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)
  hask.func @plus (%i : !hask.thunk<!hask.adt<@SimpleInt>>, %j: !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt> {
      %icons = hask.force(%i): !hask.adt<@SimpleInt>
      %reti = hask.case @SimpleInt %icons 
           [@SimpleInt -> { ^entry(%ival: !hask.value):
              %jcons = hask.force(%j):!hask.adt<@SimpleInt>
              %retj = hask.case @SimpleInt %jcons 
                  [@SimpleInt -> { ^entry(%jval: !hask.value):
                        %sum_v = hask.primop_add(%ival, %jval)
                        %boxed = hask.construct(@SimpleInt, %sum_v:!hask.value): !hask.adt<@SimpleInt>
                        hask.return(%boxed) : !hask.adt<@SimpleInt>
                  }]
              hask.return(%retj):!hask.adt<@SimpleInt>
           }]
      hask.return(%reti): !hask.adt<@SimpleInt>
    }

  
  
  hask.func @one () -> !hask.adt<@SimpleInt> {
       %v = hask.make_i64(1)
       %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt>
       hask.return(%boxed): !hask.adt<@SimpleInt>
     }


  hask.func @two () -> !hask.adt<@SimpleInt> {
       %v = hask.make_i64(2)
       %boxed = hask.construct(@SimpleInt, %v:!hask.value): !hask.adt<@SimpleInt>
       hask.return(%boxed): !hask.adt<@SimpleInt>
     }


  // 1 + 2 = 3
  hask.func@main () -> !hask.adt<@SimpleInt> {
      %input = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
      %input_t = hask.ap(%input: !hask.fn<() -> !hask.adt<@SimpleInt>>)

      %input2 = hask.ref(@two) :!hask.fn<() -> !hask.adt<@SimpleInt>>
      %input2_t = hask.ap(%input2 : !hask.fn<() -> !hask.adt<@SimpleInt>>)

      %plus = hask.ref(@plus)  : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) ->  !hask.adt<@SimpleInt>>
      %out_t = hask.ap(%plus : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, 
        %input_t, %input2_t)
      %out_v = hask.force(%out_t): !hask.adt<@SimpleInt>
      hask.return(%out_v) : !hask.adt<@SimpleInt>
    }
}


