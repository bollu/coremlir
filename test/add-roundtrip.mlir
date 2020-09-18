

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>,%arg1:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.force(%arg0):!hask.adt<@SimpleInt>
      %2 = hask.caseSSA @SimpleInt %1 [@SimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1):!hask.adt<@SimpleInt>
        %4 = hask.caseSSA @SimpleInt %3 [@SimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_add(%arg2,%arg3)
          %6 = hask.construct(@SimpleInt, %5 : !hask.value) : !hask.adt<@SimpleInt>
          hask.return(%6) : !hask.adt<@SimpleInt>
        }]

        hask.return(%4) : !hask.adt<@SimpleInt>
      }]

      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
  hask.func @one {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(1 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value) : !hask.adt<@SimpleInt>
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @two {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(2 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value) : !hask.adt<@SimpleInt>
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
      %2 = hask.apSSA(%1 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
      %3 = hask.ref(@two) : !hask.fn<() -> !hask.adt<@SimpleInt>>
      %4 = hask.apSSA(%3 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
      %5 = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %6 = hask.apSSA(%5 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %2, %4)
      %7 = hask.force(%6):!hask.adt<@SimpleInt>
      hask.return(%7) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@SimpleInt>>
  }
}