

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
  ^bb0(%arg0: !hask.thunk<!hask.adt<@SimpleInt>>, %arg1: !hask.thunk<!hask.adt<@SimpleInt>>):  // no predecessors
    %0 = hask.force(%arg0):!hask.adt<@SimpleInt>
    %1 = hask.case @SimpleInt %0 [@SimpleInt ->  {
    ^bb0(%arg2: !hask.value):  // no predecessors
      %2 = hask.force(%arg1):!hask.adt<@SimpleInt>
      %3 = hask.case @SimpleInt %2 [@SimpleInt ->  {
      ^bb0(%arg3: !hask.value):  // no predecessors
        %4 = hask.primop_add(%arg2,%arg3)
        %5 = hask.construct(@SimpleInt, %4 : !hask.value) : !hask.adt<@SimpleInt>
        hask.return(%5) : !hask.adt<@SimpleInt>
      }]

      hask.return(%3) : !hask.adt<@SimpleInt>
    }]

    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @one {
    %0 = hask.make_i64(1 : i64)
    %1 = hask.construct(@SimpleInt, %0 : !hask.value) : !hask.adt<@SimpleInt>
    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @two {
    %0 = hask.make_i64(2 : i64)
    %1 = hask.construct(@SimpleInt, %0 : !hask.value) : !hask.adt<@SimpleInt>
    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @main {
    %0 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
    %1 = hask.ap(%0 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
    %2 = hask.ref(@two) : !hask.fn<() -> !hask.adt<@SimpleInt>>
    %3 = hask.ap(%2 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
    %4 = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
    %5 = hask.ap(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %1, %3)
    %6 = hask.force(%5):!hask.adt<@SimpleInt>
    hask.return(%6) : !hask.adt<@SimpleInt>
  }
}