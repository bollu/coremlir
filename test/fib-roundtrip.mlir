

module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]
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
  hask.func @minus {
  ^bb0(%arg0: !hask.thunk<!hask.adt<@SimpleInt>>, %arg1: !hask.thunk<!hask.adt<@SimpleInt>>):  // no predecessors
    %0 = hask.force(%arg0):!hask.adt<@SimpleInt>
    %1 = hask.case @SimpleInt %0 [@SimpleInt ->  {
    ^bb0(%arg2: !hask.value):  // no predecessors
      %2 = hask.force(%arg1):!hask.adt<@SimpleInt>
      %3 = hask.case @SimpleInt %2 [@SimpleInt ->  {
      ^bb0(%arg3: !hask.value):  // no predecessors
        %4 = hask.primop_sub(%arg2,%arg3)
        %5 = hask.construct(@SimpleInt, %4 : !hask.value) : !hask.adt<@SimpleInt>
        hask.return(%5) : !hask.adt<@SimpleInt>
      }]

      hask.return(%3) : !hask.adt<@SimpleInt>
    }]

    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @zero {
    %0 = hask.make_i64(0 : i64)
    %1 = hask.construct(@SimpleInt, %0 : !hask.value) : !hask.adt<@SimpleInt>
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
  hask.func @eight {
    %0 = hask.make_i64(8 : i64)
    %1 = hask.construct(@SimpleInt, %0 : !hask.value) : !hask.adt<@SimpleInt>
    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @fib {
  ^bb0(%arg0: !hask.thunk<!hask.adt<@SimpleInt>>):  // no predecessors
    %0 = hask.force(%arg0):!hask.adt<@SimpleInt>
    %1 = hask.case @SimpleInt %0 [@SimpleInt ->  {
    ^bb0(%arg1: !hask.value):  // no predecessors
      %2 = hask.caseint %arg1 [0 : i64 ->  {
        %3 = hask.ref(@zero) : !hask.fn<() -> !hask.adt<@SimpleInt>>
        %4 = hask.ap(%3 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
        %5 = hask.force(%4):!hask.adt<@SimpleInt>
        hask.return(%5) : !hask.adt<@SimpleInt>
      }]
 [1 : i64 ->  {
        %3 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
        %4 = hask.ap(%3 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
        %5 = hask.force(%4):!hask.adt<@SimpleInt>
        hask.return(%5) : !hask.adt<@SimpleInt>
      }]
 [@default ->  {
        %3 = hask.ref(@fib) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
        %4 = hask.ref(@minus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
        %5 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
        %6 = hask.ap(%5 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
        %7 = hask.ap(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %arg0, %6)
        %8 = hask.ap(%3 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %7)
        %9 = hask.force(%8):!hask.adt<@SimpleInt>
        %10 = hask.ref(@two) : !hask.fn<() -> !hask.adt<@SimpleInt>>
        %11 = hask.ap(%10 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
        %12 = hask.ap(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %arg0, %11)
        %13 = hask.ap(%3 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %12)
        %14 = hask.force(%13):!hask.adt<@SimpleInt>
        %15 = hask.thunkify(%9 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
        %16 = hask.thunkify(%14 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
        %17 = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
        %18 = hask.ap(%17 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %15, %16)
        %19 = hask.force(%18):!hask.adt<@SimpleInt>
        hask.return(%19) : !hask.adt<@SimpleInt>
      }]

      hask.return(%2) : !hask.adt<@SimpleInt>
    }]

    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @main {
    %0 = hask.make_i64(6 : i64)
    %1 = hask.construct(@SimpleInt, %0 : !hask.value) : !hask.adt<@SimpleInt>
    %2 = hask.thunkify(%1 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
    %3 = hask.ref(@fib) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
    %4 = hask.ap(%3 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %2)
    %5 = hask.force(%4):!hask.adt<@SimpleInt>
    hask.return(%5) : !hask.adt<@SimpleInt>
  }
}