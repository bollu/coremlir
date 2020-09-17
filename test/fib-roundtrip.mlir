

module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>,%arg1:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.force(%arg0):!hask.adt<@SimpleInt>
      %2 = hask.caseSSA @SimpleInt %1 [@SimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1):!hask.adt<@SimpleInt>
        %4 = hask.caseSSA @SimpleInt %3 [@SimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_add(%arg2,%arg3)
          %6 = hask.construct(@SimpleInt, %5 : !hask.value)
          hask.return(%6) : !hask.adt<@SimpleInt>
        }]

        hask.return(%4) : !hask.adt<@SimpleInt>
      }]

      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
  hask.func @minus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>,%arg1:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.force(%arg0):!hask.adt<@SimpleInt>
      %2 = hask.caseSSA @SimpleInt %1 [@SimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1):!hask.adt<@SimpleInt>
        %4 = hask.caseSSA @SimpleInt %3 [@SimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_sub(%arg2,%arg3)
          %6 = hask.construct(@SimpleInt, %5 : !hask.value)
          hask.return(%6) : !hask.adt<@SimpleInt>
        }]

        hask.return(%4) : !hask.adt<@SimpleInt>
      }]

      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
  hask.func @zero {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(0 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value)
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @one {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(1 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value)
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @two {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(2 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value)
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @eight {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(8 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value)
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @fib {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.force(%arg0):!hask.adt<@SimpleInt>
      %2 = hask.caseSSA @SimpleInt %1 [@SimpleInt ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %3 = hask.caseint %arg1 [0 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@zero) : !hask.fn<() -> !hask.adt<@SimpleInt>>
          %5 = hask.apSSA(%4 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
          %6 = hask.force(%5):!hask.adt<@SimpleInt>
          hask.return(%6) : !hask.adt<@SimpleInt>
        }]
 [1 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
          %5 = hask.apSSA(%4 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
          %6 = hask.force(%5):!hask.adt<@SimpleInt>
          hask.return(%6) : !hask.adt<@SimpleInt>
        }]
 [@default ->  {
          %4 = hask.ref(@fib) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
          %5 = hask.ref(@minus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
          %6 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
          %7 = hask.apSSA(%6 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
          %8 = hask.apSSA(%5 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %arg0, %7)
          %9 = hask.apSSA(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %8)
          %10 = hask.force(%9):!hask.adt<@SimpleInt>
          %11 = hask.ref(@two) : !hask.fn<() -> !hask.adt<@SimpleInt>>
          %12 = hask.apSSA(%11 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
          %13 = hask.apSSA(%5 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %arg0, %12)
          %14 = hask.apSSA(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %13)
          %15 = hask.force(%14):!hask.adt<@SimpleInt>
          %16 = hask.thunkify(%10 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
          %17 = hask.thunkify(%15 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
          %18 = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
          %19 = hask.apSSA(%18 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %16, %17)
          %20 = hask.force(%19):!hask.adt<@SimpleInt>
          hask.return(%20) : !hask.adt<@SimpleInt>
        }]

        hask.return(%3) : !hask.adt<@SimpleInt>
      }]

      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.make_i64(6 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value)
      %3 = hask.thunkify(%2 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
      %4 = hask.ref(@fib) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %5 = hask.apSSA(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %3)
      %6 = hask.force(%5):!hask.adt<@SimpleInt>
      hask.return(%6) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
}