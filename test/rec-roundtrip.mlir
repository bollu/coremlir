

module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]
  hask.func @f {
    %0 = hask.lambda(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.force(%arg0):!hask.adt<@SimpleInt>
      %2 = hask.case @SimpleInt %1 [@SimpleInt ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %3 = hask.caseint %arg1 [0 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.make_i64(42 : i64)
          %5 = hask.construct(@SimpleInt, %4 : !hask.value) : !hask.adt<@SimpleInt>
          hask.return(%5) : !hask.adt<@SimpleInt>
        }]
 [@default ->  {
          %4 = hask.ref(@f) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
          %5 = hask.make_i64(1 : i64)
          %6 = hask.primop_sub(%arg1,%5)
          %7 = hask.construct(@SimpleInt, %6 : !hask.value) : !hask.adt<@SimpleInt>
          %8 = hask.thunkify(%7 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
          %9 = hask.ap(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %8)
          %10 = hask.force(%9):!hask.adt<@SimpleInt>
          hask.return(%10) : !hask.adt<@SimpleInt>
        }]

        hask.return(%3) : !hask.adt<@SimpleInt>
      }]

      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
  hask.func @main {
    %0 = hask.lambda(%arg0:!hask.thunk<!hask.adt<@SimpleInt>>) {
      %1 = hask.make_i64(6 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value) : !hask.adt<@SimpleInt>
      %3 = hask.thunkify(%2 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
      %4 = hask.ref(@f) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
      %5 = hask.ap(%4 :!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, %3)
      %6 = hask.force(%5):!hask.adt<@SimpleInt>
      hask.return(%6) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
  }
}