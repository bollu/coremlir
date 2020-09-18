

module {
  hask.adt @EitherBox [#hask.data_constructor<@Left [@Box]>, #hask.data_constructor<@Right [@Box]>]
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @extract {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.adt<@Either>>) {
      %1 = hask.force(%arg0):!hask.adt<@Either>
      %2 = hask.caseSSA @Either %1 [@Right ->  {
      ^bb0(%arg1: !hask.thunk<!hask.adt<@Either>>):  // no predecessors
        %3 = hask.force(%arg1):!hask.adt<@Either>
        %4 = hask.caseSSA @Either %3 [@Right ->  {
        ^bb0(%arg2: !hask.thunk<!hask.value>):  // no predecessors
          %5 = hask.force(%arg2):!hask.value
          hask.return(%5) : !hask.value
        }]

        hask.return(%4) : !hask.value
      }]

      hask.return(%2) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>
  }
  hask.func @one {
    %0 = hask.lambdaSSA() {
      %1 = hask.make_i64(1 : i64)
      %2 = hask.construct(@SimpleInt, %1 : !hask.value) : !hask.adt<@SimpleInt>
      hask.return(%2) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@SimpleInt>>
  }
  hask.func @leftOne {
    %0 = hask.lambdaSSA() {
      %1 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
      %2 = hask.apSSA(%1 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
      %3 = hask.construct(@Left, %2 : !hask.thunk<!hask.adt<@SimpleInt>>) : !hask.adt<@Either>
      hask.return(%3) : !hask.adt<@Either>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@Either>>
  }
  hask.func @rightLeftOne {
    %0 = hask.lambdaSSA() {
      %1 = hask.ref(@leftOne) : !hask.fn<() -> !hask.adt<@Either>>
      %2 = hask.apSSA(%1 :!hask.fn<() -> !hask.adt<@Either>>)
      %3 = hask.construct(@Right, %2 : !hask.thunk<!hask.adt<@Either>>) : !hask.adt<@Either>
      %4 = hask.thunkify(%3 :!hask.adt<@Either>):!hask.thunk<!hask.adt<@Either>>
      hask.return(%3) : !hask.adt<@Either>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@Either>>
  }
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.ref(@rightLeftOne) : !hask.fn<() -> !hask.adt<@Either>>
      %2 = hask.apSSA(%1 :!hask.fn<() -> !hask.adt<@Either>>)
      %3 = hask.ref(@extract) : !hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>
      %4 = hask.apSSA(%3 :!hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>, %2)
      %5 = hask.force(%4):!hask.value
      hask.return(%5) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
  }
}