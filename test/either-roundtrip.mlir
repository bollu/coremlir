

module {
  hask.adt @EitherBox [#hask.data_constructor<@Left [@Box]>, #hask.data_constructor<@Right [@Box]>]
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @extract {
    %0 = hask.force(%arg0):!hask.adt<@Either>
    %1 = hask.case @Either %0 [@Right ->  {
    ^bb0(%arg1: !hask.thunk<!hask.adt<@Either>>):  // no predecessors
      %2 = hask.force(%arg1):!hask.adt<@Either>
      %3 = hask.case @Either %2 [@Left ->  {
      ^bb0(%arg2: !hask.thunk<!hask.value>):  // no predecessors
        %4 = hask.force(%arg2):!hask.value
        hask.return(%4) : !hask.value
      }]

      hask.return(%3) : !hask.value
    }]

    hask.return(%1) : !hask.value
  }
  hask.func @one {
    %0 = hask.make_i64(1 : i64)
    %1 = hask.construct(@SimpleInt, %0 : !hask.value) : !hask.adt<@SimpleInt>
    hask.return(%1) : !hask.adt<@SimpleInt>
  }
  hask.func @leftOne {
    %0 = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
    %1 = hask.ap(%0 :!hask.fn<() -> !hask.adt<@SimpleInt>>)
    %2 = hask.construct(@Left, %1 : !hask.thunk<!hask.adt<@SimpleInt>>) : !hask.adt<@Either>
    hask.return(%2) : !hask.adt<@Either>
  }
  hask.func @rightLeftOne {
    %0 = hask.ref(@leftOne) : !hask.fn<() -> !hask.adt<@Either>>
    %1 = hask.ap(%0 :!hask.fn<() -> !hask.adt<@Either>>)
    %2 = hask.construct(@Right, %1 : !hask.thunk<!hask.adt<@Either>>) : !hask.adt<@Either>
    hask.return(%2) : !hask.adt<@Either>
  }
  hask.func @main {
    %0 = hask.ref(@rightLeftOne) : !hask.fn<() -> !hask.adt<@Either>>
    %1 = hask.ap(%0 :!hask.fn<() -> !hask.adt<@Either>>)
    %2 = hask.ref(@extract) : !hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>
    %3 = hask.ap(%2 :!hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>, %1)
    %4 = hask.force(%3):!hask.value
    hask.return(%4) : !hask.value
  }
}