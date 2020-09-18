

module {
  hask.adt @EitherBox [#hask.data_constructor<@Left [@Box]>, #hask.data_constructor<@Right [@Box]>]
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
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
      %2 = hask.make_i64(42 : i64)
      %3 = hask.construct(@SimpleInt, %2 : !hask.value) : !hask.adt<@SimpleInt>
      hask.return(%3) : !hask.adt<@SimpleInt>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@SimpleInt>>
  }
}