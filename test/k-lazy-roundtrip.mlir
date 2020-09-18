

module {
  hask.func @k {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.value>,%arg1:!hask.thunk<!hask.value>) {
      %1 = hask.force(%arg0):!hask.value
      hask.return(%1) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>
  }
  hask.func @loop {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.ref(@loop) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
      %2 = hask.apSSA(%1 :!hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, %arg0)
      %3 = hask.force(%2):!hask.value
      hask.return(%3) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
  }
  hask.adt @X [#hask.data_constructor<@MkX []>]
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.make_i64(42 : i64)
      %2 = hask.construct(@X, %1 : !hask.value) : !hask.adt<@X>
      %3 = hask.transmute(%2 :!hask.adt<@X>):!hask.value
      %4 = hask.thunkify(%3 :!hask.value):!hask.thunk<!hask.value>
      %5 = hask.ref(@loop) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
      %6 = hask.apSSA(%5 :!hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, %4)
      %7 = hask.ref(@k) : !hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>
      %8 = hask.apSSA(%7 :!hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>, %4, %6)
      %9 = hask.force(%8):!hask.value
      hask.return(%9) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
  }
}