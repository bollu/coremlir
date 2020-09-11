

module {
  hask.func @k {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk,%arg1:!hask.thunk) {
      hask.return(%arg0) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.func @loop {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.ref(@loop) : !hask.fn<!hask.thunk, !hask.thunk>
      %2 = hask.apSSA(%1 :!hask.fn<!hask.thunk, !hask.thunk>, %arg0)
      %3 = hask.force(%2 :!hask.thunk):!hask.thunk
      hask.return(%3) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
  hask.adt @X [#hask.data_constructor<@MkX []>]
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.make_i64(42 : i64)
      %2 = hask.construct(@X, %1)
      %3 = hask.ref(@k) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
      %4 = hask.ref(@loop) : !hask.fn<!hask.thunk, !hask.thunk>
      %5 = hask.apSSA(%4 :!hask.fn<!hask.thunk, !hask.thunk>, %2)
      %6 = hask.apSSA(%3 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %2, %5)
      %7 = hask.force(%6 :!hask.thunk):!hask.value
      hask.return(%7) : !hask.value
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.value>
  }
}