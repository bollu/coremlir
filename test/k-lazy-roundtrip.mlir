

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
      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
  hask.adt @X [#hask.data_constructor<@MkX []>]
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.construct(@X)
      %2 = hask.ref(@k) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
      %3 = hask.ref(@loop) : !hask.fn<!hask.thunk, !hask.thunk>
      %4 = hask.apSSA(%3 :!hask.fn<!hask.thunk, !hask.thunk>, %1)
      %5 = hask.apSSA(%2 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %1, %4)
      %6 = hask.force(%5)
      hask.return(%6) : !hask.value
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.value>
  }
}