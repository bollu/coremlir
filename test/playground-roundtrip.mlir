

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
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.construct(@X)
      hask.return(%1) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
}