module {
  hask.func @k {
    %lambda = hask.lambdaSSA(%x: !hask.thunk, %y: !hask.thunk) {
      hask.return(%x) : !hask.thunk
    }
    hask.return(%lambda) :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }

  // loop a = loop a
  hask.func @loop {
    %lambda = hask.lambdaSSA(%a: !hask.thunk) {
      %loop = hask.ref(@loop) : !hask.fn<!hask.thunk, !hask.thunk>
      %out = hask.apSSA(%loop : !hask.fn<!hask.thunk, !hask.thunk>, %a)
      hask.return(%out) : !hask.thunk
    }
    hask.return(%lambda) : !hask.fn<!hask.thunk, !hask.thunk>
  }

  // hask.func @main {
  //   %lambda = hask.lambdaSSA(%_: !hask.thunk) {
  //     %x = hask.construct(@X)
  //     hask.return(%x): !hask.thunk
  //   }
  //   hask.return(%lambda):!hask.fn<!hask.thunk, !hask.thunk>
  // }
}
