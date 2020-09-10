module {
  // k x y = x
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

  hask.adt @X [#hask.data_constructor<@MkX []>]

  // k (x:X) (y:(loop X)) = x
  // main = 
  //     let y = loop x -- builds a closure.
  //     in k x y
  hask.func @main {
    %lambda = hask.lambdaSSA(%_: !hask.thunk) {
      %x = hask.construct(@X)
      %k = hask.ref(@k) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
      %loop = hask.ref(@loop) :  !hask.fn<!hask.thunk, !hask.thunk>
      %y = hask.apSSA(%loop : !hask.fn<!hask.thunk, !hask.thunk>, %x)
      %out_t = hask.apSSA(%k: !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %x, %y)
      %out = hask.force(%out_t)
      hask.return(%out) : !hask.value
    }
    hask.return(%lambda) :!hask.fn<!hask.thunk, !hask.value>
  }
}
