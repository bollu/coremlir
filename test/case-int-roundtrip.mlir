

module {
  hask.func @prec {
    %0 = hask.lambdaSSA(%arg0:!hask.value) {
      %1 = hask.caseint %arg0 [0 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        hask.return(%arg1) : !hask.value
      }]
 [@default ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %2 = hask.make_i64(1 : i64)
        %3 = hask.primop_sub(%arg1,%2)
        hask.return(%3) : !hask.value
      }]

      hask.return(%1) : !hask.value
    }
    hask.return(%0) : !hask.fn<!hask.value, !hask.value>
  }
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.make_i64(42 : i64)
      %2 = hask.ref(@prec) : !hask.fn<!hask.value, !hask.value>
      %3 = hask.apSSA(%2 :!hask.fn<!hask.value, !hask.value>, %1)
      %4 = hask.force(%3 :!hask.value):!hask.value
      %5 = hask.construct(@X, %4)
      hask.return(%5) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
}