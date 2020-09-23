

module {
  hask.func @prec {
    %0 = hask.lambda(%arg0:!hask.value) {
      %1 = hask.caseint %arg0 [0 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        hask.return(%arg1) : !hask.value
      }]
 [@default ->  {
        %2 = hask.make_i64(1 : i64)
        %3 = hask.primop_sub(%arg0,%2)
        hask.return(%3) : !hask.value
      }]

      hask.return(%1) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.value) -> !hask.value>
  }
  hask.func @main {
    %0 = hask.lambda(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.make_i64(42 : i64)
      %2 = hask.caseint %1 [0 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        hask.return(%arg1) : !hask.value
      }]
 [@default ->  {
        %3 = hask.make_i64(1 : i64)
        %4 = hask.primop_sub(%1,%3)
        hask.return(%4) : !hask.value
      }]

      hask.return(%2) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
  }
}