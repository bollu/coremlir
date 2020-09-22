

module {
  hask.func @main {
    %0 = hask.lambda(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.make_i64(43 : i64)
      %2 = hask.caseint %1 [0 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        hask.return(%arg1) : !hask.value
      }]
 [@default ->  {
        %4 = hask.make_i64(1 : i64)
        %5 = hask.primop_sub(%1,%4)
        hask.return(%5) : !hask.value
      }]

      %3 = hask.construct(@X, %2 : !hask.value) : !hask.adt<@X>
      hask.return(%3) : !hask.adt<@X>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@X>>
  }
}