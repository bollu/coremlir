

module {
  hask.func @prec {
    %0 = hask.caseint %arg0 [0 : i64 ->  {
    ^bb0(%arg1: !hask.value):  // no predecessors
      hask.return(%arg1) : !hask.value
    }]
 [@default ->  {
      %1 = hask.make_i64(1 : i64)
      %2 = hask.primop_sub(%arg0,%1)
      hask.return(%2) : !hask.value
    }]

    hask.return(%0) : !hask.value
  }
  hask.func @main {
    %0 = hask.make_i64(42 : i64)
    %1 = hask.ref(@prec) : !hask.fn<(!hask.value) -> !hask.value>
    %2 = hask.ap(%1 :!hask.fn<(!hask.value) -> !hask.value>, %0)
    %3 = hask.force(%2):!hask.value
    %4 = hask.construct(@X, %3 : !hask.value) : !hask.adt<@X>
    hask.return(%4) : !hask.adt<@X>
  }
}