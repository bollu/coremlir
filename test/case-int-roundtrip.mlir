

module {
  %c42_i64 = constant 42 : i64
  hask.func @prec {
    %0 = hask.lambda(%arg0:!hask.value) {
      %1 = hask.transmute(%arg0 :!hask.value):i64
      %2 = hask.caseint %1 [0 : i64 ->  {
      ^bb0(%arg1: i64):  // no predecessors
        %3 = hask.transmute(%1 :i64):!hask.value
        hask.return(%3) : !hask.value
      }]
 [@default ->  {
        %3 = subi %1, %c42_i64 : i64
        %4 = hask.transmute(%3 :i64):!hask.value
        hask.return(%4) : !hask.value
      }]

      hask.return(%2) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.value) -> !hask.value>
  }
  hask.func @main {
    %0 = hask.lambda(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.make_i64(42 : i64)
      %2 = hask.transmute(%1 :!hask.value):i64
      %3 = hask.caseint %2 [0 : i64 ->  {
      ^bb0(%arg1: i64):  // no predecessors
        %5 = hask.transmute(%2 :i64):!hask.value
        hask.return(%5) : !hask.value
      }]
 [@default ->  {
        %5 = subi %2, %c42_i64 : i64
        %6 = hask.transmute(%5 :i64):!hask.value
        hask.return(%6) : !hask.value
      }]

      %4 = hask.construct(@X, %3 : !hask.value) : !hask.adt<@X>
      hask.return(%4) : !hask.adt<@X>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@X>>
  }
}