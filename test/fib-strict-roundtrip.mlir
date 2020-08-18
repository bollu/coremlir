

module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    hask.func @fib {
      %3 = hask.lambdaSSA(%arg0) {
        %4 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          %5 = hask.make_i32(1 : i64)
          %6 = hask.apSSA(%1, %arg0, %5)
          %7 = hask.apSSA(@fib, %6)
          %8 = hask.force(%7)
          %9 = hask.apSSA(@fib, %arg0)
          %10 = hask.force(%9)
          %11 = hask.apSSA(%0, %10, %8)
          hask.return(%11)
        }]
 [0 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]
 [1 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]

        hask.return(%4)
      }
      hask.return(%3)
    }
    hask.dummy_finish
  }
}