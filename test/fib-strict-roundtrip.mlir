

module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    hask.func @fib {
      %3 = hask.lambdaSSA(%arg0) {
        %4 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          %5 = hask.apSSA(%1, %arg0)
          %6 = hask.make_i32(1 : i64)
          %7 = hask.apSSA(%5, %6)
          %8 = hask.apSSA(@fib, %7)
          %9 = hask.force(%8)
          %10 = hask.apSSA(@fib, %arg0)
          %11 = hask.force(%10)
          %12 = hask.apSSA(%0, %11)
          %13 = hask.apSSA(%12, %9)
          hask.return(%13)
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