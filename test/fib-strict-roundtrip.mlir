

module {
  %0 = hask.module {
    %1 = hask.make_data_constructor<"+#">
    %2 = hask.make_data_constructor<"-#">
    %3 = hask.make_data_constructor<"()">
    hask.func @fib {
      %5 = hask.lambdaSSA(%arg0) {
        %6 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          %7 = hask.apSSA(%2,%arg0)
          %8 = hask.make_i32(1 : i64)
          %9 = hask.apSSA(%7,%8)
          %10 = hask.apSSA(@fib)
          %11 = hask.force(%10)
          %12 = hask.apSSA(@fib)
          %13 = hask.force(%12)
          %14 = hask.apSSA(%1,%13)
          %15 = hask.apSSA(%14,%11)
          hask.return(%15)
        }]
 [0 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]
 [1 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]

        hask.return(%6)
      }
      hask.return(%5)
    }
    %4 = hask.dummy_finish
  }
}