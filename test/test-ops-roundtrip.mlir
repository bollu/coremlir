

module {
  hask.module {
    %0 = hask.make_data_constructor<"I#">
    %1 = hask.make_data_constructor<"GHC.Num.+">
    %2 = hask.make_data_constructor<"GHC.Num.-">
    %3 = hask.make_data_constructor<"GHC.Num.$fNumInt">
    hask.func @fib {
      hask.lambda(%arg0) {
        %4 = hask.apSSA(@fib, %arg0)
        %5 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: none):  // no predecessors
          hask.return(%arg0)
        }]

        hask.return(%5)
      }
    }
    hask.func @function {
      %4 = hask.make_i32(1 : i64)
      hask.return(%4)
    }
    hask.dummy_finish
  }
}