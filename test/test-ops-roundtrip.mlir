

module {
  %0 = hask.module {
    %1 = hask.make_data_constructor<"I#">
    %2 = hask.make_data_constructor<"GHC.Num.+">
    %3 = hask.make_data_constructor<"GHC.Num.-">
    %4 = hask.make_data_constructor<"GHC.Num.$fNumInt">
    %5 = hask.toplevel_binding {
      hask.lambda[%arg0] {
        %7 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: none):  // no predecessors
          hask.return(%arg0)
        }]

        hask.return(%7)
      }
    }
    %6 = hask.dummy_finish
  }
}
