

module {
  %0 = hask.module {
    hask.dominance_free_scope {
      %1 = hask.make_data_constructor<"I#">
      %2 = hask.make_data_constructor<"GHC.Num.+">
      %3 = hask.make_data_constructor<"GHC.Num.-">
      %4 = hask.make_data_constructor<"GHC.Num.$fNumInt">
      hask.func @fib {
        %7 = hask.lambdaSSA(%arg0) {
          %8 = hask.caseSSA %arg0 ["default" ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            %9 = hask.make_i32(1 : i64)
            %10 = hask.apSSA(%3,%arg0,%9)
            %11 = hask.apSSA(@fib)
            %12 = hask.force(%11)
            %13 = hask.apSSA(@fib)
            %14 = hask.force(%13)
            %15 = hask.apSSA(%2,%14)
            %16 = hask.copy(%15)
            %17 = hask.apSSA(%16,%12)
            hask.return(%17)
          }]
 [0 : i64 ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            hask.return(%arg0)
          }]
 [1 : i64 ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            hask.return(%arg0)
          }]

          hask.return(%8)
        }
        hask.return(%7)
      }
      %5 = hask.toplevel_binding {
        %7 = hask.make_i32(10 : i64)
        %8 = hask.make_string("%fib(%ten)")
        hask.return(%8)
      }
      %6 = hask.dummy_finish
    }
  }
}
