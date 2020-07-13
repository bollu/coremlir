

module {
  %0 = hask.module {
    hask.dominance_free_scope {
      %1 = hask.make_data_constructor<"I#">
      %2 = hask.make_data_constructor<"GHC.Num.+">
      %3 = hask.make_data_constructor<"GHC.Num.-">
      %4 = hask.make_data_constructor<"GHC.Num.$fNumInt">
      hask.func @foo {
        %7 = hask.lambdaSSA(%arg0) {
          %8 = hask.caseSSA %arg0 ["default" ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            %9 = hask.make_i32(1 : i64)
            %10 = hask.apSSA(%3,%arg0,%9)
            %11 = hask.make_string("fib_proxy")
            %12 = hask.apSSA(%11,%10)
            %13 = hask.caseSSA %12 ["default" ->  {
            ^bb0(%arg2: !hask.untyped):  // no predecessors
              %14 = hask.apSSA(%11,%arg0)
              %15 = hask.caseSSA %14 ["default" ->  {
              ^bb0(%arg3: !hask.untyped):  // no predecessors
                %17 = hask.apSSA(%2,%arg3)
                hask.return(%17)
              }]

              %16 = hask.apSSA(%15,%arg2)
              hask.return(%16)
            }]

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
