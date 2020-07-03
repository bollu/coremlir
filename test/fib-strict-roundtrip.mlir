

module {
  %0 = hask.module {
    hask.dominance_free_scope {
      %1 = hask.make_data_constructor<"I#">
      %2 = hask.make_data_constructor<"GHC.Num.+">
      %3 = hask.make_data_constructor<"GHC.Num.-">
      %4 = hask.make_data_constructor<"GHC.Num.$fNumInt">
      %5 = hask.toplevel_binding {
        %8 = hask.lambdaSSA(%arg0) {
          %9 = hask.caseSSA %arg0 ["default" ->  {
          ^bb0(%arg1: none):  // no predecessors
            %c1_i32 = constant 1 : i32
            %10 = hask.constant(%c1_i32, i32)
            %11 = hask.apSSA(%3,%arg0,%10)
            %12 = hask.recursive_ref {
              hask.return(%5)
            }
            %13 = hask.apSSA(%12,%11)
            %14 = hask.caseSSA %13 ["default" ->  {
            ^bb0(%arg2: none):  // no predecessors
              %15 = hask.apSSA(%12,%arg0)
              %16 = hask.caseSSA %15 ["default" ->  {
              ^bb0(%arg3: none):  // no predecessors
                %18 = hask.apSSA(%2,%arg3)
                hask.return(%18)
              }]

              %17 = hask.apSSA(%16,%arg2)
              hask.return(%17)
            }]

            hask.return(%14)
          }]
 [0 : i64 ->  {
          ^bb0(%arg1: none):  // no predecessors
            hask.return(%arg0)
          }]
 [1 : i64 ->  {
          ^bb0(%arg1: none):  // no predecessors
            hask.return(%arg0)
          }]

          hask.return(%9)
        }
        hask.return(%8)
      }
      %6 = hask.toplevel_binding {
        hask.ap( {
          hask.return(%5)
        }, {
          %c10_i32 = constant 10 : i32
          hask.make_i32(%c10_i32)
        })
      }
      %7 = hask.dummy_finish
    }
  }
}
