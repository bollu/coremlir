

module {
  %0 = hask.module {
    hask.dominance_free_scope {
      %1 = hask.make_data_constructor<"I#">
      %2 = hask.make_data_constructor<"GHC.Num.+">
      %3 = hask.make_data_constructor<"GHC.Num.-">
      %4 = hask.make_data_constructor<"GHC.Num.$fNumInt">
      %5 = hask.toplevel_binding {
        hask.lambda[%arg0] {
          %8 = hask.caseSSA %arg0 {alt0 = "default", alt1 = 0 : i64, alt2 = 1 : i64} {
          ^bb0(%arg1: none):  // no predecessors
            %c1_i32 = constant 1 : i32
            %9 = hask.constant(%c1_i32, i32)
            %10 = hask.apSSA(%3,%arg0,%9)
            %11 = hask.recursive_ref {
              hask.return(%5)
            }
            %12 = hask.apSSA(%11,%10)
            %13 = hask.caseSSA %12 {alt0 = "default"} {
            ^bb0(%arg2: none):  // no predecessors
              %14 = hask.apSSA(%11,%arg0)
              %15 = hask.caseSSA %14 {alt0 = "default"} {
              ^bb0(%arg3: none):  // no predecessors
                %17 = hask.apSSA(%2,%arg3)
                hask.return(%17)
              }
              %16 = hask.apSSA(%15,%arg2)
              hask.return(%16)
            }
            hask.return(%13)
          } {
          ^bb0(%arg1: none):  // no predecessors
            hask.return(%arg0)
          } {
          ^bb0(%arg1: none):  // no predecessors
            hask.return(%arg0)
          }
          hask.return(%8)
        }
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
