

module {
  %0 = standalone.module {
    %1 = standalone.make_data_constructor<"I#">
    %2 = standalone.make_data_constructor<"GHC.Num.+">
    %3 = standalone.make_data_constructor<"GHC.Num.-">
    %4 = standalone.make_data_constructor<"GHC.Num.$fNumInt">
    %5 = standalone.toplevel_binding {
      standalone.lambda[%arg0] {
        %8 = standalone.caseSSA %arg0 {alt0 = "default", alt1 = 0 : i64, alt2 = 1 : i64} {
        ^bb0(%arg1: !core.return):  // no predecessors
          %c1_i32 = constant 1 : i32
          %9 = standalone.constant(%c1_i32, i32)
          %10 = standalone.apSSA(%3,%arg0,%9)
          %11 = standalone.constant(%c1_i32, i32)
          %12 = standalone.apSSA(%11,%10)
          %13 = standalone.caseSSA %12 {alt0 = "default"} {
          ^bb0(%arg2: none):  // no predecessors
            %14 = standalone.apSSA(%11,%arg0)
            %15 = standalone.caseSSA %14 {alt0 = "default"} {
            ^bb0(%arg3: none):  // no predecessors
              %17 = standalone.apSSA(%2,%arg3)
              standalone.return(%17)
            }
            %16 = standalone.apSSA(%15,%arg2)
            standalone.return(%16)
          }
          standalone.return(%13)
        } {
        ^bb0(%arg1: !core.return):  // no predecessors
          standalone.return(%arg0)
        } {
        ^bb0(%arg1: !core.return):  // no predecessors
          standalone.return(%arg0)
        }
        standalone.return(%8)
      }
    }
    %6 = standalone.toplevel_binding {
      standalone.ap( {
        standalone.return(%5)
      }, {
        %c10_i32 = constant 10 : i32
        standalone.make_i32(%c10_i32)
      })
    }
    %7 = standalone.dummy_finish
  }
}
