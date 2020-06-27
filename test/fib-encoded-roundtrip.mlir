

module {
  %0 =  {
    %1 = standalone.make_data_constructor<"I#">
    %2 = standalone.make_data_constructor<"GHC.Num.+">
    %3 = standalone.make_data_constructor<"GHC.Num.-">
    %4 = standalone.make_data_constructor<"GHC.Num.$fNumInt">
    %5 =  {
      standalone.lambda[%arg0] {
        standalone.case {
          standalone.return(%arg0)
        } {alt0 = "default", alt1 = 0 : i64, alt2 = 1 : i64} {
          standalone.ap( {
            standalone.return(%2)
          }, {
            standalone.return(%4)
          }, {
            standalone.ap( {
              standalone.dominance_free_scope {
                standalone.return(%5)
              }
            }, {
              standalone.return(%arg0)
            })
          }, {
            standalone.ap( {
              standalone.return(%3)
            }, {
              standalone.return(%4)
            }, {
              standalone.return(%arg0)
            }, {
              standalone.ap( {
                standalone.return(%1)
              }, {
                %c0_i32 = constant 0 : i32
                standalone.make_i32(%c0_i32)
              })
            })
          })
        } {
          standalone.ap( {
            standalone.return(%1)
          }, {
            %c0_i32 = constant 0 : i32
            standalone.make_i32(%c0_i32)
          })
        } {
          standalone.ap( {
            standalone.return(%1)
          }, {
            %c0_i32 = constant 0 : i32
            standalone.make_i32(%c0_i32)
          })
        }
      }
    }
    %6 =  {
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
