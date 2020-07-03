

module {
  %0 = hask.module {
    %1 = hask.make_data_constructor<"I#">
    %2 = hask.make_data_constructor<"GHC.Num.+">
    %3 = hask.make_data_constructor<"GHC.Num.-">
    %4 = hask.make_data_constructor<"GHC.Num.$fNumInt">
    %5 = hask.toplevel_binding {
      hask.lambda(%arg0) {
        hask.case {
          hask.return(%arg0)
        } {alt0 = "default", alt1 = 0 : i64, alt2 = 1 : i64} {
          hask.ap( {
            hask.return(%2)
          }, {
            hask.return(%4)
          }, {
            hask.ap( {
              hask.dominance_free_scope {
                hask.return(%5)
              }
            }, {
              hask.return(%arg0)
            })
          }, {
            hask.ap( {
              hask.return(%3)
            }, {
              hask.return(%4)
            }, {
              hask.return(%arg0)
            }, {
              hask.ap( {
                hask.return(%1)
              }, {
                %c0_i32 = constant 0 : i32
                hask.make_i32(%c0_i32)
              })
            })
          })
        } {
          hask.ap( {
            hask.return(%1)
          }, {
            %c0_i32 = constant 0 : i32
            hask.make_i32(%c0_i32)
          })
        } {
          hask.ap( {
            hask.return(%1)
          }, {
            %c0_i32 = constant 0 : i32
            hask.make_i32(%c0_i32)
          })
        }
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
