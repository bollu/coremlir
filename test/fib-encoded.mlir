"custom_op" () ({ 
  standalone.dominance_free_scope {
    // The syntax that I encoded into the parser
    // %fib :: Int -> Int
    %fib = standalone.toplevel_binding {
        %constructor_ihash  = standalone.make_data_constructor<"I#"> 
        // This is kind of a lie, we should call it as inbuilt fn or whatever.
        %constructor_plus = standalone.make_data_constructor<"GHC.Num.+"> 
        %constructor_minus = standalone.make_data_constructor<"GHC.Num.-"> 
        %value_dict_num_int = standalone.make_data_constructor<"GHC.Num.$fNumInt">
  
      standalone.lambda [%i] {
        standalone.case  {standalone.return(%i)} { alt0 = "default", alt1=0, alt2=1 }
        { //default
  
          // APP(GHC.Num.+
          //   @Int
          //   GHC.Num.$fNumInt
          //   (APP(Main.fib i))
          //   (APP(Main.fib
          //      (APP(GHC.Num.-
          //         @Int
          //         GHC.Num.$fNumInt
          //         i
          //         (APP(GHC.Types.I# 1#)))))))
  
  
          standalone.ap({ standalone.return (%constructor_plus) }, // GHC.Num.+
                            { standalone.return (%value_dict_num_int) }, // GHC.Num.$fNumInt
                            { standalone.ap({ standalone.dominance_free_scope { standalone.return (%fib) } },  {standalone.return (%i)}) }, //(APP(Main.fib i))
                            // { standalone.ap({ standalone.return (%constructor_plus) },  {standalone.return (%i)}) }, // FOR TESTING WITHOUT RECURSION!: (APP(Main.fib i))
                            {   //APP(GHC.Num.- ...
                                standalone.ap({ standalone.return (%constructor_minus)}, // (APP(GHC.Num.-
                                                  { standalone.return(%value_dict_num_int) }, //GHC.Num.$fNumInt
                                                  { standalone.return(%i) }, // i
                                                  { standalone.ap({ standalone.return(%constructor_ihash)}, { %c1 = constant 0 : i32 standalone.make_i32(%c1)}) }  // (APP(GHC.Types.I# 1#)))))))
  
                                )
                            })
        }
        { // 0 -> 
            standalone.ap(
                { standalone.return (%constructor_ihash) },
                { %c0 = constant 0 : i32 standalone.make_i32 (%c0) }) 
        }
        { // 1 -> 
            standalone.ap(
                { standalone.return (%constructor_ihash) },
                { %c1 = constant 0 : i32 standalone.make_i32 (%c1) }) 
  
        }
  
      }
    } //end fib
  
    // need to add dummy terminator, FFS.
    %cNONE = standalone.make_data_constructor<"DUMMY_RETURN_PLEASE_DONT_BE_A_PETULANT_CHILD">
    standalone.return(%cNONE)
  } // end dominance_free_scope
}) : () -> ()