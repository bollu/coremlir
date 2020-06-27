// module Main where
// 
// $trModule :: Module
// 
// {- Core Size{terms=5 types=0 cos=0 vbinds=0 jbinds=0} -}
// $trModule =
//   APP(GHC.Types.Module
//     (APP(GHC.Types.TrNameS "main"#))
//     (APP(GHC.Types.TrNameS
//        "Main"#)))
// 
// rec {
// fib :: Int -> Int
// 
// {- Core Size{terms=23 types=6 cos=0 vbinds=0 jbinds=0} -}
// fib =
//   λ i →
//     case i of wild {
//       I# ds →
//         case ds of ds {
//           DEFAULT →
//             APP(GHC.Num.+
//               @Int
//               GHC.Num.$fNumInt
//               (APP(Main.fib i))
//               (APP(Main.fib
//                  (APP(GHC.Num.-
//                     @Int
//                     GHC.Num.$fNumInt
//                     i
//                     (APP(GHC.Types.I# 1#)))))))
//           0# → APP(GHC.Types.I# 0#)
//           1# → APP(GHC.Types.I# 1#)
//         }
//     }
// }
// main :: IO ()
// 
// {- Core Size{terms=6 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(System.IO.putStrLn
//     (APP(GHC.Show.show
//        @Int
//        GHC.Show.$fShowInt
//        (APP(Main.fib
//           (APP(GHC.Types.I# 10#)))))))
// 
// main :: IO ()
// 
// {- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(GHC.TopHandler.runMainIO
//     @() Main.main)


standalone.module { 
  // standalone.dominance_free_scope {

    %constructor_ihash  = standalone.make_data_constructor<"I#"> 
    // This is kind of a lie, we should call it as inbuilt fn or whatever.
    %constructor_plus = standalone.make_data_constructor<"GHC.Num.+"> 
    %constructor_minus = standalone.make_data_constructor<"GHC.Num.-"> 
    %value_dict_num_int = standalone.make_data_constructor<"GHC.Num.$fNumInt">

    // The syntax that I encoded into the parser
    // %fib :: Int -> Int
    %fib = standalone.toplevel_binding {  
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
                            { standalone.ap({  standalone.dominance_free_scope { standalone.return (%fib) } },  {standalone.return (%i)}) }, //(APP(Main.fib i))
                            // { standalone.ap({ standalone.return (%constructor_plus) },  {standalone.return (%i)}) }, // FOR TESTING WITHOUT RECURSION!: (APP(Main.fib i))
                            {   //APP(GHC.Num.- ...
                                standalone.ap({ standalone.return (%constructor_minus)}, // (APP(GHC.Num.-
                                                  { standalone.return(%value_dict_num_int) }, //GHC.Num.$fNumInt
                                                  { standalone.return(%i) }, // i
                                                  { standalone.ap({ standalone.return(%constructor_ihash)}, { %c0 = constant 0 : i32 standalone.make_i32(%c0)}) }  // (APP(GHC.Types.I# 1#)))))))
  
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
  

    %main = standalone.toplevel_binding { 
      standalone.ap ({ standalone.return (%fib) }, {%c10 = constant 10 : i32 standalone.make_i32(%c10)}) 
    }

    // need to add dummy terminator, FFS.
    // standalone.dummy_finish
    // %cNONE = standalone.make_data_constructor<"DUMMY_RETURN_PLEASE_DONT_BE_A_PETULANT_CHILD">
    // standalone.return(%cNONE)
    standalone.dummy_finish
  // } // end dominance_free_scope

  // standalone.dummy_finish
} // end module