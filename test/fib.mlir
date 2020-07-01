// fib :: Int -> Int
// fib i = case i of
//           0 -> 0
//           1 -> 1
//           n -> fib n + fib (n - 1)
// main :: IO (); main = print (fib 10)
// module Main where
// 

// Pretty printed core from `ghc-dump-core` [output directly from `desugar`]:
// =========================================================================
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



// ==================== Desugar (after optimization) ====================
// 2020-07-01 23:19:58.257881266 UTC
// 
// Result size of Desugar (after optimization)
//   = {terms: 39, types: 15, coercions: 0, joins: 0/0}
// 
// -- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
// Main.$trModule :: GHC.Types.Module
// [LclIdX]
// Main.$trModule
//   = GHC.Types.Module
//       (GHC.Types.TrNameS "main"#) (GHC.Types.TrNameS "Main"#)
// 
// Rec {
// -- RHS size: {terms: 23, types: 6, coercions: 0, joins: 0/0}
// fib [Occ=LoopBreaker] :: Int -> Int
// [LclId]
// fib
//   = \ (i_a11V :: Int) ->
//       case i_a11V of { GHC.Types.I# ds_d2PQ ->
//       case ds_d2PQ of {
//         __DEFAULT ->
//           + @ Int
//             GHC.Num.$fNumInt
//             (fib i_a11V)
//             (fib (- @ Int GHC.Num.$fNumInt i_a11V (GHC.Types.I# 1#)));
//         0# -> GHC.Types.I# 0#;
//         1# -> GHC.Types.I# 1#
//       }
//       }
// end Rec }
// 
// -- RHS size: {terms: 5, types: 1, coercions: 0, joins: 0/0}
// main :: IO ()
// [LclIdX]
// main = print @ Int GHC.Show.$fShowInt (fib (GHC.Types.I# 10#))
// 
// -- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
// :Main.main :: IO ()
// [LclIdX]
// :Main.main = GHC.TopHandler.runMainIO @ () main


