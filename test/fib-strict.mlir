// Source
// ======================================================================

// {-# LANGUAGE MagicHash #-}
// import GHC.Prim
// -- printRawInt = undefined
// fib :: Int# -> Int#
// fib i = case i of
//         0# ->  i
//         1# ->  i
//         _ ->  (fib i) +# (fib (i -# 1#))
// 
// 
// main :: IO ();
// main = let x = fib 10# in return ()


// Pretty printed core from `ghc-dump-core` [output directly from `desugar`]:
// =========================================================================
// {- desugar -}
// module Main where
// 
// rec {
// fib :: Int# -> Int#
// 
// {- Core Size{terms=21 types=4 cos=0 vbinds=0 jbinds=0} -}
// fib =
//   λ i →
//     case i of ds {
//       DEFAULT →
//         case APP(Main.fib
//                (APP(GHC.Prim.-# i 1#)))
//         of wild {
//           DEFAULT →
//             APP((case APP(Main.fib i)
//               of wild {
//                 DEFAULT →
//                   APP(GHC.Prim.+# wild)
//               })
//               wild)
//         }
//       0# → i
//       1# → i
//     }
// }
// $trModule :: Module
// 
// {- Core Size{terms=5 types=0 cos=0 vbinds=0 jbinds=0} -}
// $trModule =
//   APP(GHC.Types.Module
//     (APP(GHC.Types.TrNameS "main"#))
//     (APP(GHC.Types.TrNameS
//        "Main"#)))
// 
// $krep :: KindRep
// 
// {- Core Size{terms=3 types=1 cos=0 vbinds=0 jbinds=0} -}
// $krep =
//   APP(GHC.Types.KindRepTyConApp
//     GHC.Types.$tcInt#
//     (APP(GHC.Types.[] @KindRep)))
// 
// $tcInt :: TyCon
// 
// {- Core Size{terms=8 types=0 cos=0 vbinds=0 jbinds=0} -}
// $tcInt =
//   APP(GHC.Types.TyCon
//     11974157267237989633#
//     13669022426829607813#
//     Main.$trModule
//     (APP(GHC.Types.TrNameS "Int"#))
//     0#
//     GHC.Types.krep$*)
// 
// $krep :: KindRep
// 
// {- Core Size{terms=3 types=1 cos=0 vbinds=0 jbinds=0} -}
// $krep =
//   APP(GHC.Types.KindRepTyConApp
//     Main.$tcInt
//     (APP(GHC.Types.[] @KindRep)))
// 
// $krep :: KindRep
// 
// {- Core Size{terms=3 types=0 cos=0 vbinds=0 jbinds=0} -}
// $krep =
//   APP(GHC.Types.KindRepFun
//     $krep $krep)
// 
// $tc'IntConstructor :: TyCon
// 
// {- Core Size{terms=8 types=0 cos=0 vbinds=0 jbinds=0} -}
// $tc'IntConstructor =
//   APP(GHC.Types.TyCon
//     1317530741244836087#
//     13258383807356744379#
//     Main.$trModule
//     (APP(GHC.Types.TrNameS
//        "'IntConstructor"#))
//     0#
//     $krep)
// 
// main :: IO ()
// 
// {- Core Size{terms=7 types=3 cos=0 vbinds=0 jbinds=0} -}
// main =
//   case APP(Main.fib 10#) of x {
//     DEFAULT →
//       APP(GHC.Base.return
//         @IO
//         GHC.Base.$fMonadIO
//         @()
//         GHC.Tuple.())
//   }
// 
// main :: IO ()
// 
// {- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(GHC.TopHandler.runMainIO
//     @() Main.main)



// copy-pasted just fib from prettycore for easy reference
// ============================================================================
// fib =
//   λ i →
//     case i of ds {
//       DEFAULT →
//         case APP(Main.fib
//                (APP(GHC.Prim.-# i 1#)))
//         of wild {
//           DEFAULT →
//             APP((case APP(Main.fib i)
//               of wild {
//                 DEFAULT →
//                   APP(GHC.Prim.+# wild)
//               })
//               wild)
//         }
//       0# → i
//       1# → i
//     }
// }


hask.module { 
  hask.dominance_free_scope {

    %constructor_ihash  = hask.make_data_constructor<"I#"> 
    // This is kind of a lie, we should call it as inbuilt fn or whatever.
    %constructor_plus = hask.make_data_constructor<"GHC.Num.+"> 
    %constructor_minus = hask.make_data_constructor<"GHC.Num.-"> 
    %value_dict_num_int = hask.make_data_constructor<"GHC.Num.$fNumInt">

    // The syntax that I encoded into the parser
    // fib =
    //   λ i →
     //     case i of ds {
    // %fib :: Int -> Int
    %fib = hask.toplevel_binding  {  
      hask.lambda [%i] {
        %resulttop = hask.caseSSA %i 
            ["default" -> { //default
                    // DEFAULT →
                    //   case APP(Main.fib
                    //          (APP(GHC.Prim.-# i 1#)))
                    //   of wild {
                    //     DEFAULT →
                    //       APP((case APP(Main.fib i)
                    //         of wild {
                    //           DEFAULT →
                    //             APP(GHC.Prim.+# wild)
                    //         })
                    //         wild)
                    ^entry(%ds: none):
                      %one = constant 1 : i32
                      %core_one = hask.constant(%one, i32)
                      %i_minus_one = hask.apSSA(%constructor_minus, %i, %core_one)
                      // TODO: It is annoying that I need to define it like this; Is there
                      // _really_ no nicer way? If so, that just seems sad. 
                      %fib_proxy = hask.recursive_ref { hask.return(%fib) }
                      %fib_i_minus_one = hask.apSSA(%fib_proxy, %i_minus_one)
                      // hask.return(%fib_i_minus_one)
                      %result = hask.caseSSA %fib_i_minus_one
                                ["default" -> { //default
                                         //     DEFAULT →
                                         //       APP((case APP(Main.fib i)
                                         //         of wild {
                                         //           DEFAULT →
                                         //             APP(GHC.Prim.+# wild)
                                         //         })
                                         //         wild)
                                       ^entry(%wild: none):
                                         %fib_i = hask.apSSA(%fib_proxy, %i)
                                         %add_wild_fn = hask.caseSSA %fib_i
                                                               ["default" -> { //default
                                                                    // DEFAULT →
                                                                   //             APP(GHC.Prim.+# wild)
                                                                   ^entry(%wild_inner: none):
                                                                     %plus_wild_inner = hask.apSSA(%constructor_plus, %wild_inner)
                                                                     hask.return(%plus_wild_inner)
                                                               }]
                                         %result = hask.apSSA(%add_wild_fn, %wild)
                                         hask.return(%result)
                                     }]
                    hask.return(%result)
                  }]
                  [0 -> { // 0 ->
                    ^entry(%ds: none):
                    hask.return (%i)
                  }]
                  [1 -> { // 1 -> 
                    ^entry(%ds: none):
                    hask.return (%i)
                  }]
      hask.return(%resulttop)
      } //end lambda
  } // end fib
  

    %main = hask.toplevel_binding { 
      hask.ap ({ hask.return (%fib) }, {%c10 = constant 10 : i32 hask.make_i32(%c10)}) 
    }

    // need to add dummy terminator, FFS.
    // hask.dummy_finish
    // %cNONE = hask.make_data_constructor<"DUMMY_RETURN_PLEASE_DONT_BE_A_PETULANT_CHILD">
    // hask.return(%cNONE)
    hask.dummy_finish
  } // end dominance_free_scope

  // hask.dummy_finish
} // end module



//
// Core:
// =========================================================================
//
// ==================== Desugar (after optimization) ====================
// 2020-07-01 23:11:01.516881827 UTC
// 
// Result size of Desugar (after optimization)
//   = {terms: 69, types: 22, coercions: 0, joins: 0/0}
// 
// Rec {
// -- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
// fib [Occ=LoopBreaker] :: Int# -> Int#
// [LclId]
// fib
//   = \ (i_a11X :: Int#) ->
//       case i_a11X of {
//         __DEFAULT ->
//           case fib (-# i_a11X 1#) of wild_00 { __DEFAULT ->
//           (case fib i_a11X of wild_X5 { __DEFAULT -> +# wild_X5 }) wild_00
//           };
//         0# -> i_a11X;
//         1# -> i_a11X
//       }
// end Rec }
// 
// -- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
// Main.$trModule :: GHC.Types.Module
// [LclIdX]
// Main.$trModule
//   = GHC.Types.Module
//       (GHC.Types.TrNameS "main"#) (GHC.Types.TrNameS "Main"#)
// 
// -- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
// $krep_a1jC [InlPrag=NOUSERINLINE[~]] :: GHC.Types.KindRep
// [LclId]
// $krep_a1jC
//   = GHC.Types.KindRepTyConApp
//       GHC.Types.$tcInt# (GHC.Types.[] @ GHC.Types.KindRep)
// 
// -- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
// Main.$tcInt :: GHC.Types.TyCon
// [LclIdX]
// Main.$tcInt
//   = GHC.Types.TyCon
//       11974157267237989633##
//       13669022426829607813##
//       Main.$trModule
//       (GHC.Types.TrNameS "Int"#)
//       0#
//       GHC.Types.krep$*
// 
// -- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
// $krep_a1jD [InlPrag=NOUSERINLINE[~]] :: GHC.Types.KindRep
// [LclId]
// $krep_a1jD
//   = GHC.Types.KindRepTyConApp
//       Main.$tcInt (GHC.Types.[] @ GHC.Types.KindRep)
// 
// -- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
// $krep_a1jB [InlPrag=NOUSERINLINE[~]] :: GHC.Types.KindRep
// [LclId]
// $krep_a1jB = GHC.Types.KindRepFun $krep_a1jC $krep_a1jD
// 
// -- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
// Main.$tc'IntConstructor :: GHC.Types.TyCon
// [LclIdX]
// Main.$tc'IntConstructor
//   = GHC.Types.TyCon
//       1317530741244836087##
//       13258383807356744379##
//       Main.$trModule
//       (GHC.Types.TrNameS "'IntConstructor"#)
//       0#
//       $krep_a1jB
// 
// -- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
// main :: IO ()
// [LclIdX]
// main
//   = case fib 10# of { __DEFAULT ->
//     return @ IO GHC.Base.$fMonadIO @ () GHC.Tuple.()
//     }
// 
// -- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
// :Main.main :: IO ()
// [LclIdX]
// :Main.main = GHC.TopHandler.runMainIO @ () main