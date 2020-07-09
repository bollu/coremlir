// Main
// Core2MLIR: GenMLIR BeforeCorePrep
hask.module {
  hask.recursive_ref {
    %fib = hask.toplevel_binding {
             %lambda_10 =
               hask.lambdaSSA(%i_a12E) {
                 %case_0 =
                   hask.caseSSA %i_a12E
                   ["default" ->
                     { ^entry(%ds_d1jZ: none):
                         %app_0  =  hask.apSSA(%minus_hash, %i_a12E)
                         %lit_1  =  hask.make_i32(1)
                         %app_2  =  hask.apSSA(%app_0, %lit_1)
                         %app_3  =  hask.apSSA(%fib, %app_2)
                         %case_4 =
                           hask.caseSSA %app_3
                           ["default" ->
                             { ^entry(%wild_00: none):
                                 %app_4  =  hask.apSSA(%fib, %i_a12E)
                                 %case_5 =
                                   hask.caseSSA %app_4
                                   ["default" ->
                                     { ^entry(%wild_X5: none):
                                         %app_5  =  hask.apSSA(%plus_hash, %wild_X5)
                                         hask.return(%app_5)
                                     }]
                                 %app_7  =  hask.apSSA(%case_5, %wild_00)
                                 hask.return(%app_7)
                             }]
                         hask.return(%case_4)
                     }]
                   [0 ->
                     { ^entry(%ds_d1jZ: none):
                         hask.return(%i_a12E)
                     }]
                   [1 ->
                     { ^entry(%ds_d1jZ: none):
                         hask.return(%i_a12E)
                     }]
                 hask.return(%case_0)
               }
             hask.return(%lambda_10)
           }
  }
  hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
//main:Main.fib [Occ=LoopBreaker]
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> ghc-prim-0.5.3:GHC.Prim.Int#
//[LclId]
//main:Main.fib
//  = \ (i_a12E :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      case i_a12E of {
//        __DEFAULT ->
//          case main:Main.fib (ghc-prim-0.5.3:GHC.Prim.-# i_a12E 1#)
//          of wild_00
//          { __DEFAULT ->
//          (case main:Main.fib i_a12E of wild_X5 { __DEFAULT ->
//           ghc-prim-0.5.3:GHC.Prim.+# wild_X5
//           })
//            wild_00
//          };
//        0# -> i_a12E;
//        1# -> i_a12E
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:Main.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Main.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "Main"#)
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//main:Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:Main.main
//  = case main:Main.fib 10# of { __DEFAULT ->
//    base-4.12.0.0:GHC.Base.return
//      @ ghc-prim-0.5.3:GHC.Types.IO
//      base-4.12.0.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.5.3:GHC.Tuple.()
//    }
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//main::Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main::Main.main
//  = base-4.12.0.0:GHC.TopHandler.runMainIO @ () main:Main.main
//