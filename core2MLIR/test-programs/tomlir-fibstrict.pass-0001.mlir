// Main
// Core2MLIR: GenMLIR AfterCorePrep
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @sat_s1wO {
  %lambda_0 = hask.lambdaSSA(%i_s1wH) {
    %case_1 = hask.caseSSA  %i_s1wH
    ["default" ->
    {
    ^entry(%ds_s1wI: !hask.untyped):
      %app_2 = hask.apSSA(%minus_hash, %i_s1wH)
      %lit_3 = hask.make_i32(1)
      %app_4 = hask.apSSA(%app_2, %lit_3)
      %sat_s1wJ = hask.force (%app_4)
      %app_6 = hask.apSSA(@fib, %sat_s1wJ)
      %wild_s1wK = hask.force (%app_6)
      %app_8 = hask.apSSA(@fib, %i_s1wH)
      %wild_s1wL = hask.force (%app_8)
      %unimpl_10  =  hask.make_i32(42)
      %sat_s1wN = hask.force (%unimpl_10)
      %app_12 = hask.apSSA(%sat_s1wN, %wild_s1wK)
    hask.return(%app_12)
    }
    ]
    [0 ->
    {
    ^entry(%ds_s1wI: !hask.untyped):
    hask.return(%i_s1wH)
    }
    ]
    [1 ->
    {
    ^entry(%ds_s1wI: !hask.untyped):
    hask.return(%i_s1wH)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @fib {
  hask.return(@sat_s1wO)
  }
hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 31, types: 10, coercions: 0, joins: 0/1}
//sat_s1wO
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> ghc-prim-0.5.3:GHC.Prim.Int#
//[LclId]
//sat_s1wO
//  = \ (i_s1wH :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      case i_s1wH of {
//        __DEFAULT ->
//          case ghc-prim-0.5.3:GHC.Prim.-# i_s1wH 1# of sat_s1wJ [Occ=Once]
//          { __DEFAULT ->
//          case main:Main.fib sat_s1wJ of wild_s1wK [Occ=Once] { __DEFAULT ->
//          case case main:Main.fib i_s1wH of wild_s1wL [Occ=OnceL]
//               { __DEFAULT ->
//               let {
//                 sat_s1wM [Occ=OnceT[0]]
//                   :: ghc-prim-0.5.3:GHC.Prim.Int# -> ghc-prim-0.5.3:GHC.Prim.Int#
//                 [LclId]
//                 sat_s1wM
//                   = \ (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//                       ghc-prim-0.5.3:GHC.Prim.+# wild_s1wL eta_B1 } in
//               sat_s1wM
//               }
//          of sat_s1wN [Occ=Once!]
//          { __DEFAULT ->
//          sat_s1wN wild_s1wK
//          }
//          }
//          };
//        0# -> i_s1wH;
//        1# -> i_s1wH
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Main.fib [Occ=LoopBreaker]
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> ghc-prim-0.5.3:GHC.Prim.Int#
//[LclId]
//main:Main.fib = sat_s1wO
//end Rec }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wR :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1wR = ghc-prim-0.5.3:GHC.Types.TrNameS "Main"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1wQ :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1wQ = ghc-prim-0.5.3:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:Main.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Main.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module sat_s1wQ sat_s1wR
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