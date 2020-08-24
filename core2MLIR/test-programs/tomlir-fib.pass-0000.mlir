// Main
// Core2MLIR: GenMLIR BeforeCorePrep
module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
  hask.func @fib {
  %lambda_0 = hask.lambdaSSA(%i_a12E) {
    %case_1 = hask.caseSSA  %i_a12E
    [DATACONSTRUCTOR ->
    {
    ^entry(%wild_00: !hask.untyped, %ds_d2Qx: !hask.untyped):
      %case_2 = hask.caseSSA  %ds_d2Qx
      ["default" ->
      {
      ^entry(%ds_X2QD: !hask.untyped):
        %type_3 = hask.make_string("TYPEINFO_ERASED")
        %app_4 = hask.apSSA(%+, %type_3)
        %app_5 = hask.apSSA(%app_4, %$fNumInt)
        %app_6 = hask.apSSA(@fib, %i_a12E)
        %app_7 = hask.apSSA(%app_5, %app_6)
        %type_8 = hask.make_string("TYPEINFO_ERASED")
        %app_9 = hask.apSSA(%-, %type_8)
        %app_10 = hask.apSSA(%app_9, %$fNumInt)
        %app_11 = hask.apSSA(%app_10, %i_a12E)
        %lit_12 = hask.make_i64(1)
        %app_13 = hask.apSSA(%I#, %lit_12)
        %app_14 = hask.apSSA(%app_11, %app_13)
        %app_15 = hask.apSSA(@fib, %app_14)
        %app_16 = hask.apSSA(%app_7, %app_15)
      hask.return(%app_16)
      }
      ]
      [0 ->
      {
      ^entry(%ds_X2QD: !hask.untyped):
        %lit_17 = hask.make_i64(0)
        %app_18 = hask.apSSA(%I#, %lit_17)
      hask.return(%app_18)
      }
      ]
      [1 ->
      {
      ^entry(%ds_X2QD: !hask.untyped):
        %lit_19 = hask.make_i64(1)
        %app_20 = hask.apSSA(%I#, %lit_19)
      hask.return(%app_20)
      }
      ]
    hask.return(%case_2)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:Main.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Main.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "Main"#)
//
//Rec {
//-- RHS size: {terms: 23, types: 6, coercions: 0, joins: 0/0}
//main:Main.fib [Occ=LoopBreaker]
//  :: ghc-prim-0.5.3:GHC.Types.Int -> ghc-prim-0.5.3:GHC.Types.Int
//[LclId]
//main:Main.fib
//  = \ (i_a12E :: ghc-prim-0.5.3:GHC.Types.Int) ->
//      case i_a12E of { ghc-prim-0.5.3:GHC.Types.I# ds_d2Qx ->
//      case ds_d2Qx of {
//        __DEFAULT ->
//          base-4.12.0.0:GHC.Num.+
//            @ ghc-prim-0.5.3:GHC.Types.Int
//            base-4.12.0.0:GHC.Num.$fNumInt
//            (main:Main.fib i_a12E)
//            (main:Main.fib
//               (base-4.12.0.0:GHC.Num.-
//                  @ ghc-prim-0.5.3:GHC.Types.Int
//                  base-4.12.0.0:GHC.Num.$fNumInt
//                  i_a12E
//                  (ghc-prim-0.5.3:GHC.Types.I# 1#)));
//        0# -> ghc-prim-0.5.3:GHC.Types.I# 0#;
//        1# -> ghc-prim-0.5.3:GHC.Types.I# 1#
//      }
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 1, coercions: 0, joins: 0/0}
//main:Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:Main.main
//  = base-4.12.0.0:System.IO.print
//      @ ghc-prim-0.5.3:GHC.Types.Int
//      base-4.12.0.0:GHC.Show.$fShowInt
//      (main:Main.fib (ghc-prim-0.5.3:GHC.Types.I# 10#))
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//main::Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main::Main.main
//  = base-4.12.0.0:GHC.TopHandler.runMainIO @ () main:Main.main
//