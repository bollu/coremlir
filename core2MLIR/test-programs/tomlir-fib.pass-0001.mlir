// Main
// Core2MLIR: GenMLIR AfterCorePrep
module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
  hask.func @sat_s36u {
  %lambda_0 = hask.lambdaSSA(%i_s36f) {
    %case_1 = hask.caseSSA  %i_s36f
    [DATACONSTRUCTOR ->
    {
    ^entry(%wild_s36g: !hask.untyped, %ds_s36h: !hask.untyped):
      %case_2 = hask.caseSSA  %ds_s36h
      ["default" ->
      {
      ^entry(%ds_s36i: !hask.untyped):
        %unimpl_3  =  hask.make_i32(42)
      hask.return(%unimpl_3)
      }
      ]
      [0 ->
      {
      ^entry(%ds_s36i: !hask.untyped):
        %lit_4 = hask.make_i64(0)
        %app_5 = hask.apSSA(%I#, %lit_4)
      hask.return(%app_5)
      }
      ]
      [1 ->
      {
      ^entry(%ds_s36i: !hask.untyped):
        %lit_6 = hask.make_i64(1)
        %app_7 = hask.apSSA(%I#, %lit_6)
      hask.return(%app_7)
      }
      ]
    hask.return(%case_2)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @fib {
  hask.return(@sat_s36u)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s36d :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s36d = ghc-prim-0.5.3:GHC.Types.TrNameS "Main"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s36c :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s36c = ghc-prim-0.5.3:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:Main.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Main.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module sat_s36c sat_s36d
//
//Rec {
//-- RHS size: {terms: 31, types: 10, coercions: 0, joins: 0/4}
//sat_s36u
//  :: ghc-prim-0.5.3:GHC.Types.Int -> ghc-prim-0.5.3:GHC.Types.Int
//[LclId]
//sat_s36u
//  = \ (i_s36f :: ghc-prim-0.5.3:GHC.Types.Int) ->
//      case i_s36f of { ghc-prim-0.5.3:GHC.Types.I# ds_s36h [Occ=Once!] ->
//      case ds_s36h of {
//        __DEFAULT ->
//          let {
//            sat_s36t [Occ=Once] :: ghc-prim-0.5.3:GHC.Types.Int
//            [LclId]
//            sat_s36t
//              = let {
//                  sat_s36s [Occ=Once] :: ghc-prim-0.5.3:GHC.Types.Int
//                  [LclId]
//                  sat_s36s
//                    = let {
//                        sat_s36r [Occ=Once] :: ghc-prim-0.5.3:GHC.Types.Int
//                        [LclId]
//                        sat_s36r = ghc-prim-0.5.3:GHC.Types.I# 1# } in
//                      base-4.12.0.0:GHC.Num.-
//                        @ ghc-prim-0.5.3:GHC.Types.Int
//                        base-4.12.0.0:GHC.Num.$fNumInt
//                        i_s36f
//                        sat_s36r } in
//                main:Main.fib sat_s36s } in
//          let {
//            sat_s36q [Occ=Once] :: ghc-prim-0.5.3:GHC.Types.Int
//            [LclId]
//            sat_s36q = main:Main.fib i_s36f } in
//          base-4.12.0.0:GHC.Num.+
//            @ ghc-prim-0.5.3:GHC.Types.Int
//            base-4.12.0.0:GHC.Num.$fNumInt
//            sat_s36q
//            sat_s36t;
//        0# -> ghc-prim-0.5.3:GHC.Types.I# 0#;
//        1# -> ghc-prim-0.5.3:GHC.Types.I# 1#
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Main.fib [Occ=LoopBreaker]
//  :: ghc-prim-0.5.3:GHC.Types.Int -> ghc-prim-0.5.3:GHC.Types.Int
//[LclId]
//main:Main.fib = sat_s36u
//end Rec }
//
//-- RHS size: {terms: 5, types: 1, coercions: 0, joins: 0/1}
//sat_s36A :: ghc-prim-0.5.3:GHC.Types.Int
//[LclId]
//sat_s36A
//  = let {
//      sat_s36z [Occ=Once] :: ghc-prim-0.5.3:GHC.Types.Int
//      [LclId]
//      sat_s36z = ghc-prim-0.5.3:GHC.Types.I# 10# } in
//    main:Main.fib sat_s36z
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:Main.main
//  = base-4.12.0.0:System.IO.print
//      @ ghc-prim-0.5.3:GHC.Types.Int
//      base-4.12.0.0:GHC.Show.$fShowInt
//      sat_s36A
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//main::Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main::Main.main
//  = base-4.12.0.0:GHC.TopHandler.runMainIO @ () main:Main.main
//