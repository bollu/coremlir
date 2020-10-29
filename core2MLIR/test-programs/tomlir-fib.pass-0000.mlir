// Fib
// Core2MLIR: GenMLIR BeforeCorePrep
module {
//==TYCON: SimpleInt==
//unique:rka
//|data constructors|
  ==DATACON: MkSimpleInt==
  dcOrigTyCon: SimpleInt
  dcFieldLabels: []
  dcRepType: Int# -> SimpleInt
  constructor types: [Int#]
  result type: SimpleInt
  ---
  dcSig: ([], [], [Int#], SimpleInt)
  dcFullSig: ([], [], [], [], [Int#], SimpleInt)
  dcUniverseTyVars: []
  dcArgs: [Int#]
  dcOrigArgTys: [Int#]
  dcOrigResTy: SimpleInt
  dcRepArgTys: [Int#]
//----
//ctype: Nothing
//arity: 0
//binders: []
  hask.func @plus {
  %lambda_0 = hask.lambda(%i_axC) {
    %lambda_1 = hask.lambda(%j_axD) {
      %case_2 = hask.case  %i_axC
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_00: !hask.untyped, %ival_axE: !hask.untyped):
        %case_3 = hask.case  %j_axD
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_X5: !hask.untyped, %jval_axF: !hask.untyped):
          %app_4 = hask.ap(@"+#", %ival_axE)
          %app_5 = hask.ap(%app_4, %jval_axF)
          %app_6 = hask.ap(%MkSimpleInt, %app_5)
        hask.return(%app_6)
        }
        ]
      hask.return(%case_3)
      }
      ]
      hask.return(%case_2)
    }
    hask.return(%lambda_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @minus {
  %lambda_7 = hask.lambda(%i_axG) {
    %lambda_8 = hask.lambda(%j_axH) {
      %case_9 = hask.case  %i_axG
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_00: !hask.untyped, %ival_axI: !hask.untyped):
        %case_10 = hask.case  %j_axH
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_X6: !hask.untyped, %jval_axJ: !hask.untyped):
          %app_11 = hask.ap(@"-#", %ival_axI)
          %app_12 = hask.ap(%app_11, %jval_axJ)
          %app_13 = hask.ap(%MkSimpleInt, %app_12)
        hask.return(%app_13)
        }
        ]
      hask.return(%case_10)
      }
      ]
      hask.return(%case_9)
    }
    hask.return(%lambda_8)
  }
  hask.return(%lambda_7)
  }
  hask.func @one {
  %lit_14 = hask.make_i64(1)
  %app_15 = hask.ap(%MkSimpleInt, %lit_14)
  hask.return(%app_15)
  }
  hask.func @zero {
  %lit_16 = hask.make_i64(0)
  %app_17 = hask.ap(%MkSimpleInt, %lit_16)
  hask.return(%app_17)
  }
  hask.func @fib {
  %lambda_18 = hask.lambda(%i_axK) {
    %case_19 = hask.case  %i_axK
    [@"MkSimpleInt" ->
    {
    ^entry(%wild_00: !hask.untyped, %ds_dGo: !hask.untyped):
      %case_20 = hask.case  %ds_dGo
      ["default" ->
      {
      ^entry(%ds_XGx: !hask.untyped):
        %app_21 = hask.ap(@fib, %i_axK)
        %app_22 = hask.ap(@plus, %app_21)
        %app_23 = hask.ap(@minus, %i_axK)
        %app_24 = hask.ap(%app_23, @one)
        %app_25 = hask.ap(@fib, %app_24)
        %app_26 = hask.ap(%app_22, %app_25)
      hask.return(%app_26)
      }
      ]
      [0 ->
      {
      ^entry(%ds_XGx: !hask.untyped):
      hask.return(@zero)
      }
      ]
      [1 ->
      {
      ^entry(%ds_XGx: !hask.untyped):
      hask.return(@one)
      }
      ]
    hask.return(%case_20)
    }
    ]
    hask.return(%case_19)
  }
  hask.return(%lambda_18)
  }
  hask.func @$trModule {
  %lit_27 = "main"#
  %app_28 = hask.ap(%TrNameS, %lit_27)
  %app_29 = hask.ap(%Module, %app_28)
  %lit_30 = "Fib"#
  %app_31 = hask.ap(%TrNameS, %lit_30)
  %app_32 = hask.ap(%app_29, %app_31)
  hask.return(%app_32)
  }
  hask.func @$krep_aFv {
  %app_33 = hask.ap(%KindRepTyConApp, %$tcInt#)
  %type_34 = hask.make_string("TYPEINFO_ERASED")
  %app_35 = hask.ap(%[], %type_34)
  %app_36 = hask.ap(%app_33, %app_35)
  hask.return(%app_36)
  }
  hask.func @$tcSimpleInt {
  %lit_37 = 7748924056139856202
  %app_38 = hask.ap(%TyCon, %lit_37)
  %lit_39 = 3495965821938844824
  %app_40 = hask.ap(%app_38, %lit_39)
  %app_41 = hask.ap(%app_40, @$trModule)
  %lit_42 = "SimpleInt"#
  %app_43 = hask.ap(%TrNameS, %lit_42)
  %app_44 = hask.ap(%app_41, %app_43)
  %lit_45 = hask.make_i64(0)
  %app_46 = hask.ap(%app_44, %lit_45)
  %app_47 = hask.ap(%app_46, %krep$*)
  hask.return(%app_47)
  }
  hask.func @$krep_aFw {
  %app_48 = hask.ap(%KindRepTyConApp, @$tcSimpleInt)
  %type_49 = hask.make_string("TYPEINFO_ERASED")
  %app_50 = hask.ap(%[], %type_49)
  %app_51 = hask.ap(%app_48, %app_50)
  hask.return(%app_51)
  }
  hask.func @$krep_aFu {
  %app_52 = hask.ap(%KindRepFun, @$krep_aFv)
  %app_53 = hask.ap(%app_52, @$krep_aFw)
  hask.return(%app_53)
  }
  hask.func @$tc'MkSimpleInt {
  %lit_54 = 4702169295613205611
  %app_55 = hask.ap(%TyCon, %lit_54)
  %lit_56 = 5219586552818290360
  %app_57 = hask.ap(%app_55, %lit_56)
  %app_58 = hask.ap(%app_57, @$trModule)
  %lit_59 = "'MkSimpleInt"#
  %app_60 = hask.ap(%TrNameS, %lit_59)
  %app_61 = hask.ap(%app_58, %app_60)
  %lit_62 = hask.make_i64(0)
  %app_63 = hask.ap(%app_61, %lit_62)
  %app_64 = hask.ap(%app_63, @$krep_aFu)
  hask.return(%app_64)
  }
  hask.func @main {
  %type_65 = hask.make_string("TYPEINFO_ERASED")
  %app_66 = hask.ap(%return, %type_65)
  %app_67 = hask.ap(%app_66, %$fMonadIO)
  %type_68 = hask.make_string("TYPEINFO_ERASED")
  %app_69 = hask.ap(%app_67, %type_68)
  %app_70 = hask.ap(%app_69, @"()")
  hask.return(%app_70)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 12, types: 6, coercions: 0, joins: 0/0}
//main:Fib.plus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [20 20] 31 20}]
//main:Fib.plus
//  = \ (i_axC :: main:Fib.SimpleInt) (j_axD :: main:Fib.SimpleInt) ->
//      case i_axC of { main:Fib.MkSimpleInt ival_axE ->
//      case j_axD of { main:Fib.MkSimpleInt jval_axF ->
//      main:Fib.MkSimpleInt (ghc-prim-0.6.1:GHC.Prim.+# ival_axE jval_axF)
//      }
//      }
//
//-- RHS size: {terms: 12, types: 6, coercions: 0, joins: 0/0}
//main:Fib.minus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [20 20] 31 20}]
//main:Fib.minus
//  = \ (i_axG :: main:Fib.SimpleInt) (j_axH :: main:Fib.SimpleInt) ->
//      case i_axG of { main:Fib.MkSimpleInt ival_axI ->
//      case j_axH of { main:Fib.MkSimpleInt jval_axJ ->
//      main:Fib.MkSimpleInt (ghc-prim-0.6.1:GHC.Prim.-# ival_axI jval_axJ)
//      }
//      }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:Fib.one :: main:Fib.SimpleInt
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
//main:Fib.one = main:Fib.MkSimpleInt 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:Fib.zero :: main:Fib.SimpleInt
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
//main:Fib.zero = main:Fib.MkSimpleInt 0#
//
//Rec {
//-- RHS size: {terms: 18, types: 4, coercions: 0, joins: 0/0}
//main:Fib.fib [Occ=LoopBreaker]
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [20] 140 0}]
//main:Fib.fib
//  = \ (i_axK :: main:Fib.SimpleInt) ->
//      case i_axK of { main:Fib.MkSimpleInt ds_dGo ->
//      case ds_dGo of {
//        __DEFAULT ->
//          main:Fib.plus
//            (main:Fib.fib i_axK)
//            (main:Fib.fib (main:Fib.minus i_axK main:Fib.one));
//        0# -> main:Fib.zero;
//        1# -> main:Fib.one
//      }
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$trModule :: ghc-prim-0.6.1:GHC.Types.Module
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 30}]
//main:Fib.$trModule
//  = ghc-prim-0.6.1:GHC.Types.Module
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "Fib"#)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aFv [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aFv
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      ghc-prim-0.6.1:GHC.Types.$tcInt#
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tcSimpleInt :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 70}]
//main:Fib.$tcSimpleInt
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      7748924056139856202##
//      3495965821938844824##
//      main:Fib.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "SimpleInt"#)
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aFw [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aFw
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:Fib.$tcSimpleInt
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aFu [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aFu = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aFv $krep_aFw
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tc'MkSimpleInt :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 70}]
//main:Fib.$tc'MkSimpleInt
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      4702169295613205611##
//      5219586552818290360##
//      main:Fib.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'MkSimpleInt"#)
//      0#
//      $krep_aFu
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:Fib.main :: ghc-prim-0.6.1:GHC.Types.IO ()
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=False, ConLike=False,
//         WorkFree=False, Expandable=False, Guidance=IF_ARGS [] 30 0}]
//main:Fib.main
//  = base-4.14.1.0:GHC.Base.return
//      @ ghc-prim-0.6.1:GHC.Types.IO
//      base-4.14.1.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.6.1:GHC.Tuple.()
//