// Fib
// Core2MLIR: GenMLIR AfterCorePrep
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
  hask.func @sat_sMG {
  %lambda_0 = hask.lambda(%i_sMz) {
    %lambda_1 = hask.lambda(%j_sMA) {
      %case_2 = hask.case  %i_sMz
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_sMB: !hask.untyped, %ival_sMC: !hask.untyped):
        %case_3 = hask.case  %j_sMA
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_sMD: !hask.untyped, %jval_sME: !hask.untyped):
          %app_4 = hask.ap(@"+#", %ival_sMC)
          %app_5 = hask.ap(%app_4, %jval_sME)
          %sat_sMF = hask.force (%app_5)
          %app_7 = hask.ap(@MkSimpleInt, %sat_sMF)
        hask.return(%app_7)
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
  hask.func @plus {
  hask.return(@sat_sMG)
  }
  hask.func @sat_sMP {
  %lambda_8 = hask.lambda(%i_sMI) {
    %lambda_9 = hask.lambda(%j_sMJ) {
      %case_10 = hask.case  %i_sMI
      [@"MkSimpleInt" ->
      {
      ^entry(%wild_sMK: !hask.untyped, %ival_sML: !hask.untyped):
        %case_11 = hask.case  %j_sMJ
        [@"MkSimpleInt" ->
        {
        ^entry(%wild_sMM: !hask.untyped, %jval_sMN: !hask.untyped):
          %app_12 = hask.ap(@"-#", %ival_sML)
          %app_13 = hask.ap(%app_12, %jval_sMN)
          %sat_sMO = hask.force (%app_13)
          %app_15 = hask.ap(@MkSimpleInt, %sat_sMO)
        hask.return(%app_15)
        }
        ]
      hask.return(%case_11)
      }
      ]
      hask.return(%case_10)
    }
    hask.return(%lambda_9)
  }
  hask.return(%lambda_8)
  }
  hask.func @minus {
  hask.return(@sat_sMP)
  }
  hask.func @one {
  %lit_16 = hask.make_i64(1)
  %app_17 = hask.ap(@MkSimpleInt, %lit_16)
  hask.return(%app_17)
  }
  hask.func @zero {
  %lit_18 = hask.make_i64(0)
  %app_19 = hask.ap(@MkSimpleInt, %lit_18)
  hask.return(%app_19)
  }
  hask.func @sat_sN0 {
  %lambda_20 = hask.lambda(%i_sMT) {
    %case_21 = hask.case  %i_sMT
    [@"MkSimpleInt" ->
    {
    ^entry(%wild_sMU: !hask.untyped, %ds_sMV: !hask.untyped):
      %case_22 = hask.case  %ds_sMV
      ["default" ->
      {
      ^entry(%ds_sMW: !hask.untyped):
        %unimpl_let_nonrec23  =  hask.make_i32(42)
      hask.return(%unimpl_let_nonrec23)
      }
      ]
      [0 ->
      {
      ^entry(%ds_sMW: !hask.untyped):
      hask.return(@zero)
      }
      ]
      [1 ->
      {
      ^entry(%ds_sMW: !hask.untyped):
      hask.return(@one)
      }
      ]
    hask.return(%case_22)
    }
    ]
    hask.return(%case_21)
  }
  hask.return(%lambda_20)
  }
  hask.func @fib {
  hask.return(@sat_sN0)
  }
  hask.func @sat_sN3 {
  %lit_24 = "Fib"#
  %app_25 = hask.ap(%TrNameS, %lit_24)
  hask.return(%app_25)
  }
  hask.func @sat_sN2 {
  %lit_26 = "main"#
  %app_27 = hask.ap(%TrNameS, %lit_26)
  hask.return(%app_27)
  }
  hask.func @$trModule {
  %app_28 = hask.ap(%Module, @sat_sN2)
  %app_29 = hask.ap(%app_28, @sat_sN3)
  hask.return(%app_29)
  }
  hask.func @$krep_sN4 {
  %app_30 = hask.ap(%KindRepTyConApp, %$tcInt#)
  %type_31 = hask.make_string("TYPEINFO_ERASED")
  %app_32 = hask.ap(%[], %type_31)
  %app_33 = hask.ap(%app_30, %app_32)
  hask.return(%app_33)
  }
  hask.func @sat_sN6 {
  %lit_34 = "SimpleInt"#
  %app_35 = hask.ap(%TrNameS, %lit_34)
  hask.return(%app_35)
  }
  hask.func @$tcSimpleInt {
  %lit_36 = 7748924056139856202
  %app_37 = hask.ap(%TyCon, %lit_36)
  %lit_38 = 3495965821938844824
  %app_39 = hask.ap(%app_37, %lit_38)
  %app_40 = hask.ap(%app_39, @$trModule)
  %app_41 = hask.ap(%app_40, @sat_sN6)
  %lit_42 = hask.make_i64(0)
  %app_43 = hask.ap(%app_41, %lit_42)
  %app_44 = hask.ap(%app_43, %krep$*)
  hask.return(%app_44)
  }
  hask.func @$krep_sN7 {
  %app_45 = hask.ap(%KindRepTyConApp, @$tcSimpleInt)
  %type_46 = hask.make_string("TYPEINFO_ERASED")
  %app_47 = hask.ap(%[], %type_46)
  %app_48 = hask.ap(%app_45, %app_47)
  hask.return(%app_48)
  }
  hask.func @$krep_sN8 {
  %app_49 = hask.ap(%KindRepFun, @$krep_sN4)
  %app_50 = hask.ap(%app_49, @$krep_sN7)
  hask.return(%app_50)
  }
  hask.func @sat_sNa {
  %lit_51 = "'MkSimpleInt"#
  %app_52 = hask.ap(%TrNameS, %lit_51)
  hask.return(%app_52)
  }
  hask.func @$tc'MkSimpleInt {
  %lit_53 = 4702169295613205611
  %app_54 = hask.ap(%TyCon, %lit_53)
  %lit_55 = 5219586552818290360
  %app_56 = hask.ap(%app_54, %lit_55)
  %app_57 = hask.ap(%app_56, @$trModule)
  %app_58 = hask.ap(%app_57, @sat_sNa)
  %lit_59 = hask.make_i64(0)
  %app_60 = hask.ap(%app_58, %lit_59)
  %app_61 = hask.ap(%app_60, @$krep_sN8)
  hask.return(%app_61)
  }
  hask.func @main {
  %type_62 = hask.make_string("TYPEINFO_ERASED")
  %app_63 = hask.ap(%return, %type_62)
  %app_64 = hask.ap(%app_63, %$fMonadIO)
  %type_65 = hask.make_string("TYPEINFO_ERASED")
  %app_66 = hask.ap(%app_64, %type_65)
  %app_67 = hask.ap(%app_66, @"()")
  hask.return(%app_67)
  }
  hask.func @MkSimpleInt {
  %lambda_68 = hask.lambda(%eta_B1) {
    %app_69 = hask.ap(@MkSimpleInt, %eta_B1)
    hask.return(%app_69)
  }
  hask.return(%lambda_68)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 15, types: 7, coercions: 0, joins: 0/0}
//sat_sMG
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclId]
//sat_sMG
//  = \ (i_sMz [Occ=Once!] :: main:Fib.SimpleInt)
//      (j_sMA [Occ=Once!] :: main:Fib.SimpleInt) ->
//      case i_sMz of { main:Fib.MkSimpleInt ival_sMC [Occ=Once] ->
//      case j_sMA of { main:Fib.MkSimpleInt jval_sME [Occ=Once] ->
//      case ghc-prim-0.6.1:GHC.Prim.+# ival_sMC jval_sME
//      of sat_sMF [Occ=Once]
//      { __DEFAULT ->
//      main:Fib.MkSimpleInt sat_sMF
//      }
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Fib.plus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX, Unf=OtherCon []]
//main:Fib.plus = sat_sMG
//
//-- RHS size: {terms: 15, types: 7, coercions: 0, joins: 0/0}
//sat_sMP
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclId]
//sat_sMP
//  = \ (i_sMI [Occ=Once!] :: main:Fib.SimpleInt)
//      (j_sMJ [Occ=Once!] :: main:Fib.SimpleInt) ->
//      case i_sMI of { main:Fib.MkSimpleInt ival_sML [Occ=Once] ->
//      case j_sMJ of { main:Fib.MkSimpleInt jval_sMN [Occ=Once] ->
//      case ghc-prim-0.6.1:GHC.Prim.-# ival_sML jval_sMN
//      of sat_sMO [Occ=Once]
//      { __DEFAULT ->
//      main:Fib.MkSimpleInt sat_sMO
//      }
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Fib.minus
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX, Unf=OtherCon []]
//main:Fib.minus = sat_sMP
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:Fib.one :: main:Fib.SimpleInt
//[LclIdX, Unf=OtherCon []]
//main:Fib.one = main:Fib.MkSimpleInt 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:Fib.zero :: main:Fib.SimpleInt
//[LclIdX, Unf=OtherCon []]
//main:Fib.zero = main:Fib.MkSimpleInt 0#
//
//Rec {
//-- RHS size: {terms: 24, types: 7, coercions: 0, joins: 0/3}
//sat_sN0 :: main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclId]
//sat_sN0
//  = \ (i_sMT :: main:Fib.SimpleInt) ->
//      case i_sMT of { main:Fib.MkSimpleInt ds_sMV [Occ=Once!] ->
//      case ds_sMV of {
//        __DEFAULT ->
//          let {
//            sat_sMZ [Occ=Once] :: main:Fib.SimpleInt
//            [LclId]
//            sat_sMZ
//              = let {
//                  sat_sMY [Occ=Once] :: main:Fib.SimpleInt
//                  [LclId]
//                  sat_sMY = main:Fib.minus i_sMT main:Fib.one } in
//                main:Fib.fib sat_sMY } in
//          let {
//            sat_sMX [Occ=Once] :: main:Fib.SimpleInt
//            [LclId]
//            sat_sMX = main:Fib.fib i_sMT } in
//          main:Fib.plus sat_sMX sat_sMZ;
//        0# -> main:Fib.zero;
//        1# -> main:Fib.one
//      }
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:Fib.fib [Occ=LoopBreaker]
//  :: main:Fib.SimpleInt -> main:Fib.SimpleInt
//[LclIdX, Unf=OtherCon []]
//main:Fib.fib = sat_sN0
//end Rec }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sN3 :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sN3 = ghc-prim-0.6.1:GHC.Types.TrNameS "Fib"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sN2 :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sN2 = ghc-prim-0.6.1:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$trModule :: ghc-prim-0.6.1:GHC.Types.Module
//[LclIdX, Unf=OtherCon []]
//main:Fib.$trModule
//  = ghc-prim-0.6.1:GHC.Types.Module sat_sN2 sat_sN3
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sN4 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sN4
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      ghc-prim-0.6.1:GHC.Types.$tcInt#
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sN6 :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sN6 = ghc-prim-0.6.1:GHC.Types.TrNameS "SimpleInt"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tcSimpleInt :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:Fib.$tcSimpleInt
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      7748924056139856202##
//      3495965821938844824##
//      main:Fib.$trModule
//      sat_sN6
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sN7 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sN7
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:Fib.$tcSimpleInt
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sN8 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sN8 = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sN4 $krep_sN7
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNa :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNa = ghc-prim-0.6.1:GHC.Types.TrNameS "'MkSimpleInt"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:Fib.$tc'MkSimpleInt :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:Fib.$tc'MkSimpleInt
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      4702169295613205611##
//      5219586552818290360##
//      main:Fib.$trModule
//      sat_sNa
//      0#
//      $krep_sN8
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:Fib.main :: ghc-prim-0.6.1:GHC.Types.IO ()
//[LclIdX]
//main:Fib.main
//  = base-4.14.1.0:GHC.Base.return
//      @ ghc-prim-0.6.1:GHC.Types.IO
//      base-4.14.1.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.6.1:GHC.Tuple.()
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:Fib.MkSimpleInt
//  :: ghc-prim-0.6.1:GHC.Prim.Int# -> main:Fib.SimpleInt
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m,
// Unf=OtherCon []]
//main:Fib.MkSimpleInt
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#) ->
//      main:Fib.MkSimpleInt eta_B1
//