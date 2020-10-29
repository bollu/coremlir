// NonrecSum
// Core2MLIR: GenMLIR BeforeCorePrep
module {
//==TYCON: AbstractProd==
//unique:rka
//|data constructors|
  ==DATACON: MkAbstractProd==
  dcOrigTyCon: AbstractProd
  dcFieldLabels: []
  dcRepType: forall a b. a -> b -> AbstractProd a b
  constructor types: [a_akq, b_akr]
  result type: AbstractProd a_akq b_akr
  ---
  dcSig: ([a_akq, b_akr],
          [],
          [a_akq, b_akr],
          AbstractProd a_akq b_akr)
  dcFullSig: ([a_akq, b_akr],
              [],
              [],
              [],
              [a_akq, b_akr],
              AbstractProd a_akq b_akr)
  dcUniverseTyVars: [a_akq, b_akr]
  dcArgs: [a_akq, b_akr]
  dcOrigArgTys: [a_akq, b_akr]
  dcOrigResTy: AbstractProd a_akq b_akr
  dcRepArgTys: [a_akq, b_akr]
//----
//ctype: Nothing
//arity: 2
//binders: [anon-vis (@ a_akq), anon-vis (@ b_akr)]
//==TYCON: ConcreteRecSum==
//unique:rkc
//|data constructors|
  ==DATACON: ConcreteRecSumCons==
  dcOrigTyCon: ConcreteRecSum
  dcFieldLabels: []
  dcRepType: Int# -> ConcreteRecSum -> ConcreteRecSum
  constructor types: [Int#, ConcreteRecSum]
  result type: ConcreteRecSum
  ---
  dcSig: ([], [], [Int#, ConcreteRecSum], ConcreteRecSum)
  dcFullSig: ([],
              [],
              [],
              [],
              [Int#, ConcreteRecSum],
              ConcreteRecSum)
  dcUniverseTyVars: []
  dcArgs: [Int#, ConcreteRecSum]
  dcOrigArgTys: [Int#, ConcreteRecSum]
  dcOrigResTy: ConcreteRecSum
  dcRepArgTys: [Int#, ConcreteRecSum]
  ==DATACON: ConcreteRecSumNone==
  dcOrigTyCon: ConcreteRecSum
  dcFieldLabels: []
  dcRepType: ConcreteRecSum
  constructor types: []
  result type: ConcreteRecSum
  ---
  dcSig: ([], [], [], ConcreteRecSum)
  dcFullSig: ([], [], [], [], [], ConcreteRecSum)
  dcUniverseTyVars: []
  dcArgs: []
  dcOrigArgTys: []
  dcOrigResTy: ConcreteRecSum
  dcRepArgTys: []
//----
//ctype: Nothing
//arity: 0
//binders: []
//==TYCON: ConcreteRec==
//unique:rkf
//|data constructors|
  ==DATACON: MkConcreteRec==
  dcOrigTyCon: ConcreteRec
  dcFieldLabels: []
  dcRepType: Int# -> ConcreteRec -> ConcreteRec
  constructor types: [Int#, ConcreteRec]
  result type: ConcreteRec
  ---
  dcSig: ([], [], [Int#, ConcreteRec], ConcreteRec)
  dcFullSig: ([], [], [], [], [Int#, ConcreteRec], ConcreteRec)
  dcUniverseTyVars: []
  dcArgs: [Int#, ConcreteRec]
  dcOrigArgTys: [Int#, ConcreteRec]
  dcOrigResTy: ConcreteRec
  dcRepArgTys: [Int#, ConcreteRec]
//----
//ctype: Nothing
//arity: 0
//binders: []
//==TYCON: ConcreteSum==
//unique:rkh
//|data constructors|
  ==DATACON: ConcreteLeft==
  dcOrigTyCon: ConcreteSum
  dcFieldLabels: []
  dcRepType: Int# -> ConcreteSum
  constructor types: [Int#]
  result type: ConcreteSum
  ---
  dcSig: ([], [], [Int#], ConcreteSum)
  dcFullSig: ([], [], [], [], [Int#], ConcreteSum)
  dcUniverseTyVars: []
  dcArgs: [Int#]
  dcOrigArgTys: [Int#]
  dcOrigResTy: ConcreteSum
  dcRepArgTys: [Int#]
  ==DATACON: ConcreteRight==
  dcOrigTyCon: ConcreteSum
  dcFieldLabels: []
  dcRepType: Int# -> ConcreteSum
  constructor types: [Int#]
  result type: ConcreteSum
  ---
  dcSig: ([], [], [Int#], ConcreteSum)
  dcFullSig: ([], [], [], [], [Int#], ConcreteSum)
  dcUniverseTyVars: []
  dcArgs: [Int#]
  dcOrigArgTys: [Int#]
  dcOrigResTy: ConcreteSum
  dcRepArgTys: [Int#]
//----
//ctype: Nothing
//arity: 0
//binders: []
//==TYCON: ConcreteProd==
//unique:rkk
//|data constructors|
  ==DATACON: MkConcreteProd==
  dcOrigTyCon: ConcreteProd
  dcFieldLabels: []
  dcRepType: Int# -> Int# -> ConcreteProd
  constructor types: [Int#, Int#]
  result type: ConcreteProd
  ---
  dcSig: ([], [], [Int#, Int#], ConcreteProd)
  dcFullSig: ([], [], [], [], [Int#, Int#], ConcreteProd)
  dcUniverseTyVars: []
  dcArgs: [Int#, Int#]
  dcOrigArgTys: [Int#, Int#]
  dcOrigResTy: ConcreteProd
  dcRepArgTys: [Int#, Int#]
//----
//ctype: Nothing
//arity: 0
//binders: []
  hask.func @f {
  %lambda_0 = hask.lambda(%x_axL) {
    %case_1 = hask.case  %x_axL
    [@"ConcreteLeft" ->
    {
    ^entry(%wild_00: !hask.untyped, %i_axM: !hask.untyped):
      %app_2 = hask.ap(%ConcreteRight, %i_axM)
    hask.return(%app_2)
    }
    ]
    [@"ConcreteRight" ->
    {
    ^entry(%wild_00: !hask.untyped, %i_axN: !hask.untyped):
      %app_3 = hask.ap(%ConcreteLeft, %i_axN)
    hask.return(%app_3)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @sslone {
  %lit_4 = hask.make_i64(1)
  %app_5 = hask.ap(%ConcreteLeft, %lit_4)
  hask.return(%app_5)
  }
  hask.func @$trModule {
  %lit_6 = "main"#
  %app_7 = hask.ap(%TrNameS, %lit_6)
  %app_8 = hask.ap(%Module, %app_7)
  %lit_9 = "NonrecSum"#
  %app_10 = hask.ap(%TrNameS, %lit_9)
  %app_11 = hask.ap(%app_8, %app_10)
  hask.return(%app_11)
  }
  hask.func @$krep_aGs {
  %lit_12 = hask.make_i64(1)
  %app_13 = hask.ap(%I#, %lit_12)
  %app_14 = hask.ap(%$WKindRepVar, %app_13)
  hask.return(%app_14)
  }
  hask.func @$krep_aGq {
  %lit_15 = hask.make_i64(0)
  %app_16 = hask.ap(%I#, %lit_15)
  %app_17 = hask.ap(%$WKindRepVar, %app_16)
  hask.return(%app_17)
  }
  hask.func @$krep_aGv {
  %app_18 = hask.ap(%KindRepTyConApp, %$tcInt#)
  %type_19 = hask.make_string("TYPEINFO_ERASED")
  %app_20 = hask.ap(%[], %type_19)
  %app_21 = hask.ap(%app_18, %app_20)
  hask.return(%app_21)
  }
  hask.func @$tcConcreteProd {
  %lit_22 = 9627986161123870636
  %app_23 = hask.ap(%TyCon, %lit_22)
  %lit_24 = 8521208971585772379
  %app_25 = hask.ap(%app_23, %lit_24)
  %app_26 = hask.ap(%app_25, @$trModule)
  %lit_27 = "ConcreteProd"#
  %app_28 = hask.ap(%TrNameS, %lit_27)
  %app_29 = hask.ap(%app_26, %app_28)
  %lit_30 = hask.make_i64(0)
  %app_31 = hask.ap(%app_29, %lit_30)
  %app_32 = hask.ap(%app_31, %krep$*)
  hask.return(%app_32)
  }
  hask.func @$krep_aGF {
  %app_33 = hask.ap(%KindRepTyConApp, @$tcConcreteProd)
  %type_34 = hask.make_string("TYPEINFO_ERASED")
  %app_35 = hask.ap(%[], %type_34)
  %app_36 = hask.ap(%app_33, %app_35)
  hask.return(%app_36)
  }
  hask.func @$krep_aGE {
  %app_37 = hask.ap(%KindRepFun, @$krep_aGv)
  %app_38 = hask.ap(%app_37, @$krep_aGF)
  hask.return(%app_38)
  }
  hask.func @$krep_aGD {
  %app_39 = hask.ap(%KindRepFun, @$krep_aGv)
  %app_40 = hask.ap(%app_39, @$krep_aGE)
  hask.return(%app_40)
  }
  hask.func @$tc'MkConcreteProd {
  %lit_41 = 12942902748065332888
  %app_42 = hask.ap(%TyCon, %lit_41)
  %lit_43 = 624046941574007678
  %app_44 = hask.ap(%app_42, %lit_43)
  %app_45 = hask.ap(%app_44, @$trModule)
  %lit_46 = "'MkConcreteProd"#
  %app_47 = hask.ap(%TrNameS, %lit_46)
  %app_48 = hask.ap(%app_45, %app_47)
  %lit_49 = hask.make_i64(0)
  %app_50 = hask.ap(%app_48, %lit_49)
  %app_51 = hask.ap(%app_50, @$krep_aGD)
  hask.return(%app_51)
  }
  hask.func @$tcConcreteSum {
  %lit_52 = 11895830767168603041
  %app_53 = hask.ap(%TyCon, %lit_52)
  %lit_54 = 16356649953315961404
  %app_55 = hask.ap(%app_53, %lit_54)
  %app_56 = hask.ap(%app_55, @$trModule)
  %lit_57 = "ConcreteSum"#
  %app_58 = hask.ap(%TrNameS, %lit_57)
  %app_59 = hask.ap(%app_56, %app_58)
  %lit_60 = hask.make_i64(0)
  %app_61 = hask.ap(%app_59, %lit_60)
  %app_62 = hask.ap(%app_61, %krep$*)
  hask.return(%app_62)
  }
  hask.func @$krep_aGC {
  %app_63 = hask.ap(%KindRepTyConApp, @$tcConcreteSum)
  %type_64 = hask.make_string("TYPEINFO_ERASED")
  %app_65 = hask.ap(%[], %type_64)
  %app_66 = hask.ap(%app_63, %app_65)
  hask.return(%app_66)
  }
  hask.func @$krep_aGB {
  %app_67 = hask.ap(%KindRepFun, @$krep_aGv)
  %app_68 = hask.ap(%app_67, @$krep_aGC)
  hask.return(%app_68)
  }
  hask.func @$tc'ConcreteLeft {
  %lit_69 = 17686946225345529826
  %app_70 = hask.ap(%TyCon, %lit_69)
  %lit_71 = 11843024788528544323
  %app_72 = hask.ap(%app_70, %lit_71)
  %app_73 = hask.ap(%app_72, @$trModule)
  %lit_74 = "'ConcreteLeft"#
  %app_75 = hask.ap(%TrNameS, %lit_74)
  %app_76 = hask.ap(%app_73, %app_75)
  %lit_77 = hask.make_i64(0)
  %app_78 = hask.ap(%app_76, %lit_77)
  %app_79 = hask.ap(%app_78, @$krep_aGB)
  hask.return(%app_79)
  }
  hask.func @$tc'ConcreteRight {
  %lit_80 = 3526397897247831921
  %app_81 = hask.ap(%TyCon, %lit_80)
  %lit_82 = 16352906645058643170
  %app_83 = hask.ap(%app_81, %lit_82)
  %app_84 = hask.ap(%app_83, @$trModule)
  %lit_85 = "'ConcreteRight"#
  %app_86 = hask.ap(%TrNameS, %lit_85)
  %app_87 = hask.ap(%app_84, %app_86)
  %lit_88 = hask.make_i64(0)
  %app_89 = hask.ap(%app_87, %lit_88)
  %app_90 = hask.ap(%app_89, @$krep_aGB)
  hask.return(%app_90)
  }
  hask.func @$tcConcreteRec {
  %lit_91 = 9813922555586650147
  %app_92 = hask.ap(%TyCon, %lit_91)
  %lit_93 = 728115828137284603
  %app_94 = hask.ap(%app_92, %lit_93)
  %app_95 = hask.ap(%app_94, @$trModule)
  %lit_96 = "ConcreteRec"#
  %app_97 = hask.ap(%TrNameS, %lit_96)
  %app_98 = hask.ap(%app_95, %app_97)
  %lit_99 = hask.make_i64(0)
  %app_100 = hask.ap(%app_98, %lit_99)
  %app_101 = hask.ap(%app_100, %krep$*)
  hask.return(%app_101)
  }
  hask.func @$krep_aGA {
  %app_102 = hask.ap(%KindRepTyConApp, @$tcConcreteRec)
  %type_103 = hask.make_string("TYPEINFO_ERASED")
  %app_104 = hask.ap(%[], %type_103)
  %app_105 = hask.ap(%app_102, %app_104)
  hask.return(%app_105)
  }
  hask.func @$krep_aGz {
  %app_106 = hask.ap(%KindRepFun, @$krep_aGA)
  %app_107 = hask.ap(%app_106, @$krep_aGA)
  hask.return(%app_107)
  }
  hask.func @$krep_aGy {
  %app_108 = hask.ap(%KindRepFun, @$krep_aGv)
  %app_109 = hask.ap(%app_108, @$krep_aGz)
  hask.return(%app_109)
  }
  hask.func @$tc'MkConcreteRec {
  %lit_110 = 780870275437019420
  %app_111 = hask.ap(%TyCon, %lit_110)
  %lit_112 = 1987179208485961632
  %app_113 = hask.ap(%app_111, %lit_112)
  %app_114 = hask.ap(%app_113, @$trModule)
  %lit_115 = "'MkConcreteRec"#
  %app_116 = hask.ap(%TrNameS, %lit_115)
  %app_117 = hask.ap(%app_114, %app_116)
  %lit_118 = hask.make_i64(0)
  %app_119 = hask.ap(%app_117, %lit_118)
  %app_120 = hask.ap(%app_119, @$krep_aGy)
  hask.return(%app_120)
  }
  hask.func @$tcConcreteRecSum {
  %lit_121 = 868865939143367437
  %app_122 = hask.ap(%TyCon, %lit_121)
  %lit_123 = 4283065319836759626
  %app_124 = hask.ap(%app_122, %lit_123)
  %app_125 = hask.ap(%app_124, @$trModule)
  %lit_126 = "ConcreteRecSum"#
  %app_127 = hask.ap(%TrNameS, %lit_126)
  %app_128 = hask.ap(%app_125, %app_127)
  %lit_129 = hask.make_i64(0)
  %app_130 = hask.ap(%app_128, %lit_129)
  %app_131 = hask.ap(%app_130, %krep$*)
  hask.return(%app_131)
  }
  hask.func @$krep_aGx {
  %app_132 = hask.ap(%KindRepTyConApp, @$tcConcreteRecSum)
  %type_133 = hask.make_string("TYPEINFO_ERASED")
  %app_134 = hask.ap(%[], %type_133)
  %app_135 = hask.ap(%app_132, %app_134)
  hask.return(%app_135)
  }
  hask.func @$tc'ConcreteRecSumNone {
  %lit_136 = 8932361323873389123
  %app_137 = hask.ap(%TyCon, %lit_136)
  %lit_138 = 15462504305832975424
  %app_139 = hask.ap(%app_137, %lit_138)
  %app_140 = hask.ap(%app_139, @$trModule)
  %lit_141 = "'ConcreteRecSumNone"#
  %app_142 = hask.ap(%TrNameS, %lit_141)
  %app_143 = hask.ap(%app_140, %app_142)
  %lit_144 = hask.make_i64(0)
  %app_145 = hask.ap(%app_143, %lit_144)
  %app_146 = hask.ap(%app_145, @$krep_aGx)
  hask.return(%app_146)
  }
  hask.func @$krep_aGw {
  %app_147 = hask.ap(%KindRepFun, @$krep_aGx)
  %app_148 = hask.ap(%app_147, @$krep_aGx)
  hask.return(%app_148)
  }
  hask.func @$krep_aGu {
  %app_149 = hask.ap(%KindRepFun, @$krep_aGv)
  %app_150 = hask.ap(%app_149, @$krep_aGw)
  hask.return(%app_150)
  }
  hask.func @$tc'ConcreteRecSumCons {
  %lit_151 = 782896096195591732
  %app_152 = hask.ap(%TyCon, %lit_151)
  %lit_153 = 2603195489806365008
  %app_154 = hask.ap(%app_152, %lit_153)
  %app_155 = hask.ap(%app_154, @$trModule)
  %lit_156 = "'ConcreteRecSumCons"#
  %app_157 = hask.ap(%TrNameS, %lit_156)
  %app_158 = hask.ap(%app_155, %app_157)
  %lit_159 = hask.make_i64(0)
  %app_160 = hask.ap(%app_158, %lit_159)
  %app_161 = hask.ap(%app_160, @$krep_aGu)
  hask.return(%app_161)
  }
  hask.func @$tcAbstractProd {
  %lit_162 = 12056464016441906154
  %app_163 = hask.ap(%TyCon, %lit_162)
  %lit_164 = 6100392904543390536
  %app_165 = hask.ap(%app_163, %lit_164)
  %app_166 = hask.ap(%app_165, @$trModule)
  %lit_167 = "AbstractProd"#
  %app_168 = hask.ap(%TrNameS, %lit_167)
  %app_169 = hask.ap(%app_166, %app_168)
  %lit_170 = hask.make_i64(0)
  %app_171 = hask.ap(%app_169, %lit_170)
  %app_172 = hask.ap(%app_171, %krep$*->*->*)
  hask.return(%app_172)
  }
  hask.func @$krep_aGt {
  %app_173 = hask.ap(%KindRepTyConApp, @$tcAbstractProd)
  %type_174 = hask.make_string("TYPEINFO_ERASED")
  %app_175 = hask.ap(%:, %type_174)
  %app_176 = hask.ap(%app_175, @$krep_aGq)
  %type_177 = hask.make_string("TYPEINFO_ERASED")
  %app_178 = hask.ap(%:, %type_177)
  %app_179 = hask.ap(%app_178, @$krep_aGs)
  %type_180 = hask.make_string("TYPEINFO_ERASED")
  %app_181 = hask.ap(%[], %type_180)
  %app_182 = hask.ap(%app_179, %app_181)
  %app_183 = hask.ap(%app_176, %app_182)
  %app_184 = hask.ap(%app_173, %app_183)
  hask.return(%app_184)
  }
  hask.func @$krep_aGr {
  %app_185 = hask.ap(%KindRepFun, @$krep_aGs)
  %app_186 = hask.ap(%app_185, @$krep_aGt)
  hask.return(%app_186)
  }
  hask.func @$krep_aGp {
  %app_187 = hask.ap(%KindRepFun, @$krep_aGq)
  %app_188 = hask.ap(%app_187, @$krep_aGr)
  hask.return(%app_188)
  }
  hask.func @$tc'MkAbstractProd {
  %lit_189 = 16938648863639686556
  %app_190 = hask.ap(%TyCon, %lit_189)
  %lit_191 = 18023809221364289304
  %app_192 = hask.ap(%app_190, %lit_191)
  %app_193 = hask.ap(%app_192, @$trModule)
  %lit_194 = "'MkAbstractProd"#
  %app_195 = hask.ap(%TrNameS, %lit_194)
  %app_196 = hask.ap(%app_193, %app_195)
  %lit_197 = hask.make_i64(2)
  %app_198 = hask.ap(%app_196, %lit_197)
  %app_199 = hask.ap(%app_198, @$krep_aGp)
  hask.return(%app_199)
  }
  hask.func @main {
  %type_200 = hask.make_string("TYPEINFO_ERASED")
  %app_201 = hask.ap(%return, %type_200)
  %app_202 = hask.ap(%app_201, %$fMonadIO)
  %type_203 = hask.make_string("TYPEINFO_ERASED")
  %app_204 = hask.ap(%app_202, %type_203)
  %app_205 = hask.ap(%app_204, @"()")
  hask.return(%app_205)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 9, types: 4, coercions: 0, joins: 0/0}
//main:NonrecSum.f
//  :: main:NonrecSum.ConcreteSum -> main:NonrecSum.ConcreteSum
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [40] 40 40}]
//main:NonrecSum.f
//  = \ (x_axL :: main:NonrecSum.ConcreteSum) ->
//      case x_axL of {
//        main:NonrecSum.ConcreteLeft i_axM ->
//          main:NonrecSum.ConcreteRight i_axM;
//        main:NonrecSum.ConcreteRight i_axN ->
//          main:NonrecSum.ConcreteLeft i_axN
//      }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.sslone :: main:NonrecSum.ConcreteSum
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
//main:NonrecSum.sslone = main:NonrecSum.ConcreteLeft 1#
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$trModule :: ghc-prim-0.6.1:GHC.Types.Module
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 90 30}]
//main:NonrecSum.$trModule
//  = ghc-prim-0.6.1:GHC.Types.Module
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "NonrecSum"#)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGs [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=False, ConLike=False,
//         WorkFree=False, Expandable=False, Guidance=IF_ARGS [] 30 0}]
//$krep_aGs
//  = ghc-prim-0.6.1:GHC.Types.$WKindRepVar
//      (ghc-prim-0.6.1:GHC.Types.I# 1#)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGq [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=False, ConLike=False,
//         WorkFree=False, Expandable=False, Guidance=IF_ARGS [] 30 0}]
//$krep_aGq
//  = ghc-prim-0.6.1:GHC.Types.$WKindRepVar
//      (ghc-prim-0.6.1:GHC.Types.I# 0#)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aGv [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGv
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      ghc-prim-0.6.1:GHC.Types.$tcInt#
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 70}]
//main:NonrecSum.$tcConcreteProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      9627986161123870636##
//      8521208971585772379##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteProd"#)
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aGF [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGF
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteProd
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGE [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGE = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGv $krep_aGF
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGD [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGD = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGv $krep_aGE
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 70}]
//main:NonrecSum.$tc'MkConcreteProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      12942902748065332888##
//      624046941574007678##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'MkConcreteProd"#)
//      0#
//      $krep_aGD
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteSum :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 70}]
//main:NonrecSum.$tcConcreteSum
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      11895830767168603041##
//      16356649953315961404##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteSum"#)
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aGC [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGC
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteSum
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGB [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGB = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGv $krep_aGC
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteLeft :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 70}]
//main:NonrecSum.$tc'ConcreteLeft
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      17686946225345529826##
//      11843024788528544323##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteLeft"#)
//      0#
//      $krep_aGB
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRight :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 70}]
//main:NonrecSum.$tc'ConcreteRight
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      3526397897247831921##
//      16352906645058643170##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteRight"#)
//      0#
//      $krep_aGB
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRec :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 70}]
//main:NonrecSum.$tcConcreteRec
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      9813922555586650147##
//      728115828137284603##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteRec"#)
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aGA [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGA
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRec
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGz [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGz = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGA $krep_aGA
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGy [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGy = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGv $krep_aGz
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteRec :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 70}]
//main:NonrecSum.$tc'MkConcreteRec
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      780870275437019420##
//      1987179208485961632##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'MkConcreteRec"#)
//      0#
//      $krep_aGy
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRecSum :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 70}]
//main:NonrecSum.$tcConcreteRecSum
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      868865939143367437##
//      4283065319836759626##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteRecSum"#)
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_aGx [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGx
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRecSum
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumNone
//  :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 80 70}]
//main:NonrecSum.$tc'ConcreteRecSumNone
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      8932361323873389123##
//      15462504305832975424##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteRecSumNone"#)
//      0#
//      $krep_aGx
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGw [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGw = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGx $krep_aGx
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGu [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGu = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGv $krep_aGw
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumCons
//  :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 80 70}]
//main:NonrecSum.$tc'ConcreteRecSumCons
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      782896096195591732##
//      2603195489806365008##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteRecSumCons"#)
//      0#
//      $krep_aGu
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcAbstractProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 70}]
//main:NonrecSum.$tcAbstractProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      12056464016441906154##
//      6100392904543390536##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "AbstractProd"#)
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*->*->*
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//$krep_aGt [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 30 30}]
//$krep_aGt
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcAbstractProd
//      (ghc-prim-0.6.1:GHC.Types.:
//         @ ghc-prim-0.6.1:GHC.Types.KindRep
//         $krep_aGq
//         (ghc-prim-0.6.1:GHC.Types.:
//            @ ghc-prim-0.6.1:GHC.Types.KindRep
//            $krep_aGs
//            (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)))
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGr [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGr = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGs $krep_aGt
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_aGp [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
//$krep_aGp = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_aGq $krep_aGr
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkAbstractProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
//         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 70 70}]
//main:NonrecSum.$tc'MkAbstractProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      16938648863639686556##
//      18023809221364289304##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.6.1:GHC.Types.TrNameS "'MkAbstractProd"#)
//      2#
//      $krep_aGp
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.main :: ghc-prim-0.6.1:GHC.Types.IO ()
//[LclIdX,
// Unf=Unf{Src=<vanilla>, TopLvl=True, Value=False, ConLike=False,
//         WorkFree=False, Expandable=False, Guidance=IF_ARGS [] 30 0}]
//main:NonrecSum.main
//  = base-4.14.1.0:GHC.Base.return
//      @ ghc-prim-0.6.1:GHC.Types.IO
//      base-4.14.1.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.6.1:GHC.Tuple.()
//