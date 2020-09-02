// NonrecSum
// Core2MLIR: GenMLIR BeforeCorePrep
module {
//==TYCON: AbstractProd==
//unique:rwp
//|data constructors|
  ==DATACON: MkAbstractProd==
  dcOrigTyCon: AbstractProd
  dcFieldLabels: []
  dcRepType: forall a b. a -> b -> AbstractProd a b
  constructor types: [a_azg[sk:0], b_azh[sk:0]]
  result type: AbstractProd a_azg b_azh
  ---
  dcSig: ([a_azg, b_azh],
          [],
          [a_azg[sk:0], b_azh[sk:0]],
          AbstractProd a_azg b_azh)
  dcFullSig: ([a_azg, b_azh],
              [],
              [],
              [],
              [a_azg[sk:0], b_azh[sk:0]],
              AbstractProd a_azg b_azh)
  dcUniverseTyVars: [a_azg, b_azh]
  dcArgs: [a_azg[sk:0], b_azh[sk:0]]
  dcOrigArgTys: [a_azg[sk:0], b_azh[sk:0]]
  dcOrigResTy: AbstractProd a_azg b_azh
  dcRepArgTys: [a_azg[sk:0], b_azh[sk:0]]
//----
//ctype: Nothing
//arity: 2
//binders: [anon (a_azg), anon (b_azh)]
//==TYCON: ConcreteRecSum==
//unique:rz2
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
//unique:rz5
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
//unique:rz7
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
//unique:rza
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
  %lambda_0 = hask.lambdaSSA(%x_a12Z) {
    %case_1 = hask.caseSSA  %x_a12Z
    [@"ConcreteLeft" ->
    {
    ^entry(%wild_00: !hask.untyped, %i_a130: !hask.untyped):
      %app_2 = hask.apSSA(%ConcreteRight, %i_a130)
    hask.return(%app_2)
    }
    ]
    [@"ConcreteRight" ->
    {
    ^entry(%wild_00: !hask.untyped, %i_a131: !hask.untyped):
      %app_3 = hask.apSSA(%ConcreteLeft, %i_a131)
    hask.return(%app_3)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @sslone {
  %lit_4 = hask.make_i64(1)
  %app_5 = hask.apSSA(%ConcreteLeft, %lit_4)
  hask.return(%app_5)
  }
  hask.func @$trModule {
  %lit_6 = hask.make_string("main")
  %app_7 = hask.apSSA(%TrNameS, %lit_6)
  %app_8 = hask.apSSA(%Module, %app_7)
  %lit_9 = hask.make_string("NonrecSum")
  %app_10 = hask.apSSA(%TrNameS, %lit_9)
  %app_11 = hask.apSSA(%app_8, %app_10)
  hask.return(%app_11)
  }
  hask.func @$krep_a1l9 {
  %lit_12 = hask.make_i64(1)
  %app_13 = hask.apSSA(%I#, %lit_12)
  %app_14 = hask.apSSA(%$WKindRepVar, %app_13)
  hask.return(%app_14)
  }
  hask.func @$krep_a1l7 {
  %lit_15 = hask.make_i64(0)
  %app_16 = hask.apSSA(%I#, %lit_15)
  %app_17 = hask.apSSA(%$WKindRepVar, %app_16)
  hask.return(%app_17)
  }
  hask.func @$krep_a1lc {
  %app_18 = hask.apSSA(%KindRepTyConApp, %$tcInt#)
  %type_19 = hask.make_string("TYPEINFO_ERASED")
  %app_20 = hask.apSSA(%[], %type_19)
  %app_21 = hask.apSSA(%app_18, %app_20)
  hask.return(%app_21)
  }
  hask.func @$tcConcreteProd {
  %lit_22 = 9627986161123870636
  %app_23 = hask.apSSA(%TyCon, %lit_22)
  %lit_24 = 8521208971585772379
  %app_25 = hask.apSSA(%app_23, %lit_24)
  %app_26 = hask.apSSA(%app_25, @$trModule)
  %lit_27 = hask.make_string("ConcreteProd")
  %app_28 = hask.apSSA(%TrNameS, %lit_27)
  %app_29 = hask.apSSA(%app_26, %app_28)
  %lit_30 = hask.make_i64(0)
  %app_31 = hask.apSSA(%app_29, %lit_30)
  %app_32 = hask.apSSA(%app_31, %krep$*)
  hask.return(%app_32)
  }
  hask.func @$krep_a1lm {
  %app_33 = hask.apSSA(%KindRepTyConApp, @$tcConcreteProd)
  %type_34 = hask.make_string("TYPEINFO_ERASED")
  %app_35 = hask.apSSA(%[], %type_34)
  %app_36 = hask.apSSA(%app_33, %app_35)
  hask.return(%app_36)
  }
  hask.func @$krep_a1ll {
  %app_37 = hask.apSSA(%KindRepFun, @$krep_a1lc)
  %app_38 = hask.apSSA(%app_37, @$krep_a1lm)
  hask.return(%app_38)
  }
  hask.func @$krep_a1lk {
  %app_39 = hask.apSSA(%KindRepFun, @$krep_a1lc)
  %app_40 = hask.apSSA(%app_39, @$krep_a1ll)
  hask.return(%app_40)
  }
  hask.func @$tc'MkConcreteProd {
  %lit_41 = 12942902748065332888
  %app_42 = hask.apSSA(%TyCon, %lit_41)
  %lit_43 = 624046941574007678
  %app_44 = hask.apSSA(%app_42, %lit_43)
  %app_45 = hask.apSSA(%app_44, @$trModule)
  %lit_46 = hask.make_string("'MkConcreteProd")
  %app_47 = hask.apSSA(%TrNameS, %lit_46)
  %app_48 = hask.apSSA(%app_45, %app_47)
  %lit_49 = hask.make_i64(0)
  %app_50 = hask.apSSA(%app_48, %lit_49)
  %app_51 = hask.apSSA(%app_50, @$krep_a1lk)
  hask.return(%app_51)
  }
  hask.func @$tcConcreteSum {
  %lit_52 = 11895830767168603041
  %app_53 = hask.apSSA(%TyCon, %lit_52)
  %lit_54 = 16356649953315961404
  %app_55 = hask.apSSA(%app_53, %lit_54)
  %app_56 = hask.apSSA(%app_55, @$trModule)
  %lit_57 = hask.make_string("ConcreteSum")
  %app_58 = hask.apSSA(%TrNameS, %lit_57)
  %app_59 = hask.apSSA(%app_56, %app_58)
  %lit_60 = hask.make_i64(0)
  %app_61 = hask.apSSA(%app_59, %lit_60)
  %app_62 = hask.apSSA(%app_61, %krep$*)
  hask.return(%app_62)
  }
  hask.func @$krep_a1lj {
  %app_63 = hask.apSSA(%KindRepTyConApp, @$tcConcreteSum)
  %type_64 = hask.make_string("TYPEINFO_ERASED")
  %app_65 = hask.apSSA(%[], %type_64)
  %app_66 = hask.apSSA(%app_63, %app_65)
  hask.return(%app_66)
  }
  hask.func @$krep_a1li {
  %app_67 = hask.apSSA(%KindRepFun, @$krep_a1lc)
  %app_68 = hask.apSSA(%app_67, @$krep_a1lj)
  hask.return(%app_68)
  }
  hask.func @$tc'ConcreteLeft {
  %lit_69 = 17686946225345529826
  %app_70 = hask.apSSA(%TyCon, %lit_69)
  %lit_71 = 11843024788528544323
  %app_72 = hask.apSSA(%app_70, %lit_71)
  %app_73 = hask.apSSA(%app_72, @$trModule)
  %lit_74 = hask.make_string("'ConcreteLeft")
  %app_75 = hask.apSSA(%TrNameS, %lit_74)
  %app_76 = hask.apSSA(%app_73, %app_75)
  %lit_77 = hask.make_i64(0)
  %app_78 = hask.apSSA(%app_76, %lit_77)
  %app_79 = hask.apSSA(%app_78, @$krep_a1li)
  hask.return(%app_79)
  }
  hask.func @$tc'ConcreteRight {
  %lit_80 = 3526397897247831921
  %app_81 = hask.apSSA(%TyCon, %lit_80)
  %lit_82 = 16352906645058643170
  %app_83 = hask.apSSA(%app_81, %lit_82)
  %app_84 = hask.apSSA(%app_83, @$trModule)
  %lit_85 = hask.make_string("'ConcreteRight")
  %app_86 = hask.apSSA(%TrNameS, %lit_85)
  %app_87 = hask.apSSA(%app_84, %app_86)
  %lit_88 = hask.make_i64(0)
  %app_89 = hask.apSSA(%app_87, %lit_88)
  %app_90 = hask.apSSA(%app_89, @$krep_a1li)
  hask.return(%app_90)
  }
  hask.func @$tcConcreteRec {
  %lit_91 = 9813922555586650147
  %app_92 = hask.apSSA(%TyCon, %lit_91)
  %lit_93 = 728115828137284603
  %app_94 = hask.apSSA(%app_92, %lit_93)
  %app_95 = hask.apSSA(%app_94, @$trModule)
  %lit_96 = hask.make_string("ConcreteRec")
  %app_97 = hask.apSSA(%TrNameS, %lit_96)
  %app_98 = hask.apSSA(%app_95, %app_97)
  %lit_99 = hask.make_i64(0)
  %app_100 = hask.apSSA(%app_98, %lit_99)
  %app_101 = hask.apSSA(%app_100, %krep$*)
  hask.return(%app_101)
  }
  hask.func @$krep_a1lh {
  %app_102 = hask.apSSA(%KindRepTyConApp, @$tcConcreteRec)
  %type_103 = hask.make_string("TYPEINFO_ERASED")
  %app_104 = hask.apSSA(%[], %type_103)
  %app_105 = hask.apSSA(%app_102, %app_104)
  hask.return(%app_105)
  }
  hask.func @$krep_a1lg {
  %app_106 = hask.apSSA(%KindRepFun, @$krep_a1lh)
  %app_107 = hask.apSSA(%app_106, @$krep_a1lh)
  hask.return(%app_107)
  }
  hask.func @$krep_a1lf {
  %app_108 = hask.apSSA(%KindRepFun, @$krep_a1lc)
  %app_109 = hask.apSSA(%app_108, @$krep_a1lg)
  hask.return(%app_109)
  }
  hask.func @$tc'MkConcreteRec {
  %lit_110 = 780870275437019420
  %app_111 = hask.apSSA(%TyCon, %lit_110)
  %lit_112 = 1987179208485961632
  %app_113 = hask.apSSA(%app_111, %lit_112)
  %app_114 = hask.apSSA(%app_113, @$trModule)
  %lit_115 = hask.make_string("'MkConcreteRec")
  %app_116 = hask.apSSA(%TrNameS, %lit_115)
  %app_117 = hask.apSSA(%app_114, %app_116)
  %lit_118 = hask.make_i64(0)
  %app_119 = hask.apSSA(%app_117, %lit_118)
  %app_120 = hask.apSSA(%app_119, @$krep_a1lf)
  hask.return(%app_120)
  }
  hask.func @$tcConcreteRecSum {
  %lit_121 = 868865939143367437
  %app_122 = hask.apSSA(%TyCon, %lit_121)
  %lit_123 = 4283065319836759626
  %app_124 = hask.apSSA(%app_122, %lit_123)
  %app_125 = hask.apSSA(%app_124, @$trModule)
  %lit_126 = hask.make_string("ConcreteRecSum")
  %app_127 = hask.apSSA(%TrNameS, %lit_126)
  %app_128 = hask.apSSA(%app_125, %app_127)
  %lit_129 = hask.make_i64(0)
  %app_130 = hask.apSSA(%app_128, %lit_129)
  %app_131 = hask.apSSA(%app_130, %krep$*)
  hask.return(%app_131)
  }
  hask.func @$krep_a1le {
  %app_132 = hask.apSSA(%KindRepTyConApp, @$tcConcreteRecSum)
  %type_133 = hask.make_string("TYPEINFO_ERASED")
  %app_134 = hask.apSSA(%[], %type_133)
  %app_135 = hask.apSSA(%app_132, %app_134)
  hask.return(%app_135)
  }
  hask.func @$tc'ConcreteRecSumNone {
  %lit_136 = 8932361323873389123
  %app_137 = hask.apSSA(%TyCon, %lit_136)
  %lit_138 = 15462504305832975424
  %app_139 = hask.apSSA(%app_137, %lit_138)
  %app_140 = hask.apSSA(%app_139, @$trModule)
  %lit_141 = hask.make_string("'ConcreteRecSumNone")
  %app_142 = hask.apSSA(%TrNameS, %lit_141)
  %app_143 = hask.apSSA(%app_140, %app_142)
  %lit_144 = hask.make_i64(0)
  %app_145 = hask.apSSA(%app_143, %lit_144)
  %app_146 = hask.apSSA(%app_145, @$krep_a1le)
  hask.return(%app_146)
  }
  hask.func @$krep_a1ld {
  %app_147 = hask.apSSA(%KindRepFun, @$krep_a1le)
  %app_148 = hask.apSSA(%app_147, @$krep_a1le)
  hask.return(%app_148)
  }
  hask.func @$krep_a1lb {
  %app_149 = hask.apSSA(%KindRepFun, @$krep_a1lc)
  %app_150 = hask.apSSA(%app_149, @$krep_a1ld)
  hask.return(%app_150)
  }
  hask.func @$tc'ConcreteRecSumCons {
  %lit_151 = 782896096195591732
  %app_152 = hask.apSSA(%TyCon, %lit_151)
  %lit_153 = 2603195489806365008
  %app_154 = hask.apSSA(%app_152, %lit_153)
  %app_155 = hask.apSSA(%app_154, @$trModule)
  %lit_156 = hask.make_string("'ConcreteRecSumCons")
  %app_157 = hask.apSSA(%TrNameS, %lit_156)
  %app_158 = hask.apSSA(%app_155, %app_157)
  %lit_159 = hask.make_i64(0)
  %app_160 = hask.apSSA(%app_158, %lit_159)
  %app_161 = hask.apSSA(%app_160, @$krep_a1lb)
  hask.return(%app_161)
  }
  hask.func @$tcAbstractProd {
  %lit_162 = 12056464016441906154
  %app_163 = hask.apSSA(%TyCon, %lit_162)
  %lit_164 = 6100392904543390536
  %app_165 = hask.apSSA(%app_163, %lit_164)
  %app_166 = hask.apSSA(%app_165, @$trModule)
  %lit_167 = hask.make_string("AbstractProd")
  %app_168 = hask.apSSA(%TrNameS, %lit_167)
  %app_169 = hask.apSSA(%app_166, %app_168)
  %lit_170 = hask.make_i64(0)
  %app_171 = hask.apSSA(%app_169, %lit_170)
  %app_172 = hask.apSSA(%app_171, %krep$*->*->*)
  hask.return(%app_172)
  }
  hask.func @$krep_a1la {
  %app_173 = hask.apSSA(%KindRepTyConApp, @$tcAbstractProd)
  %type_174 = hask.make_string("TYPEINFO_ERASED")
  %app_175 = hask.apSSA(%:, %type_174)
  %app_176 = hask.apSSA(%app_175, @$krep_a1l7)
  %type_177 = hask.make_string("TYPEINFO_ERASED")
  %app_178 = hask.apSSA(%:, %type_177)
  %app_179 = hask.apSSA(%app_178, @$krep_a1l9)
  %type_180 = hask.make_string("TYPEINFO_ERASED")
  %app_181 = hask.apSSA(%[], %type_180)
  %app_182 = hask.apSSA(%app_179, %app_181)
  %app_183 = hask.apSSA(%app_176, %app_182)
  %app_184 = hask.apSSA(%app_173, %app_183)
  hask.return(%app_184)
  }
  hask.func @$krep_a1l8 {
  %app_185 = hask.apSSA(%KindRepFun, @$krep_a1l9)
  %app_186 = hask.apSSA(%app_185, @$krep_a1la)
  hask.return(%app_186)
  }
  hask.func @$krep_a1l6 {
  %app_187 = hask.apSSA(%KindRepFun, @$krep_a1l7)
  %app_188 = hask.apSSA(%app_187, @$krep_a1l8)
  hask.return(%app_188)
  }
  hask.func @$tc'MkAbstractProd {
  %lit_189 = 16938648863639686556
  %app_190 = hask.apSSA(%TyCon, %lit_189)
  %lit_191 = 18023809221364289304
  %app_192 = hask.apSSA(%app_190, %lit_191)
  %app_193 = hask.apSSA(%app_192, @$trModule)
  %lit_194 = hask.make_string("'MkAbstractProd")
  %app_195 = hask.apSSA(%TrNameS, %lit_194)
  %app_196 = hask.apSSA(%app_193, %app_195)
  %lit_197 = hask.make_i64(2)
  %app_198 = hask.apSSA(%app_196, %lit_197)
  %app_199 = hask.apSSA(%app_198, @$krep_a1l6)
  hask.return(%app_199)
  }
  hask.func @main {
  %type_200 = hask.make_string("TYPEINFO_ERASED")
  %app_201 = hask.apSSA(%return, %type_200)
  %app_202 = hask.apSSA(%app_201, %$fMonadIO)
  %type_203 = hask.make_string("TYPEINFO_ERASED")
  %app_204 = hask.apSSA(%app_202, %type_203)
  %app_205 = hask.apSSA(%app_204, @"()")
  hask.return(%app_205)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 9, types: 4, coercions: 0, joins: 0/0}
//main:NonrecSum.f
//  :: main:NonrecSum.ConcreteSum -> main:NonrecSum.ConcreteSum
//[LclIdX]
//main:NonrecSum.f
//  = \ (x_a12Z :: main:NonrecSum.ConcreteSum) ->
//      case x_a12Z of {
//        main:NonrecSum.ConcreteLeft i_a130 ->
//          main:NonrecSum.ConcreteRight i_a130;
//        main:NonrecSum.ConcreteRight i_a131 ->
//          main:NonrecSum.ConcreteLeft i_a131
//      }
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.sslone :: main:NonrecSum.ConcreteSum
//[LclIdX]
//main:NonrecSum.sslone = main:NonrecSum.ConcreteLeft 1#
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:NonrecSum.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "NonrecSum"#)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1l9 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1l9
//  = ghc-prim-0.5.3:GHC.Types.$WKindRepVar
//      (ghc-prim-0.5.3:GHC.Types.I# 1#)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1l7 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1l7
//  = ghc-prim-0.5.3:GHC.Types.$WKindRepVar
//      (ghc-prim-0.5.3:GHC.Types.I# 0#)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1lc [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lc
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      ghc-prim-0.5.3:GHC.Types.$tcInt#
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      9627986161123870636##
//      8521208971585772379##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteProd"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1lm [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lm
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteProd
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1ll [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1ll
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1lc $krep_a1lm
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1lk [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lk
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1lc $krep_a1ll
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'MkConcreteProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      12942902748065332888##
//      624046941574007678##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'MkConcreteProd"#)
//      0#
//      $krep_a1lk
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteSum :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteSum
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      11895830767168603041##
//      16356649953315961404##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteSum"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1lj [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lj
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteSum
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1li [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1li
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1lc $krep_a1lj
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteLeft :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteLeft
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      17686946225345529826##
//      11843024788528544323##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteLeft"#)
//      0#
//      $krep_a1li
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRight :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteRight
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      3526397897247831921##
//      16352906645058643170##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteRight"#)
//      0#
//      $krep_a1li
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRec :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteRec
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      9813922555586650147##
//      728115828137284603##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteRec"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1lh [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lh
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRec
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1lg [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lg
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1lh $krep_a1lh
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1lf [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lf
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1lc $krep_a1lg
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteRec :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'MkConcreteRec
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      780870275437019420##
//      1987179208485961632##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'MkConcreteRec"#)
//      0#
//      $krep_a1lf
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRecSum :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteRecSum
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      868865939143367437##
//      4283065319836759626##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteRecSum"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_a1le [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1le
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRecSum
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumNone
//  :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteRecSumNone
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      8932361323873389123##
//      15462504305832975424##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteRecSumNone"#)
//      0#
//      $krep_a1le
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1ld [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1ld
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1le $krep_a1le
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1lb [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1lb
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1lc $krep_a1ld
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumCons
//  :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteRecSumCons
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      782896096195591732##
//      2603195489806365008##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteRecSumCons"#)
//      0#
//      $krep_a1lb
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcAbstractProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcAbstractProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      12056464016441906154##
//      6100392904543390536##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "AbstractProd"#)
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*->*->*
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//$krep_a1la [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1la
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcAbstractProd
//      (ghc-prim-0.5.3:GHC.Types.:
//         @ ghc-prim-0.5.3:GHC.Types.KindRep
//         $krep_a1l7
//         (ghc-prim-0.5.3:GHC.Types.:
//            @ ghc-prim-0.5.3:GHC.Types.KindRep
//            $krep_a1l9
//            (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)))
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1l8 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1l8
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1l9 $krep_a1la
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_a1l6 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_a1l6
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_a1l7 $krep_a1l8
//
//-- RHS size: {terms: 8, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkAbstractProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'MkAbstractProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      16938648863639686556##
//      18023809221364289304##
//      main:NonrecSum.$trModule
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "'MkAbstractProd"#)
//      2#
//      $krep_a1l6
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:NonrecSum.main
//  = base-4.12.0.0:GHC.Base.return
//      @ ghc-prim-0.5.3:GHC.Types.IO
//      base-4.12.0.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.5.3:GHC.Tuple.()
//