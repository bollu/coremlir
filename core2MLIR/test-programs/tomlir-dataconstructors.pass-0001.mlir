// NonrecSum
// Core2MLIR: GenMLIR AfterCorePrep
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
  hask.func @sat_s1ya {
  %lambda_0 = hask.lambdaSSA(%x_s1y6) {
    %case_1 = hask.caseSSA  %x_s1y6
    [@"ConcreteLeft" ->
    {
    ^entry(%wild_s1y7: !hask.untyped, %i_s1y8: !hask.untyped):
      %app_2 = hask.apSSA(@ConcreteRight, %i_s1y8)
    hask.return(%app_2)
    }
    ]
    [@"ConcreteRight" ->
    {
    ^entry(%wild_s1y7: !hask.untyped, %i_s1y9: !hask.untyped):
      %app_3 = hask.apSSA(@ConcreteLeft, %i_s1y9)
    hask.return(%app_3)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @f {
  hask.return(@sat_s1ya)
  }
  hask.func @sslone {
  %lit_4 = hask.make_i64(1)
  %app_5 = hask.apSSA(@ConcreteLeft, %lit_4)
  hask.return(%app_5)
  }
  hask.func @sat_s1ye {
  %lit_6 = hask.make_string("NonrecSum")
  %app_7 = hask.apSSA(%TrNameS, %lit_6)
  hask.return(%app_7)
  }
  hask.func @sat_s1yd {
  %lit_8 = hask.make_string("main")
  %app_9 = hask.apSSA(%TrNameS, %lit_8)
  hask.return(%app_9)
  }
  hask.func @$trModule {
  %app_10 = hask.apSSA(%Module, @sat_s1yd)
  %app_11 = hask.apSSA(%app_10, @sat_s1ye)
  hask.return(%app_11)
  }
  hask.func @sat_s1yg {
  %lit_12 = hask.make_i64(1)
  %app_13 = hask.apSSA(%I#, %lit_12)
  hask.return(%app_13)
  }
  hask.func @$krep_s1yf {
  %app_14 = hask.apSSA(%$WKindRepVar, @sat_s1yg)
  hask.return(%app_14)
  }
  hask.func @sat_s1yi {
  %lit_15 = hask.make_i64(0)
  %app_16 = hask.apSSA(%I#, %lit_15)
  hask.return(%app_16)
  }
  hask.func @$krep_s1yh {
  %app_17 = hask.apSSA(%$WKindRepVar, @sat_s1yi)
  hask.return(%app_17)
  }
  hask.func @$krep_s1yj {
  %app_18 = hask.apSSA(%KindRepTyConApp, %$tcInt#)
  %type_19 = hask.make_string("TYPEINFO_ERASED")
  %app_20 = hask.apSSA(%[], %type_19)
  %app_21 = hask.apSSA(%app_18, %app_20)
  hask.return(%app_21)
  }
  hask.func @sat_s1yl {
  %lit_22 = hask.make_string("ConcreteProd")
  %app_23 = hask.apSSA(%TrNameS, %lit_22)
  hask.return(%app_23)
  }
  hask.func @$tcConcreteProd {
  %lit_24 = 9627986161123870636
  %app_25 = hask.apSSA(%TyCon, %lit_24)
  %lit_26 = 8521208971585772379
  %app_27 = hask.apSSA(%app_25, %lit_26)
  %app_28 = hask.apSSA(%app_27, @$trModule)
  %app_29 = hask.apSSA(%app_28, @sat_s1yl)
  %lit_30 = hask.make_i64(0)
  %app_31 = hask.apSSA(%app_29, %lit_30)
  %app_32 = hask.apSSA(%app_31, %krep$*)
  hask.return(%app_32)
  }
  hask.func @$krep_s1ym {
  %app_33 = hask.apSSA(%KindRepTyConApp, @$tcConcreteProd)
  %type_34 = hask.make_string("TYPEINFO_ERASED")
  %app_35 = hask.apSSA(%[], %type_34)
  %app_36 = hask.apSSA(%app_33, %app_35)
  hask.return(%app_36)
  }
  hask.func @$krep_s1yn {
  %app_37 = hask.apSSA(%KindRepFun, @$krep_s1yj)
  %app_38 = hask.apSSA(%app_37, @$krep_s1ym)
  hask.return(%app_38)
  }
  hask.func @$krep_s1yo {
  %app_39 = hask.apSSA(%KindRepFun, @$krep_s1yj)
  %app_40 = hask.apSSA(%app_39, @$krep_s1yn)
  hask.return(%app_40)
  }
  hask.func @sat_s1yq {
  %lit_41 = hask.make_string("'MkConcreteProd")
  %app_42 = hask.apSSA(%TrNameS, %lit_41)
  hask.return(%app_42)
  }
  hask.func @$tc'MkConcreteProd {
  %lit_43 = 12942902748065332888
  %app_44 = hask.apSSA(%TyCon, %lit_43)
  %lit_45 = 624046941574007678
  %app_46 = hask.apSSA(%app_44, %lit_45)
  %app_47 = hask.apSSA(%app_46, @$trModule)
  %app_48 = hask.apSSA(%app_47, @sat_s1yq)
  %lit_49 = hask.make_i64(0)
  %app_50 = hask.apSSA(%app_48, %lit_49)
  %app_51 = hask.apSSA(%app_50, @$krep_s1yo)
  hask.return(%app_51)
  }
  hask.func @sat_s1ys {
  %lit_52 = hask.make_string("ConcreteSum")
  %app_53 = hask.apSSA(%TrNameS, %lit_52)
  hask.return(%app_53)
  }
  hask.func @$tcConcreteSum {
  %lit_54 = 11895830767168603041
  %app_55 = hask.apSSA(%TyCon, %lit_54)
  %lit_56 = 16356649953315961404
  %app_57 = hask.apSSA(%app_55, %lit_56)
  %app_58 = hask.apSSA(%app_57, @$trModule)
  %app_59 = hask.apSSA(%app_58, @sat_s1ys)
  %lit_60 = hask.make_i64(0)
  %app_61 = hask.apSSA(%app_59, %lit_60)
  %app_62 = hask.apSSA(%app_61, %krep$*)
  hask.return(%app_62)
  }
  hask.func @$krep_s1yt {
  %app_63 = hask.apSSA(%KindRepTyConApp, @$tcConcreteSum)
  %type_64 = hask.make_string("TYPEINFO_ERASED")
  %app_65 = hask.apSSA(%[], %type_64)
  %app_66 = hask.apSSA(%app_63, %app_65)
  hask.return(%app_66)
  }
  hask.func @$krep_s1yu {
  %app_67 = hask.apSSA(%KindRepFun, @$krep_s1yj)
  %app_68 = hask.apSSA(%app_67, @$krep_s1yt)
  hask.return(%app_68)
  }
  hask.func @sat_s1yw {
  %lit_69 = hask.make_string("'ConcreteLeft")
  %app_70 = hask.apSSA(%TrNameS, %lit_69)
  hask.return(%app_70)
  }
  hask.func @$tc'ConcreteLeft {
  %lit_71 = 17686946225345529826
  %app_72 = hask.apSSA(%TyCon, %lit_71)
  %lit_73 = 11843024788528544323
  %app_74 = hask.apSSA(%app_72, %lit_73)
  %app_75 = hask.apSSA(%app_74, @$trModule)
  %app_76 = hask.apSSA(%app_75, @sat_s1yw)
  %lit_77 = hask.make_i64(0)
  %app_78 = hask.apSSA(%app_76, %lit_77)
  %app_79 = hask.apSSA(%app_78, @$krep_s1yu)
  hask.return(%app_79)
  }
  hask.func @sat_s1yy {
  %lit_80 = hask.make_string("'ConcreteRight")
  %app_81 = hask.apSSA(%TrNameS, %lit_80)
  hask.return(%app_81)
  }
  hask.func @$tc'ConcreteRight {
  %lit_82 = 3526397897247831921
  %app_83 = hask.apSSA(%TyCon, %lit_82)
  %lit_84 = 16352906645058643170
  %app_85 = hask.apSSA(%app_83, %lit_84)
  %app_86 = hask.apSSA(%app_85, @$trModule)
  %app_87 = hask.apSSA(%app_86, @sat_s1yy)
  %lit_88 = hask.make_i64(0)
  %app_89 = hask.apSSA(%app_87, %lit_88)
  %app_90 = hask.apSSA(%app_89, @$krep_s1yu)
  hask.return(%app_90)
  }
  hask.func @sat_s1yA {
  %lit_91 = hask.make_string("ConcreteRec")
  %app_92 = hask.apSSA(%TrNameS, %lit_91)
  hask.return(%app_92)
  }
  hask.func @$tcConcreteRec {
  %lit_93 = 9813922555586650147
  %app_94 = hask.apSSA(%TyCon, %lit_93)
  %lit_95 = 728115828137284603
  %app_96 = hask.apSSA(%app_94, %lit_95)
  %app_97 = hask.apSSA(%app_96, @$trModule)
  %app_98 = hask.apSSA(%app_97, @sat_s1yA)
  %lit_99 = hask.make_i64(0)
  %app_100 = hask.apSSA(%app_98, %lit_99)
  %app_101 = hask.apSSA(%app_100, %krep$*)
  hask.return(%app_101)
  }
  hask.func @$krep_s1yB {
  %app_102 = hask.apSSA(%KindRepTyConApp, @$tcConcreteRec)
  %type_103 = hask.make_string("TYPEINFO_ERASED")
  %app_104 = hask.apSSA(%[], %type_103)
  %app_105 = hask.apSSA(%app_102, %app_104)
  hask.return(%app_105)
  }
  hask.func @$krep_s1yC {
  %app_106 = hask.apSSA(%KindRepFun, @$krep_s1yB)
  %app_107 = hask.apSSA(%app_106, @$krep_s1yB)
  hask.return(%app_107)
  }
  hask.func @$krep_s1yD {
  %app_108 = hask.apSSA(%KindRepFun, @$krep_s1yj)
  %app_109 = hask.apSSA(%app_108, @$krep_s1yC)
  hask.return(%app_109)
  }
  hask.func @sat_s1yF {
  %lit_110 = hask.make_string("'MkConcreteRec")
  %app_111 = hask.apSSA(%TrNameS, %lit_110)
  hask.return(%app_111)
  }
  hask.func @$tc'MkConcreteRec {
  %lit_112 = 780870275437019420
  %app_113 = hask.apSSA(%TyCon, %lit_112)
  %lit_114 = 1987179208485961632
  %app_115 = hask.apSSA(%app_113, %lit_114)
  %app_116 = hask.apSSA(%app_115, @$trModule)
  %app_117 = hask.apSSA(%app_116, @sat_s1yF)
  %lit_118 = hask.make_i64(0)
  %app_119 = hask.apSSA(%app_117, %lit_118)
  %app_120 = hask.apSSA(%app_119, @$krep_s1yD)
  hask.return(%app_120)
  }
  hask.func @sat_s1yH {
  %lit_121 = hask.make_string("ConcreteRecSum")
  %app_122 = hask.apSSA(%TrNameS, %lit_121)
  hask.return(%app_122)
  }
  hask.func @$tcConcreteRecSum {
  %lit_123 = 868865939143367437
  %app_124 = hask.apSSA(%TyCon, %lit_123)
  %lit_125 = 4283065319836759626
  %app_126 = hask.apSSA(%app_124, %lit_125)
  %app_127 = hask.apSSA(%app_126, @$trModule)
  %app_128 = hask.apSSA(%app_127, @sat_s1yH)
  %lit_129 = hask.make_i64(0)
  %app_130 = hask.apSSA(%app_128, %lit_129)
  %app_131 = hask.apSSA(%app_130, %krep$*)
  hask.return(%app_131)
  }
  hask.func @$krep_s1yI {
  %app_132 = hask.apSSA(%KindRepTyConApp, @$tcConcreteRecSum)
  %type_133 = hask.make_string("TYPEINFO_ERASED")
  %app_134 = hask.apSSA(%[], %type_133)
  %app_135 = hask.apSSA(%app_132, %app_134)
  hask.return(%app_135)
  }
  hask.func @sat_s1yK {
  %lit_136 = hask.make_string("'ConcreteRecSumNone")
  %app_137 = hask.apSSA(%TrNameS, %lit_136)
  hask.return(%app_137)
  }
  hask.func @$tc'ConcreteRecSumNone {
  %lit_138 = 8932361323873389123
  %app_139 = hask.apSSA(%TyCon, %lit_138)
  %lit_140 = 15462504305832975424
  %app_141 = hask.apSSA(%app_139, %lit_140)
  %app_142 = hask.apSSA(%app_141, @$trModule)
  %app_143 = hask.apSSA(%app_142, @sat_s1yK)
  %lit_144 = hask.make_i64(0)
  %app_145 = hask.apSSA(%app_143, %lit_144)
  %app_146 = hask.apSSA(%app_145, @$krep_s1yI)
  hask.return(%app_146)
  }
  hask.func @$krep_s1yL {
  %app_147 = hask.apSSA(%KindRepFun, @$krep_s1yI)
  %app_148 = hask.apSSA(%app_147, @$krep_s1yI)
  hask.return(%app_148)
  }
  hask.func @$krep_s1yM {
  %app_149 = hask.apSSA(%KindRepFun, @$krep_s1yj)
  %app_150 = hask.apSSA(%app_149, @$krep_s1yL)
  hask.return(%app_150)
  }
  hask.func @sat_s1yO {
  %lit_151 = hask.make_string("'ConcreteRecSumCons")
  %app_152 = hask.apSSA(%TrNameS, %lit_151)
  hask.return(%app_152)
  }
  hask.func @$tc'ConcreteRecSumCons {
  %lit_153 = 782896096195591732
  %app_154 = hask.apSSA(%TyCon, %lit_153)
  %lit_155 = 2603195489806365008
  %app_156 = hask.apSSA(%app_154, %lit_155)
  %app_157 = hask.apSSA(%app_156, @$trModule)
  %app_158 = hask.apSSA(%app_157, @sat_s1yO)
  %lit_159 = hask.make_i64(0)
  %app_160 = hask.apSSA(%app_158, %lit_159)
  %app_161 = hask.apSSA(%app_160, @$krep_s1yM)
  hask.return(%app_161)
  }
  hask.func @sat_s1yQ {
  %lit_162 = hask.make_string("AbstractProd")
  %app_163 = hask.apSSA(%TrNameS, %lit_162)
  hask.return(%app_163)
  }
  hask.func @$tcAbstractProd {
  %lit_164 = 12056464016441906154
  %app_165 = hask.apSSA(%TyCon, %lit_164)
  %lit_166 = 6100392904543390536
  %app_167 = hask.apSSA(%app_165, %lit_166)
  %app_168 = hask.apSSA(%app_167, @$trModule)
  %app_169 = hask.apSSA(%app_168, @sat_s1yQ)
  %lit_170 = hask.make_i64(0)
  %app_171 = hask.apSSA(%app_169, %lit_170)
  %app_172 = hask.apSSA(%app_171, %krep$*->*->*)
  hask.return(%app_172)
  }
  hask.func @sat_s1yS {
  %type_173 = hask.make_string("TYPEINFO_ERASED")
  %app_174 = hask.apSSA(%:, %type_173)
  %app_175 = hask.apSSA(%app_174, @$krep_s1yf)
  %type_176 = hask.make_string("TYPEINFO_ERASED")
  %app_177 = hask.apSSA(%[], %type_176)
  %app_178 = hask.apSSA(%app_175, %app_177)
  hask.return(%app_178)
  }
  hask.func @sat_s1yT {
  %type_179 = hask.make_string("TYPEINFO_ERASED")
  %app_180 = hask.apSSA(%:, %type_179)
  %app_181 = hask.apSSA(%app_180, @$krep_s1yh)
  %app_182 = hask.apSSA(%app_181, @sat_s1yS)
  hask.return(%app_182)
  }
  hask.func @$krep_s1yR {
  %app_183 = hask.apSSA(%KindRepTyConApp, @$tcAbstractProd)
  %app_184 = hask.apSSA(%app_183, @sat_s1yT)
  hask.return(%app_184)
  }
  hask.func @$krep_s1yU {
  %app_185 = hask.apSSA(%KindRepFun, @$krep_s1yf)
  %app_186 = hask.apSSA(%app_185, @$krep_s1yR)
  hask.return(%app_186)
  }
  hask.func @$krep_s1yV {
  %app_187 = hask.apSSA(%KindRepFun, @$krep_s1yh)
  %app_188 = hask.apSSA(%app_187, @$krep_s1yU)
  hask.return(%app_188)
  }
  hask.func @sat_s1yX {
  %lit_189 = hask.make_string("'MkAbstractProd")
  %app_190 = hask.apSSA(%TrNameS, %lit_189)
  hask.return(%app_190)
  }
  hask.func @$tc'MkAbstractProd {
  %lit_191 = 16938648863639686556
  %app_192 = hask.apSSA(%TyCon, %lit_191)
  %lit_193 = 18023809221364289304
  %app_194 = hask.apSSA(%app_192, %lit_193)
  %app_195 = hask.apSSA(%app_194, @$trModule)
  %app_196 = hask.apSSA(%app_195, @sat_s1yX)
  %lit_197 = hask.make_i64(2)
  %app_198 = hask.apSSA(%app_196, %lit_197)
  %app_199 = hask.apSSA(%app_198, @$krep_s1yV)
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
  hask.func @MkAbstractProd {
  %lambda_206 = hask.lambdaSSA(%a_azg) {
    %lambda_207 = hask.lambdaSSA(%b_azh) {
      %lambda_208 = hask.lambdaSSA(%eta_B2) {
        %lambda_209 = hask.lambdaSSA(%eta_B1) {
          %type_210 = hask.make_string("TYPEINFO_ERASED")
          %app_211 = hask.apSSA(@MkAbstractProd, %type_210)
          %type_212 = hask.make_string("TYPEINFO_ERASED")
          %app_213 = hask.apSSA(%app_211, %type_212)
          %app_214 = hask.apSSA(%app_213, %eta_B2)
          %app_215 = hask.apSSA(%app_214, %eta_B1)
          hask.return(%app_215)
        }
        hask.return(%lambda_209)
      }
      hask.return(%lambda_208)
    }
    hask.return(%lambda_207)
  }
  hask.return(%lambda_206)
  }
  hask.func @ConcreteRecSumCons {
  %lambda_216 = hask.lambdaSSA(%eta_B2) {
    %lambda_217 = hask.lambdaSSA(%eta_B1) {
      %app_218 = hask.apSSA(@ConcreteRecSumCons, %eta_B2)
      %app_219 = hask.apSSA(%app_218, %eta_B1)
      hask.return(%app_219)
    }
    hask.return(%lambda_217)
  }
  hask.return(%lambda_216)
  }
  hask.func @ConcreteRecSumNone {
  hask.return(@ConcreteRecSumNone)
  }
  hask.func @MkConcreteRec {
  %lambda_220 = hask.lambdaSSA(%eta_B2) {
    %lambda_221 = hask.lambdaSSA(%eta_B1) {
      %app_222 = hask.apSSA(@MkConcreteRec, %eta_B2)
      %app_223 = hask.apSSA(%app_222, %eta_B1)
      hask.return(%app_223)
    }
    hask.return(%lambda_221)
  }
  hask.return(%lambda_220)
  }
  hask.func @ConcreteLeft {
  %lambda_224 = hask.lambdaSSA(%eta_B1) {
    %app_225 = hask.apSSA(@ConcreteLeft, %eta_B1)
    hask.return(%app_225)
  }
  hask.return(%lambda_224)
  }
  hask.func @ConcreteRight {
  %lambda_226 = hask.lambdaSSA(%eta_B1) {
    %app_227 = hask.apSSA(@ConcreteRight, %eta_B1)
    hask.return(%app_227)
  }
  hask.return(%lambda_226)
  }
  hask.func @MkConcreteProd {
  %lambda_228 = hask.lambdaSSA(%eta_B2) {
    %lambda_229 = hask.lambdaSSA(%eta_B1) {
      %app_230 = hask.apSSA(@MkConcreteProd, %eta_B2)
      %app_231 = hask.apSSA(%app_230, %eta_B1)
      hask.return(%app_231)
    }
    hask.return(%lambda_229)
  }
  hask.return(%lambda_228)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 9, types: 4, coercions: 0, joins: 0/0}
//sat_s1ya
//  :: main:NonrecSum.ConcreteSum -> main:NonrecSum.ConcreteSum
//[LclId]
//sat_s1ya
//  = \ (x_s1y6 [Occ=Once!] :: main:NonrecSum.ConcreteSum) ->
//      case x_s1y6 of {
//        main:NonrecSum.ConcreteLeft i_s1y8 [Occ=Once] ->
//          main:NonrecSum.ConcreteRight i_s1y8;
//        main:NonrecSum.ConcreteRight i_s1y9 [Occ=Once] ->
//          main:NonrecSum.ConcreteLeft i_s1y9
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.f
//  :: main:NonrecSum.ConcreteSum -> main:NonrecSum.ConcreteSum
//[LclIdX]
//main:NonrecSum.f = sat_s1ya
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.sslone :: main:NonrecSum.ConcreteSum
//[LclIdX]
//main:NonrecSum.sslone = main:NonrecSum.ConcreteLeft 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1ye :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1ye = ghc-prim-0.5.3:GHC.Types.TrNameS "NonrecSum"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yd :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yd = ghc-prim-0.5.3:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:NonrecSum.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module sat_s1yd sat_s1ye
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yg :: ghc-prim-0.5.3:GHC.Types.KindBndr
//[LclId]
//sat_s1yg = ghc-prim-0.5.3:GHC.Types.I# 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yf [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yf = ghc-prim-0.5.3:GHC.Types.$WKindRepVar sat_s1yg
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yi :: ghc-prim-0.5.3:GHC.Types.KindBndr
//[LclId]
//sat_s1yi = ghc-prim-0.5.3:GHC.Types.I# 0#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yh [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yh = ghc-prim-0.5.3:GHC.Types.$WKindRepVar sat_s1yi
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1yj [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yj
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      ghc-prim-0.5.3:GHC.Types.$tcInt#
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yl :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yl = ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      9627986161123870636##
//      8521208971585772379##
//      main:NonrecSum.$trModule
//      sat_s1yl
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1ym [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1ym
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteProd
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yn [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yn
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yj $krep_s1ym
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yo [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yo
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yj $krep_s1yn
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yq :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yq = ghc-prim-0.5.3:GHC.Types.TrNameS "'MkConcreteProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'MkConcreteProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      12942902748065332888##
//      624046941574007678##
//      main:NonrecSum.$trModule
//      sat_s1yq
//      0#
//      $krep_s1yo
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1ys :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1ys = ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteSum"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteSum :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteSum
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      11895830767168603041##
//      16356649953315961404##
//      main:NonrecSum.$trModule
//      sat_s1ys
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1yt [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yt
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteSum
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yu [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yu
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yj $krep_s1yt
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yw :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yw = ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteLeft"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteLeft :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteLeft
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      17686946225345529826##
//      11843024788528544323##
//      main:NonrecSum.$trModule
//      sat_s1yw
//      0#
//      $krep_s1yu
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yy :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yy = ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteRight"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRight :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteRight
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      3526397897247831921##
//      16352906645058643170##
//      main:NonrecSum.$trModule
//      sat_s1yy
//      0#
//      $krep_s1yu
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yA :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yA = ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteRec"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRec :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteRec
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      9813922555586650147##
//      728115828137284603##
//      main:NonrecSum.$trModule
//      sat_s1yA
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1yB [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yB
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRec
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yC [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yC
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yB $krep_s1yB
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yD [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yD
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yj $krep_s1yC
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yF :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yF = ghc-prim-0.5.3:GHC.Types.TrNameS "'MkConcreteRec"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteRec :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'MkConcreteRec
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      780870275437019420##
//      1987179208485961632##
//      main:NonrecSum.$trModule
//      sat_s1yF
//      0#
//      $krep_s1yD
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yH :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yH = ghc-prim-0.5.3:GHC.Types.TrNameS "ConcreteRecSum"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRecSum :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcConcreteRecSum
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      868865939143367437##
//      4283065319836759626##
//      main:NonrecSum.$trModule
//      sat_s1yH
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_s1yI [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yI
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRecSum
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yK :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yK = ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteRecSumNone"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumNone
//  :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteRecSumNone
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      8932361323873389123##
//      15462504305832975424##
//      main:NonrecSum.$trModule
//      sat_s1yK
//      0#
//      $krep_s1yI
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yL [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yL
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yI $krep_s1yI
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yM [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yM
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yj $krep_s1yL
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yO :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yO = ghc-prim-0.5.3:GHC.Types.TrNameS "'ConcreteRecSumCons"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumCons
//  :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'ConcreteRecSumCons
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      782896096195591732##
//      2603195489806365008##
//      main:NonrecSum.$trModule
//      sat_s1yO
//      0#
//      $krep_s1yM
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yQ :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yQ = ghc-prim-0.5.3:GHC.Types.TrNameS "AbstractProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcAbstractProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tcAbstractProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      12056464016441906154##
//      6100392904543390536##
//      main:NonrecSum.$trModule
//      sat_s1yQ
//      0#
//      ghc-prim-0.5.3:GHC.Types.krep$*->*->*
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//sat_s1yS :: [ghc-prim-0.5.3:GHC.Types.KindRep]
//[LclId]
//sat_s1yS
//  = ghc-prim-0.5.3:GHC.Types.:
//      @ ghc-prim-0.5.3:GHC.Types.KindRep
//      $krep_s1yf
//      (ghc-prim-0.5.3:GHC.Types.[] @ ghc-prim-0.5.3:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//sat_s1yT :: [ghc-prim-0.5.3:GHC.Types.KindRep]
//[LclId]
//sat_s1yT
//  = ghc-prim-0.5.3:GHC.Types.:
//      @ ghc-prim-0.5.3:GHC.Types.KindRep $krep_s1yh sat_s1yS
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yR [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yR
//  = ghc-prim-0.5.3:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcAbstractProd sat_s1yT
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yU [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yU
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yf $krep_s1yR
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_s1yV [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.5.3:GHC.Types.KindRep
//[LclId]
//$krep_s1yV
//  = ghc-prim-0.5.3:GHC.Types.KindRepFun $krep_s1yh $krep_s1yU
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_s1yX :: ghc-prim-0.5.3:GHC.Types.TrName
//[LclId]
//sat_s1yX = ghc-prim-0.5.3:GHC.Types.TrNameS "'MkAbstractProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkAbstractProd :: ghc-prim-0.5.3:GHC.Types.TyCon
//[LclIdX]
//main:NonrecSum.$tc'MkAbstractProd
//  = ghc-prim-0.5.3:GHC.Types.TyCon
//      16938648863639686556##
//      18023809221364289304##
//      main:NonrecSum.$trModule
//      sat_s1yX
//      2#
//      $krep_s1yV
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
//-- RHS size: {terms: 7, types: 8, coercions: 0, joins: 0/0}
//main:NonrecSum.MkAbstractProd
//  :: forall a b. a -> b -> main:NonrecSum.AbstractProd a b
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m,
// Unf=OtherCon []]
//main:NonrecSum.MkAbstractProd
//  = \ (@ a_azg)
//      (@ b_azh)
//      (eta_B2 [Occ=Once] :: a_azg[sk:0])
//      (eta_B1 [Occ=Once] :: b_azh[sk:0]) ->
//      main:NonrecSum.MkAbstractProd @ a_azg @ b_azh eta_B2 eta_B1
//
//-- RHS size: {terms: 5, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteRecSumCons
//  :: ghc-prim-0.5.3:GHC.Prim.Int#
//     -> main:NonrecSum.ConcreteRecSum -> main:NonrecSum.ConcreteRecSum
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m1,
// Unf=OtherCon []]
//main:NonrecSum.ConcreteRecSumCons
//  = \ (eta_B2 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#)
//      (eta_B1 [Occ=Once] :: main:NonrecSum.ConcreteRecSum) ->
//      main:NonrecSum.ConcreteRecSumCons eta_B2 eta_B1
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteRecSumNone :: main:NonrecSum.ConcreteRecSum
//[GblId[DataCon], Caf=NoCafRefs, Unf=OtherCon []]
//main:NonrecSum.ConcreteRecSumNone
//  = main:NonrecSum.ConcreteRecSumNone
//
//-- RHS size: {terms: 5, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.MkConcreteRec
//  :: ghc-prim-0.5.3:GHC.Prim.Int#
//     -> main:NonrecSum.ConcreteRec -> main:NonrecSum.ConcreteRec
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m,
// Unf=OtherCon []]
//main:NonrecSum.MkConcreteRec
//  = \ (eta_B2 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#)
//      (eta_B1 [Occ=Once] :: main:NonrecSum.ConcreteRec) ->
//      main:NonrecSum.MkConcreteRec eta_B2 eta_B1
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteLeft
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> main:NonrecSum.ConcreteSum
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m1,
// Unf=OtherCon []]
//main:NonrecSum.ConcreteLeft
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      main:NonrecSum.ConcreteLeft eta_B1
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteRight
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> main:NonrecSum.ConcreteSum
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m2,
// Unf=OtherCon []]
//main:NonrecSum.ConcreteRight
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      main:NonrecSum.ConcreteRight eta_B1
//
//-- RHS size: {terms: 5, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.MkConcreteProd
//  :: ghc-prim-0.5.3:GHC.Prim.Int#
//     -> ghc-prim-0.5.3:GHC.Prim.Int# -> main:NonrecSum.ConcreteProd
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m,
// Unf=OtherCon []]
//main:NonrecSum.MkConcreteProd
//  = \ (eta_B2 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#)
//      (eta_B1 [Occ=Once] :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      main:NonrecSum.MkConcreteProd eta_B2 eta_B1
//