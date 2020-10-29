// NonrecSum
// Core2MLIR: GenMLIR AfterCorePrep
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
  hask.func @sat_sN2 {
  %lambda_0 = hask.lambda(%x_sMY) {
    %case_1 = hask.case  %x_sMY
    [@"ConcreteLeft" ->
    {
    ^entry(%wild_sMZ: !hask.untyped, %i_sN0: !hask.untyped):
      %app_2 = hask.ap(@ConcreteRight, %i_sN0)
    hask.return(%app_2)
    }
    ]
    [@"ConcreteRight" ->
    {
    ^entry(%wild_sMZ: !hask.untyped, %i_sN1: !hask.untyped):
      %app_3 = hask.ap(@ConcreteLeft, %i_sN1)
    hask.return(%app_3)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
  hask.func @f {
  hask.return(@sat_sN2)
  }
  hask.func @sslone {
  %lit_4 = hask.make_i64(1)
  %app_5 = hask.ap(@ConcreteLeft, %lit_4)
  hask.return(%app_5)
  }
  hask.func @sat_sN6 {
  %lit_6 = "NonrecSum"#
  %app_7 = hask.ap(%TrNameS, %lit_6)
  hask.return(%app_7)
  }
  hask.func @sat_sN5 {
  %lit_8 = "main"#
  %app_9 = hask.ap(%TrNameS, %lit_8)
  hask.return(%app_9)
  }
  hask.func @$trModule {
  %app_10 = hask.ap(%Module, @sat_sN5)
  %app_11 = hask.ap(%app_10, @sat_sN6)
  hask.return(%app_11)
  }
  hask.func @sat_sN8 {
  %lit_12 = hask.make_i64(1)
  %app_13 = hask.ap(%I#, %lit_12)
  hask.return(%app_13)
  }
  hask.func @$krep_sN7 {
  %app_14 = hask.ap(%$WKindRepVar, @sat_sN8)
  hask.return(%app_14)
  }
  hask.func @sat_sNa {
  %lit_15 = hask.make_i64(0)
  %app_16 = hask.ap(%I#, %lit_15)
  hask.return(%app_16)
  }
  hask.func @$krep_sN9 {
  %app_17 = hask.ap(%$WKindRepVar, @sat_sNa)
  hask.return(%app_17)
  }
  hask.func @$krep_sNb {
  %app_18 = hask.ap(%KindRepTyConApp, %$tcInt#)
  %type_19 = hask.make_string("TYPEINFO_ERASED")
  %app_20 = hask.ap(%[], %type_19)
  %app_21 = hask.ap(%app_18, %app_20)
  hask.return(%app_21)
  }
  hask.func @sat_sNd {
  %lit_22 = "ConcreteProd"#
  %app_23 = hask.ap(%TrNameS, %lit_22)
  hask.return(%app_23)
  }
  hask.func @$tcConcreteProd {
  %lit_24 = 9627986161123870636
  %app_25 = hask.ap(%TyCon, %lit_24)
  %lit_26 = 8521208971585772379
  %app_27 = hask.ap(%app_25, %lit_26)
  %app_28 = hask.ap(%app_27, @$trModule)
  %app_29 = hask.ap(%app_28, @sat_sNd)
  %lit_30 = hask.make_i64(0)
  %app_31 = hask.ap(%app_29, %lit_30)
  %app_32 = hask.ap(%app_31, %krep$*)
  hask.return(%app_32)
  }
  hask.func @$krep_sNe {
  %app_33 = hask.ap(%KindRepTyConApp, @$tcConcreteProd)
  %type_34 = hask.make_string("TYPEINFO_ERASED")
  %app_35 = hask.ap(%[], %type_34)
  %app_36 = hask.ap(%app_33, %app_35)
  hask.return(%app_36)
  }
  hask.func @$krep_sNf {
  %app_37 = hask.ap(%KindRepFun, @$krep_sNb)
  %app_38 = hask.ap(%app_37, @$krep_sNe)
  hask.return(%app_38)
  }
  hask.func @$krep_sNg {
  %app_39 = hask.ap(%KindRepFun, @$krep_sNb)
  %app_40 = hask.ap(%app_39, @$krep_sNf)
  hask.return(%app_40)
  }
  hask.func @sat_sNi {
  %lit_41 = "'MkConcreteProd"#
  %app_42 = hask.ap(%TrNameS, %lit_41)
  hask.return(%app_42)
  }
  hask.func @$tc'MkConcreteProd {
  %lit_43 = 12942902748065332888
  %app_44 = hask.ap(%TyCon, %lit_43)
  %lit_45 = 624046941574007678
  %app_46 = hask.ap(%app_44, %lit_45)
  %app_47 = hask.ap(%app_46, @$trModule)
  %app_48 = hask.ap(%app_47, @sat_sNi)
  %lit_49 = hask.make_i64(0)
  %app_50 = hask.ap(%app_48, %lit_49)
  %app_51 = hask.ap(%app_50, @$krep_sNg)
  hask.return(%app_51)
  }
  hask.func @sat_sNk {
  %lit_52 = "ConcreteSum"#
  %app_53 = hask.ap(%TrNameS, %lit_52)
  hask.return(%app_53)
  }
  hask.func @$tcConcreteSum {
  %lit_54 = 11895830767168603041
  %app_55 = hask.ap(%TyCon, %lit_54)
  %lit_56 = 16356649953315961404
  %app_57 = hask.ap(%app_55, %lit_56)
  %app_58 = hask.ap(%app_57, @$trModule)
  %app_59 = hask.ap(%app_58, @sat_sNk)
  %lit_60 = hask.make_i64(0)
  %app_61 = hask.ap(%app_59, %lit_60)
  %app_62 = hask.ap(%app_61, %krep$*)
  hask.return(%app_62)
  }
  hask.func @$krep_sNl {
  %app_63 = hask.ap(%KindRepTyConApp, @$tcConcreteSum)
  %type_64 = hask.make_string("TYPEINFO_ERASED")
  %app_65 = hask.ap(%[], %type_64)
  %app_66 = hask.ap(%app_63, %app_65)
  hask.return(%app_66)
  }
  hask.func @$krep_sNm {
  %app_67 = hask.ap(%KindRepFun, @$krep_sNb)
  %app_68 = hask.ap(%app_67, @$krep_sNl)
  hask.return(%app_68)
  }
  hask.func @sat_sNo {
  %lit_69 = "'ConcreteLeft"#
  %app_70 = hask.ap(%TrNameS, %lit_69)
  hask.return(%app_70)
  }
  hask.func @$tc'ConcreteLeft {
  %lit_71 = 17686946225345529826
  %app_72 = hask.ap(%TyCon, %lit_71)
  %lit_73 = 11843024788528544323
  %app_74 = hask.ap(%app_72, %lit_73)
  %app_75 = hask.ap(%app_74, @$trModule)
  %app_76 = hask.ap(%app_75, @sat_sNo)
  %lit_77 = hask.make_i64(0)
  %app_78 = hask.ap(%app_76, %lit_77)
  %app_79 = hask.ap(%app_78, @$krep_sNm)
  hask.return(%app_79)
  }
  hask.func @sat_sNq {
  %lit_80 = "'ConcreteRight"#
  %app_81 = hask.ap(%TrNameS, %lit_80)
  hask.return(%app_81)
  }
  hask.func @$tc'ConcreteRight {
  %lit_82 = 3526397897247831921
  %app_83 = hask.ap(%TyCon, %lit_82)
  %lit_84 = 16352906645058643170
  %app_85 = hask.ap(%app_83, %lit_84)
  %app_86 = hask.ap(%app_85, @$trModule)
  %app_87 = hask.ap(%app_86, @sat_sNq)
  %lit_88 = hask.make_i64(0)
  %app_89 = hask.ap(%app_87, %lit_88)
  %app_90 = hask.ap(%app_89, @$krep_sNm)
  hask.return(%app_90)
  }
  hask.func @sat_sNs {
  %lit_91 = "ConcreteRec"#
  %app_92 = hask.ap(%TrNameS, %lit_91)
  hask.return(%app_92)
  }
  hask.func @$tcConcreteRec {
  %lit_93 = 9813922555586650147
  %app_94 = hask.ap(%TyCon, %lit_93)
  %lit_95 = 728115828137284603
  %app_96 = hask.ap(%app_94, %lit_95)
  %app_97 = hask.ap(%app_96, @$trModule)
  %app_98 = hask.ap(%app_97, @sat_sNs)
  %lit_99 = hask.make_i64(0)
  %app_100 = hask.ap(%app_98, %lit_99)
  %app_101 = hask.ap(%app_100, %krep$*)
  hask.return(%app_101)
  }
  hask.func @$krep_sNt {
  %app_102 = hask.ap(%KindRepTyConApp, @$tcConcreteRec)
  %type_103 = hask.make_string("TYPEINFO_ERASED")
  %app_104 = hask.ap(%[], %type_103)
  %app_105 = hask.ap(%app_102, %app_104)
  hask.return(%app_105)
  }
  hask.func @$krep_sNu {
  %app_106 = hask.ap(%KindRepFun, @$krep_sNt)
  %app_107 = hask.ap(%app_106, @$krep_sNt)
  hask.return(%app_107)
  }
  hask.func @$krep_sNv {
  %app_108 = hask.ap(%KindRepFun, @$krep_sNb)
  %app_109 = hask.ap(%app_108, @$krep_sNu)
  hask.return(%app_109)
  }
  hask.func @sat_sNx {
  %lit_110 = "'MkConcreteRec"#
  %app_111 = hask.ap(%TrNameS, %lit_110)
  hask.return(%app_111)
  }
  hask.func @$tc'MkConcreteRec {
  %lit_112 = 780870275437019420
  %app_113 = hask.ap(%TyCon, %lit_112)
  %lit_114 = 1987179208485961632
  %app_115 = hask.ap(%app_113, %lit_114)
  %app_116 = hask.ap(%app_115, @$trModule)
  %app_117 = hask.ap(%app_116, @sat_sNx)
  %lit_118 = hask.make_i64(0)
  %app_119 = hask.ap(%app_117, %lit_118)
  %app_120 = hask.ap(%app_119, @$krep_sNv)
  hask.return(%app_120)
  }
  hask.func @sat_sNz {
  %lit_121 = "ConcreteRecSum"#
  %app_122 = hask.ap(%TrNameS, %lit_121)
  hask.return(%app_122)
  }
  hask.func @$tcConcreteRecSum {
  %lit_123 = 868865939143367437
  %app_124 = hask.ap(%TyCon, %lit_123)
  %lit_125 = 4283065319836759626
  %app_126 = hask.ap(%app_124, %lit_125)
  %app_127 = hask.ap(%app_126, @$trModule)
  %app_128 = hask.ap(%app_127, @sat_sNz)
  %lit_129 = hask.make_i64(0)
  %app_130 = hask.ap(%app_128, %lit_129)
  %app_131 = hask.ap(%app_130, %krep$*)
  hask.return(%app_131)
  }
  hask.func @$krep_sNA {
  %app_132 = hask.ap(%KindRepTyConApp, @$tcConcreteRecSum)
  %type_133 = hask.make_string("TYPEINFO_ERASED")
  %app_134 = hask.ap(%[], %type_133)
  %app_135 = hask.ap(%app_132, %app_134)
  hask.return(%app_135)
  }
  hask.func @sat_sNC {
  %lit_136 = "'ConcreteRecSumNone"#
  %app_137 = hask.ap(%TrNameS, %lit_136)
  hask.return(%app_137)
  }
  hask.func @$tc'ConcreteRecSumNone {
  %lit_138 = 8932361323873389123
  %app_139 = hask.ap(%TyCon, %lit_138)
  %lit_140 = 15462504305832975424
  %app_141 = hask.ap(%app_139, %lit_140)
  %app_142 = hask.ap(%app_141, @$trModule)
  %app_143 = hask.ap(%app_142, @sat_sNC)
  %lit_144 = hask.make_i64(0)
  %app_145 = hask.ap(%app_143, %lit_144)
  %app_146 = hask.ap(%app_145, @$krep_sNA)
  hask.return(%app_146)
  }
  hask.func @$krep_sND {
  %app_147 = hask.ap(%KindRepFun, @$krep_sNA)
  %app_148 = hask.ap(%app_147, @$krep_sNA)
  hask.return(%app_148)
  }
  hask.func @$krep_sNE {
  %app_149 = hask.ap(%KindRepFun, @$krep_sNb)
  %app_150 = hask.ap(%app_149, @$krep_sND)
  hask.return(%app_150)
  }
  hask.func @sat_sNG {
  %lit_151 = "'ConcreteRecSumCons"#
  %app_152 = hask.ap(%TrNameS, %lit_151)
  hask.return(%app_152)
  }
  hask.func @$tc'ConcreteRecSumCons {
  %lit_153 = 782896096195591732
  %app_154 = hask.ap(%TyCon, %lit_153)
  %lit_155 = 2603195489806365008
  %app_156 = hask.ap(%app_154, %lit_155)
  %app_157 = hask.ap(%app_156, @$trModule)
  %app_158 = hask.ap(%app_157, @sat_sNG)
  %lit_159 = hask.make_i64(0)
  %app_160 = hask.ap(%app_158, %lit_159)
  %app_161 = hask.ap(%app_160, @$krep_sNE)
  hask.return(%app_161)
  }
  hask.func @sat_sNI {
  %lit_162 = "AbstractProd"#
  %app_163 = hask.ap(%TrNameS, %lit_162)
  hask.return(%app_163)
  }
  hask.func @$tcAbstractProd {
  %lit_164 = 12056464016441906154
  %app_165 = hask.ap(%TyCon, %lit_164)
  %lit_166 = 6100392904543390536
  %app_167 = hask.ap(%app_165, %lit_166)
  %app_168 = hask.ap(%app_167, @$trModule)
  %app_169 = hask.ap(%app_168, @sat_sNI)
  %lit_170 = hask.make_i64(0)
  %app_171 = hask.ap(%app_169, %lit_170)
  %app_172 = hask.ap(%app_171, %krep$*->*->*)
  hask.return(%app_172)
  }
  hask.func @sat_sNK {
  %type_173 = hask.make_string("TYPEINFO_ERASED")
  %app_174 = hask.ap(%:, %type_173)
  %app_175 = hask.ap(%app_174, @$krep_sN7)
  %type_176 = hask.make_string("TYPEINFO_ERASED")
  %app_177 = hask.ap(%[], %type_176)
  %app_178 = hask.ap(%app_175, %app_177)
  hask.return(%app_178)
  }
  hask.func @sat_sNL {
  %type_179 = hask.make_string("TYPEINFO_ERASED")
  %app_180 = hask.ap(%:, %type_179)
  %app_181 = hask.ap(%app_180, @$krep_sN9)
  %app_182 = hask.ap(%app_181, @sat_sNK)
  hask.return(%app_182)
  }
  hask.func @$krep_sNJ {
  %app_183 = hask.ap(%KindRepTyConApp, @$tcAbstractProd)
  %app_184 = hask.ap(%app_183, @sat_sNL)
  hask.return(%app_184)
  }
  hask.func @$krep_sNM {
  %app_185 = hask.ap(%KindRepFun, @$krep_sN7)
  %app_186 = hask.ap(%app_185, @$krep_sNJ)
  hask.return(%app_186)
  }
  hask.func @$krep_sNN {
  %app_187 = hask.ap(%KindRepFun, @$krep_sN9)
  %app_188 = hask.ap(%app_187, @$krep_sNM)
  hask.return(%app_188)
  }
  hask.func @sat_sNP {
  %lit_189 = "'MkAbstractProd"#
  %app_190 = hask.ap(%TrNameS, %lit_189)
  hask.return(%app_190)
  }
  hask.func @$tc'MkAbstractProd {
  %lit_191 = 16938648863639686556
  %app_192 = hask.ap(%TyCon, %lit_191)
  %lit_193 = 18023809221364289304
  %app_194 = hask.ap(%app_192, %lit_193)
  %app_195 = hask.ap(%app_194, @$trModule)
  %app_196 = hask.ap(%app_195, @sat_sNP)
  %lit_197 = hask.make_i64(2)
  %app_198 = hask.ap(%app_196, %lit_197)
  %app_199 = hask.ap(%app_198, @$krep_sNN)
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
  hask.func @MkAbstractProd {
  %lambda_206 = hask.lambda(%a_akq) {
    %lambda_207 = hask.lambda(%b_akr) {
      %lambda_208 = hask.lambda(%eta_B2) {
        %lambda_209 = hask.lambda(%eta_B1) {
          %type_210 = hask.make_string("TYPEINFO_ERASED")
          %app_211 = hask.ap(@MkAbstractProd, %type_210)
          %type_212 = hask.make_string("TYPEINFO_ERASED")
          %app_213 = hask.ap(%app_211, %type_212)
          %app_214 = hask.ap(%app_213, %eta_B2)
          %app_215 = hask.ap(%app_214, %eta_B1)
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
  %lambda_216 = hask.lambda(%eta_B2) {
    %lambda_217 = hask.lambda(%eta_B1) {
      %app_218 = hask.ap(@ConcreteRecSumCons, %eta_B2)
      %app_219 = hask.ap(%app_218, %eta_B1)
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
  %lambda_220 = hask.lambda(%eta_B2) {
    %lambda_221 = hask.lambda(%eta_B1) {
      %app_222 = hask.ap(@MkConcreteRec, %eta_B2)
      %app_223 = hask.ap(%app_222, %eta_B1)
      hask.return(%app_223)
    }
    hask.return(%lambda_221)
  }
  hask.return(%lambda_220)
  }
  hask.func @ConcreteLeft {
  %lambda_224 = hask.lambda(%eta_B1) {
    %app_225 = hask.ap(@ConcreteLeft, %eta_B1)
    hask.return(%app_225)
  }
  hask.return(%lambda_224)
  }
  hask.func @ConcreteRight {
  %lambda_226 = hask.lambda(%eta_B1) {
    %app_227 = hask.ap(@ConcreteRight, %eta_B1)
    hask.return(%app_227)
  }
  hask.return(%lambda_226)
  }
  hask.func @MkConcreteProd {
  %lambda_228 = hask.lambda(%eta_B2) {
    %lambda_229 = hask.lambda(%eta_B1) {
      %app_230 = hask.ap(@MkConcreteProd, %eta_B2)
      %app_231 = hask.ap(%app_230, %eta_B1)
      hask.return(%app_231)
    }
    hask.return(%lambda_229)
  }
  hask.return(%lambda_228)
  }
}
// ============ Haskell Core ========================
//-- RHS size: {terms: 9, types: 4, coercions: 0, joins: 0/0}
//sat_sN2 :: main:NonrecSum.ConcreteSum -> main:NonrecSum.ConcreteSum
//[LclId]
//sat_sN2
//  = \ (x_sMY [Occ=Once!] :: main:NonrecSum.ConcreteSum) ->
//      case x_sMY of {
//        main:NonrecSum.ConcreteLeft i_sN0 [Occ=Once] ->
//          main:NonrecSum.ConcreteRight i_sN0;
//        main:NonrecSum.ConcreteRight i_sN1 [Occ=Once] ->
//          main:NonrecSum.ConcreteLeft i_sN1
//      }
//
//-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.f
//  :: main:NonrecSum.ConcreteSum -> main:NonrecSum.ConcreteSum
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.f = sat_sN2
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.sslone :: main:NonrecSum.ConcreteSum
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.sslone = main:NonrecSum.ConcreteLeft 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sN6 :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sN6 = ghc-prim-0.6.1:GHC.Types.TrNameS "NonrecSum"#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sN5 :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sN5 = ghc-prim-0.6.1:GHC.Types.TrNameS "main"#
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$trModule :: ghc-prim-0.6.1:GHC.Types.Module
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$trModule
//  = ghc-prim-0.6.1:GHC.Types.Module sat_sN5 sat_sN6
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sN8 :: ghc-prim-0.6.1:GHC.Types.KindBndr
//[LclId]
//sat_sN8 = ghc-prim-0.6.1:GHC.Types.I# 1#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//$krep_sN7 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId]
//$krep_sN7 = ghc-prim-0.6.1:GHC.Types.$WKindRepVar sat_sN8
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNa :: ghc-prim-0.6.1:GHC.Types.KindBndr
//[LclId]
//sat_sNa = ghc-prim-0.6.1:GHC.Types.I# 0#
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//$krep_sN9 [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId]
//$krep_sN9 = ghc-prim-0.6.1:GHC.Types.$WKindRepVar sat_sNa
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sNb [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNb
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      ghc-prim-0.6.1:GHC.Types.$tcInt#
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNd :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNd = ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tcConcreteProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      9627986161123870636##
//      8521208971585772379##
//      main:NonrecSum.$trModule
//      sat_sNd
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sNe [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNe
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteProd
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNf [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNf = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNb $krep_sNe
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNg [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNg = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNb $krep_sNf
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNi :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNi = ghc-prim-0.6.1:GHC.Types.TrNameS "'MkConcreteProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'MkConcreteProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      12942902748065332888##
//      624046941574007678##
//      main:NonrecSum.$trModule
//      sat_sNi
//      0#
//      $krep_sNg
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNk :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNk = ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteSum"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteSum :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tcConcreteSum
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      11895830767168603041##
//      16356649953315961404##
//      main:NonrecSum.$trModule
//      sat_sNk
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sNl [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNl
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteSum
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNm [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNm = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNb $krep_sNl
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNo :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNo = ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteLeft"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteLeft :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'ConcreteLeft
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      17686946225345529826##
//      11843024788528544323##
//      main:NonrecSum.$trModule
//      sat_sNo
//      0#
//      $krep_sNm
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNq :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNq = ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteRight"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRight :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'ConcreteRight
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      3526397897247831921##
//      16352906645058643170##
//      main:NonrecSum.$trModule
//      sat_sNq
//      0#
//      $krep_sNm
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNs :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNs = ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteRec"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRec :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tcConcreteRec
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      9813922555586650147##
//      728115828137284603##
//      main:NonrecSum.$trModule
//      sat_sNs
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sNt [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNt
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRec
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNu [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNu = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNt $krep_sNt
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNv [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNv = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNb $krep_sNu
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNx :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNx = ghc-prim-0.6.1:GHC.Types.TrNameS "'MkConcreteRec"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkConcreteRec :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'MkConcreteRec
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      780870275437019420##
//      1987179208485961632##
//      main:NonrecSum.$trModule
//      sat_sNx
//      0#
//      $krep_sNv
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNz :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNz = ghc-prim-0.6.1:GHC.Types.TrNameS "ConcreteRecSum"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcConcreteRecSum :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tcConcreteRecSum
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      868865939143367437##
//      4283065319836759626##
//      main:NonrecSum.$trModule
//      sat_sNz
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//$krep_sNA [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNA
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcConcreteRecSum
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNC :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNC = ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteRecSumNone"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumNone
//  :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'ConcreteRecSumNone
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      8932361323873389123##
//      15462504305832975424##
//      main:NonrecSum.$trModule
//      sat_sNC
//      0#
//      $krep_sNA
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sND [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sND = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNA $krep_sNA
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNE [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNE = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sNb $krep_sND
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNG :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNG = ghc-prim-0.6.1:GHC.Types.TrNameS "'ConcreteRecSumCons"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'ConcreteRecSumCons
//  :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'ConcreteRecSumCons
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      782896096195591732##
//      2603195489806365008##
//      main:NonrecSum.$trModule
//      sat_sNG
//      0#
//      $krep_sNE
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNI :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNI = ghc-prim-0.6.1:GHC.Types.TrNameS "AbstractProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tcAbstractProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tcAbstractProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      12056464016441906154##
//      6100392904543390536##
//      main:NonrecSum.$trModule
//      sat_sNI
//      0#
//      ghc-prim-0.6.1:GHC.Types.krep$*->*->*
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//sat_sNK :: [ghc-prim-0.6.1:GHC.Types.KindRep]
//[LclId]
//sat_sNK
//  = ghc-prim-0.6.1:GHC.Types.:
//      @ ghc-prim-0.6.1:GHC.Types.KindRep
//      $krep_sN7
//      (ghc-prim-0.6.1:GHC.Types.[] @ ghc-prim-0.6.1:GHC.Types.KindRep)
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//sat_sNL :: [ghc-prim-0.6.1:GHC.Types.KindRep]
//[LclId]
//sat_sNL
//  = ghc-prim-0.6.1:GHC.Types.:
//      @ ghc-prim-0.6.1:GHC.Types.KindRep $krep_sN9 sat_sNK
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNJ [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNJ
//  = ghc-prim-0.6.1:GHC.Types.KindRepTyConApp
//      main:NonrecSum.$tcAbstractProd sat_sNL
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNM [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNM = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sN7 $krep_sNJ
//
//-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
//$krep_sNN [InlPrag=NOUSERINLINE[~]]
//  :: ghc-prim-0.6.1:GHC.Types.KindRep
//[LclId, Unf=OtherCon []]
//$krep_sNN = ghc-prim-0.6.1:GHC.Types.KindRepFun $krep_sN9 $krep_sNM
//
//-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
//sat_sNP :: ghc-prim-0.6.1:GHC.Types.TrName
//[LclId]
//sat_sNP = ghc-prim-0.6.1:GHC.Types.TrNameS "'MkAbstractProd"#
//
//-- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
//main:NonrecSum.$tc'MkAbstractProd :: ghc-prim-0.6.1:GHC.Types.TyCon
//[LclIdX, Unf=OtherCon []]
//main:NonrecSum.$tc'MkAbstractProd
//  = ghc-prim-0.6.1:GHC.Types.TyCon
//      16938648863639686556##
//      18023809221364289304##
//      main:NonrecSum.$trModule
//      sat_sNP
//      2#
//      $krep_sNN
//
//-- RHS size: {terms: 3, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.main :: ghc-prim-0.6.1:GHC.Types.IO ()
//[LclIdX]
//main:NonrecSum.main
//  = base-4.14.1.0:GHC.Base.return
//      @ ghc-prim-0.6.1:GHC.Types.IO
//      base-4.14.1.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.6.1:GHC.Tuple.()
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
//  = \ (@ a_akq)
//      (@ b_akr)
//      (eta_B2 [Occ=Once] :: a_akq)
//      (eta_B1 [Occ=Once] :: b_akr) ->
//      main:NonrecSum.MkAbstractProd @ a_akq @ b_akr eta_B2 eta_B1
//
//-- RHS size: {terms: 5, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteRecSumCons
//  :: ghc-prim-0.6.1:GHC.Prim.Int#
//     -> main:NonrecSum.ConcreteRecSum -> main:NonrecSum.ConcreteRecSum
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m1,
// Unf=OtherCon []]
//main:NonrecSum.ConcreteRecSumCons
//  = \ (eta_B2 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#)
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
//  :: ghc-prim-0.6.1:GHC.Prim.Int#
//     -> main:NonrecSum.ConcreteRec -> main:NonrecSum.ConcreteRec
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m,
// Unf=OtherCon []]
//main:NonrecSum.MkConcreteRec
//  = \ (eta_B2 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#)
//      (eta_B1 [Occ=Once] :: main:NonrecSum.ConcreteRec) ->
//      main:NonrecSum.MkConcreteRec eta_B2 eta_B1
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteLeft
//  :: ghc-prim-0.6.1:GHC.Prim.Int# -> main:NonrecSum.ConcreteSum
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m1,
// Unf=OtherCon []]
//main:NonrecSum.ConcreteLeft
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#) ->
//      main:NonrecSum.ConcreteLeft eta_B1
//
//-- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
//main:NonrecSum.ConcreteRight
//  :: ghc-prim-0.6.1:GHC.Prim.Int# -> main:NonrecSum.ConcreteSum
//[GblId[DataCon],
// Arity=1,
// Caf=NoCafRefs,
// Str=<L,U>m2,
// Unf=OtherCon []]
//main:NonrecSum.ConcreteRight
//  = \ (eta_B1 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#) ->
//      main:NonrecSum.ConcreteRight eta_B1
//
//-- RHS size: {terms: 5, types: 2, coercions: 0, joins: 0/0}
//main:NonrecSum.MkConcreteProd
//  :: ghc-prim-0.6.1:GHC.Prim.Int#
//     -> ghc-prim-0.6.1:GHC.Prim.Int# -> main:NonrecSum.ConcreteProd
//[GblId[DataCon],
// Arity=2,
// Caf=NoCafRefs,
// Str=<L,U><L,U>m,
// Unf=OtherCon []]
//main:NonrecSum.MkConcreteProd
//  = \ (eta_B2 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#)
//      (eta_B1 [Occ=Once] :: ghc-prim-0.6.1:GHC.Prim.Int#) ->
//      main:NonrecSum.MkConcreteProd eta_B2 eta_B1
//