-- | Check if GHC can worker/wrapper repeated case analysis in a recursive
-- call
-- ANSWER: NO!
-- To be more specific, it unwraps the `n` parameter into an `I#`. It does
-- seem to box and rebox the `mx` repeatedly.
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MagicHash #-}
module RepeatedIncrMaybe(incrmN, incrmNRaw#) where
import GHC.Prim
import GHC.Types

data Maybe# = Just# Int# | Nothing#

incrm1Raw# :: Maybe# -> Maybe#
incrm1Raw# mx = case mx of Nothing# -> Nothing# ; Just# x -> Just# (x +# 1#)

incrmNRaw# :: Int# -> Maybe# -> Maybe#
incrmNRaw# n mx = 
  case n of 
    0# -> mx
    _ -> incrmNRaw# (n -# 1#) (incrm1Raw# mx)

-- Analysis: it's first forcing the Maybe to expose all the work,
-- and then calling `RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1`
-- | incrmNRaw#
-- |   = \ (n_auk :: Int#) (mx_aul :: Maybe#) ->
-- |       case n_auk of ds_d1nG {
-- |         __DEFAULT ->
-- |           case mx_aul of {
-- |             Just# x_auj ->
-- |               RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1
-- |                 (+# x_auj 1#) (-# ds_d1nG 1#);
-- |             Nothing# ->
-- |               RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw# (-# ds_d1nG 1#)
-- |           };
-- |         0# -> mx_aul
-- |       }
--
-- The worker that's doing all the work of addition:
--
-- | RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1
-- |   = \ (sc_s1qc :: Int#) (sc1_s1qb :: Int#) ->
-- |       case sc1_s1qb of ds_d1nG {
-- |         __DEFAULT ->
-- |           RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1
-- |             (+# sc_s1qc 1#) (-# ds_d1nG 1#);
-- |         0# -> RepeatedIncrMaybe.Just# sc_s1qc


incrm1 :: Maybe Int -> Maybe Int
incrm1 mx = case mx of Nothing -> Nothing; Just x -> Just (x+1)

incrmN :: Int -> Maybe Int -> Maybe Int
incrmN !n mx = 
  case n of 0 -> mx; _ -> incrmN (n - 1) (incrm1 mx)

-- Raw output:

-- | ==================== Tidy Core ====================
-- | 2020-10-09 09:36:00.424066739 UTC
-- | 
-- | Result size of Tidy Core
-- |   = {terms: 182, types: 77, coercions: 0, joins: 0/0}
-- | 
-- | Rec {
-- | -- RHS size: {terms: 10, types: 2, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw# [Occ=LoopBreaker]
-- |   :: Int# -> Maybe#
-- | [GblId, Arity=1, Caf=NoCafRefs, Str=<S,1*U>, Unf=OtherCon []]
-- | RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#
-- |   = \ (sc_s1qd :: Int#) ->
-- |       case sc_s1qd of ds_d1nG {
-- |         __DEFAULT ->
-- |           RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw# (-# ds_d1nG 1#);
-- |         0# -> RepeatedIncrMaybe.Nothing#
-- |       }
-- | end Rec }
-- | 
-- | Rec {
-- | -- RHS size: {terms: 15, types: 3, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1 [Occ=LoopBreaker]
-- |   :: Int# -> Int# -> Maybe#
-- | [GblId,
-- |  Arity=2,
-- |  Caf=NoCafRefs,
-- |  Str=<L,U><S,1*U>m1,
-- |  Unf=OtherCon []]
-- | RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1
-- |   = \ (sc_s1qc :: Int#) (sc1_s1qb :: Int#) ->
-- |       case sc1_s1qb of ds_d1nG {
-- |         __DEFAULT ->
-- |           RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1
-- |             (+# sc_s1qc 1#) (-# ds_d1nG 1#);
-- |         0# -> RepeatedIncrMaybe.Just# sc_s1qc
-- |       }
-- | end Rec }
-- | 
-- | -- RHS size: {terms: 22, types: 5, coercions: 0, joins: 0/0}
-- | incrmNRaw# :: Int# -> Maybe# -> Maybe#
-- | [GblId,
-- |  Arity=2,
-- |  Caf=NoCafRefs,
-- |  Str=<S,1*U><S,1*U>,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [30 51] 93 0}]
-- | incrmNRaw#
-- |   = \ (n_auk :: Int#) (mx_aul :: Maybe#) ->
-- |       case n_auk of ds_d1nG {
-- |         __DEFAULT ->
-- |           case mx_aul of {
-- |             Just# x_auj ->
-- |               RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1
-- |                 (+# x_auj 1#) (-# ds_d1nG 1#);
-- |             Nothing# ->
-- |               RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw# (-# ds_d1nG 1#)
-- |           };
-- |         0# -> mx_aul
-- |       }
-- | 
-- | -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$trModule4 :: Addr#
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 20 0}]
-- | RepeatedIncrMaybe.$trModule4 = "main"#
-- | 
-- | -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$trModule3 :: TrName
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m1,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- | RepeatedIncrMaybe.$trModule3
-- |   = GHC.Types.TrNameS RepeatedIncrMaybe.$trModule4
-- | 
-- | -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$trModule2 :: Addr#
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 0}]
-- | RepeatedIncrMaybe.$trModule2 = "RepeatedIncrMaybe"#
-- | 
-- | -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$trModule1 :: TrName
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m1,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- | RepeatedIncrMaybe.$trModule1
-- |   = GHC.Types.TrNameS RepeatedIncrMaybe.$trModule2
-- | 
-- | -- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$trModule :: Module
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
-- | RepeatedIncrMaybe.$trModule
-- |   = GHC.Types.Module
-- |       RepeatedIncrMaybe.$trModule3 RepeatedIncrMaybe.$trModule1
-- | 
-- | -- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
-- | $krep_r1qN :: KindRep
-- | [GblId, Caf=NoCafRefs, Str=m1, Unf=OtherCon []]
-- | $krep_r1qN
-- |   = GHC.Types.KindRepTyConApp
-- |       GHC.Types.$tcInt# (GHC.Types.[] @ KindRep)
-- | 
-- | -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tcMaybe#2 :: Addr#
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 30 0}]
-- | RepeatedIncrMaybe.$tcMaybe#2 = "Maybe#"#
-- | 
-- | -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tcMaybe#1 :: TrName
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m1,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- | RepeatedIncrMaybe.$tcMaybe#1
-- |   = GHC.Types.TrNameS RepeatedIncrMaybe.$tcMaybe#2
-- | 
-- | -- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tcMaybe# :: TyCon
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 70}]
-- | RepeatedIncrMaybe.$tcMaybe#
-- |   = GHC.Types.TyCon
-- |       6631270215449761222##
-- |       15794099770922122318##
-- |       RepeatedIncrMaybe.$trModule
-- |       RepeatedIncrMaybe.$tcMaybe#1
-- |       0#
-- |       GHC.Types.krep$*
-- | 
-- | -- RHS size: {terms: 3, types: 1, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Nothing#1 [InlPrag=NOUSERINLINE[~]]
-- |   :: KindRep
-- | [GblId, Caf=NoCafRefs, Str=m1, Unf=OtherCon []]
-- | RepeatedIncrMaybe.$tc'Nothing#1
-- |   = GHC.Types.KindRepTyConApp
-- |       RepeatedIncrMaybe.$tcMaybe# (GHC.Types.[] @ KindRep)
-- | 
-- | -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Nothing#3 :: Addr#
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 40 0}]
-- | RepeatedIncrMaybe.$tc'Nothing#3 = "'Nothing#"#
-- | 
-- | -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Nothing#2 :: TrName
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m1,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- | RepeatedIncrMaybe.$tc'Nothing#2
-- |   = GHC.Types.TrNameS RepeatedIncrMaybe.$tc'Nothing#3
-- | 
-- | -- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Nothing# :: TyCon
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 70}]
-- | RepeatedIncrMaybe.$tc'Nothing#
-- |   = GHC.Types.TyCon
-- |       9741989951278664473##
-- |       5458438331126061159##
-- |       RepeatedIncrMaybe.$trModule
-- |       RepeatedIncrMaybe.$tc'Nothing#2
-- |       0#
-- |       RepeatedIncrMaybe.$tc'Nothing#1
-- | 
-- | -- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Just#1 [InlPrag=NOUSERINLINE[~]] :: KindRep
-- | [GblId, Caf=NoCafRefs, Str=m4, Unf=OtherCon []]
-- | RepeatedIncrMaybe.$tc'Just#1
-- |   = GHC.Types.KindRepFun $krep_r1qN RepeatedIncrMaybe.$tc'Nothing#1
-- | 
-- | -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Just#3 :: Addr#
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 30 0}]
-- | RepeatedIncrMaybe.$tc'Just#3 = "'Just#"#
-- | 
-- | -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Just#2 :: TrName
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m1,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- | RepeatedIncrMaybe.$tc'Just#2
-- |   = GHC.Types.TrNameS RepeatedIncrMaybe.$tc'Just#3
-- | 
-- | -- RHS size: {terms: 7, types: 0, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$tc'Just# :: TyCon
-- | [GblId,
-- |  Caf=NoCafRefs,
-- |  Str=m,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 70}]
-- | RepeatedIncrMaybe.$tc'Just#
-- |   = GHC.Types.TyCon
-- |       1825616634654905576##
-- |       9303633944142475280##
-- |       RepeatedIncrMaybe.$trModule
-- |       RepeatedIncrMaybe.$tc'Just#2
-- |       0#
-- |       RepeatedIncrMaybe.$tc'Just#1
-- | 
-- | Rec {
-- | -- RHS size: {terms: 19, types: 6, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.incrmN_$s$wincrmN [Occ=LoopBreaker]
-- |   :: Int -> Int# -> Maybe Int
-- | [GblId,
-- |  Arity=2,
-- |  Caf=NoCafRefs,
-- |  Str=<L,U(U)><S,1*U>m2,
-- |  Unf=OtherCon []]
-- | RepeatedIncrMaybe.incrmN_$s$wincrmN
-- |   = \ (sc_s1q5 :: Int) (sc1_s1q4 :: Int#) ->
-- |       case sc1_s1q4 of ds_X1nH {
-- |         __DEFAULT ->
-- |           RepeatedIncrMaybe.incrmN_$s$wincrmN
-- |             (case sc_s1q5 of { I# x_a1oi -> GHC.Types.I# (+# x_a1oi 1#) })
-- |             (-# ds_X1nH 1#);
-- |         0# -> GHC.Maybe.Just @ Int sc_s1q5
-- |       }
-- | end Rec }
-- | 
-- | Rec {
-- | -- RHS size: {terms: 10, types: 3, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.incrmN_$s$wincrmN1 [Occ=LoopBreaker]
-- |   :: Int# -> Maybe Int
-- | [GblId, Arity=1, Caf=NoCafRefs, Str=<S,1*U>, Unf=OtherCon []]
-- | RepeatedIncrMaybe.incrmN_$s$wincrmN1
-- |   = \ (sc_s1q3 :: Int#) ->
-- |       case sc_s1q3 of ds_X1nH {
-- |         __DEFAULT -> RepeatedIncrMaybe.incrmN_$s$wincrmN1 (-# ds_X1nH 1#);
-- |         0# -> GHC.Maybe.Nothing @ Int
-- |       }
-- | end Rec }
-- | 
-- | -- RHS size: {terms: 26, types: 9, coercions: 0, joins: 0/0}
-- | RepeatedIncrMaybe.$wincrmN [InlPrag=NOUSERINLINE[2]]
-- |   :: Int# -> Maybe Int -> Maybe Int
-- | [GblId,
-- |  Arity=2,
-- |  Caf=NoCafRefs,
-- |  Str=<S,1*U><S,1*U>,
-- |  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True, Guidance=IF_ARGS [30 51] 113 0}]
-- | RepeatedIncrMaybe.$wincrmN
-- |   = \ (ww_s1ph :: Int#) (w_s1pe :: Maybe Int) ->
-- |       case ww_s1ph of ds_X1nH {
-- |         __DEFAULT ->
-- |           case w_s1pe of {
-- |             Nothing -> RepeatedIncrMaybe.incrmN_$s$wincrmN1 (-# ds_X1nH 1#);
-- |             Just x_aun ->
-- |               RepeatedIncrMaybe.incrmN_$s$wincrmN
-- |                 (case x_aun of { I# x1_a1oi -> GHC.Types.I# (+# x1_a1oi 1#) })
-- |                 (-# ds_X1nH 1#)
-- |           };
-- |         0# -> w_s1pe
-- |       }
-- | 
-- | -- RHS size: {terms: 8, types: 5, coercions: 0, joins: 0/0}
-- | incrmN [InlPrag=NOUSERINLINE[2]] :: Int -> Maybe Int -> Maybe Int
-- | [GblId,
-- |  Arity=2,
-- |  Caf=NoCafRefs,
-- |  Str=<S(S),1*U(1*U)><S,1*U>,
-- |  Unf=Unf{Src=InlineStable, TopLvl=True, Value=True, ConLike=True,
-- |          WorkFree=True, Expandable=True,
-- |          Guidance=ALWAYS_IF(arity=2,unsat_ok=True,boring_ok=False)
-- |          Tmpl= \ (w_s1pd [Occ=Once!] :: Int)
-- |                  (w1_s1pe [Occ=Once] :: Maybe Int) ->
-- |                  case w_s1pd of { I# ww1_s1ph [Occ=Once] ->
-- |                  RepeatedIncrMaybe.$wincrmN ww1_s1ph w1_s1pe
-- |                  }}]
-- | incrmN
-- |   = \ (w_s1pd :: Int) (w1_s1pe :: Maybe Int) ->
-- |       case w_s1pd of { I# ww1_s1ph ->
-- |       RepeatedIncrMaybe.$wincrmN ww1_s1ph w1_s1pe
-- |       }
-- | 
-- | 
-- | ------ Local rules for imported ids --------
-- | "SC:incrmNRaw#0"
-- |     forall (sc_s1qc :: Int#) (sc1_s1qb :: Int#).
-- |       incrmNRaw# sc1_s1qb (RepeatedIncrMaybe.Just# sc_s1qc)
-- |       = RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw#1 sc_s1qc sc1_s1qb
-- | "SC:incrmNRaw#1"
-- |     forall (sc_s1qd :: Int#).
-- |       incrmNRaw# sc_s1qd RepeatedIncrMaybe.Nothing#
-- |       = RepeatedIncrMaybe.incrmNRaw#_$sincrmNRaw# sc_s1qd
-- | "SC:$wincrmN0" [2]
-- |     forall (sc_s1q3 :: Int#).
-- |       RepeatedIncrMaybe.$wincrmN sc_s1q3 (GHC.Maybe.Nothing @ Int)
-- |       = RepeatedIncrMaybe.incrmN_$s$wincrmN1 sc_s1q3
-- | "SC:$wincrmN1" [2]
-- |     forall (sc_s1q5 :: Int) (sc1_s1q4 :: Int#).
-- |       RepeatedIncrMaybe.$wincrmN sc1_s1q4 (GHC.Maybe.Just @ Int sc_s1q5)
-- |       = RepeatedIncrMaybe.incrmN_$s$wincrmN sc_s1q5 sc1_s1q4
-- | 
