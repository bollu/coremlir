-- | Check if GHC can worker/wrapper repeated case analysis.
-- ANSWER: Yes!
{-# LANGUAGE BangPatterns #-}
module RepeatedIncrMaybe(incrm3) where

incrm1 :: Maybe Int -> Maybe Int
incrm1 mx = case mx of Nothing -> Nothing; Just x -> Just (x+1)

incrm3 :: Maybe Int -> Maybe Int
incrm3 mx = incrm1 (incrm1(incrm1(mx)))

-- ==================== Tidy Core ====================
-- 2020-10-02 08:35:06.140905279 UTC
-- 
-- Result size of Tidy Core
--   = {terms: 29, types: 18, coercions: 0, joins: 0/0}
-- 
-- -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- RepeatedIncrMaybe.$trModule4 :: GHC.Prim.Addr#
-- [GblId,
--  Caf=NoCafRefs,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 20 0}]
-- RepeatedIncrMaybe.$trModule4 = "main"#
-- 
-- -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- RepeatedIncrMaybe.$trModule3 :: GHC.Types.TrName
-- [GblId,
--  Caf=NoCafRefs,
--  Str=m1,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- RepeatedIncrMaybe.$trModule3
--   = GHC.Types.TrNameS RepeatedIncrMaybe.$trModule4
-- 
-- -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- RepeatedIncrMaybe.$trModule2 :: GHC.Prim.Addr#
-- [GblId,
--  Caf=NoCafRefs,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 0}]
-- RepeatedIncrMaybe.$trModule2 = "RepeatedIncrMaybe"#
-- 
-- -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- RepeatedIncrMaybe.$trModule1 :: GHC.Types.TrName
-- [GblId,
--  Caf=NoCafRefs,
--  Str=m1,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- RepeatedIncrMaybe.$trModule1
--   = GHC.Types.TrNameS RepeatedIncrMaybe.$trModule2
-- 
-- -- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
-- RepeatedIncrMaybe.$trModule :: GHC.Types.Module
-- [GblId,
--  Caf=NoCafRefs,
--  Str=m,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
-- RepeatedIncrMaybe.$trModule
--   = GHC.Types.Module
--       RepeatedIncrMaybe.$trModule3 RepeatedIncrMaybe.$trModule1
-- 
-- -- RHS size: {terms: 14, types: 9, coercions: 0, joins: 0/0}
-- incrm3 :: Maybe Int -> Maybe Int
-- [GblId,
--  Arity=1,
--  Caf=NoCafRefs,
--  Str=<S,1*U>,
--  Unf=Unf{Src=InlineStable, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True,
--          Guidance=ALWAYS_IF(arity=1,unsat_ok=True,boring_ok=False)
--          Tmpl= \ (mx_avd [Occ=Once!] :: Maybe Int) ->
--                  case mx_avd of {
--                    Nothing -> GHC.Maybe.Nothing @ Int;
--                    Just x_au0 [Occ=Once!] ->
--                      GHC.Maybe.Just
--                        @ Int
--                        (case x_au0 of { GHC.Types.I# x1_aB9 [Occ=Once] ->
--                         GHC.Types.I# (GHC.Prim.+# 3# x1_aB9)
--                         })
--                  }}]
-- incrm3
--   = \ (mx_avd :: Maybe Int) ->
--       case mx_avd of {
--         Nothing -> GHC.Maybe.Nothing @ Int;
--         Just x_au0 ->
--           GHC.Maybe.Just
--             @ Int
--             (case x_au0 of { GHC.Types.I# x1_aB9 ->
--              GHC.Types.I# (GHC.Prim.+# 3# x1_aB9)
--              })
--       }
-- 
-- 
--
-- 
-- ==================== Worker Wrapper binds ====================
-- 2020-10-02 08:35:06.138863785 UTC
-- 
-- Result size of Worker Wrapper binds
--   = {terms: 29, types: 18, coercions: 0, joins: 0/0}
-- 
-- -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- $trModule_sB1 :: GHC.Prim.Addr#
-- [LclId,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 20 0}]
-- $trModule_sB1 = "main"#
-- 
-- -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- $trModule_sB2 :: GHC.Types.TrName
-- [LclId,
--  Str=m1,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- $trModule_sB2 = GHC.Types.TrNameS $trModule_sB1
-- 
-- -- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
-- $trModule_sB3 :: GHC.Prim.Addr#
-- [LclId,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 60 0}]
-- $trModule_sB3 = "RepeatedIncrMaybe"#
-- 
-- -- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
-- $trModule_sB4 :: GHC.Types.TrName
-- [LclId,
--  Str=m1,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
-- $trModule_sB4 = GHC.Types.TrNameS $trModule_sB3
-- 
-- -- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
-- RepeatedIncrMaybe.$trModule :: GHC.Types.Module
-- [LclIdX,
--  Str=m,
--  Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
-- RepeatedIncrMaybe.$trModule
--   = GHC.Types.Module $trModule_sB2 $trModule_sB4
-- 
-- -- RHS size: {terms: 14, types: 9, coercions: 0, joins: 0/0}
-- incrm3 :: Maybe Int -> Maybe Int
-- [LclIdX,
--  Arity=1,
--  Str=<S,1*U>,
--  Unf=Unf{Src=InlineStable, TopLvl=True, Value=True, ConLike=True,
--          WorkFree=True, Expandable=True,
--          Guidance=ALWAYS_IF(arity=1,unsat_ok=True,boring_ok=False)
--          Tmpl= \ (mx_avd [Occ=Once!] :: Maybe Int) ->
--                  case mx_avd of {
--                    Nothing -> GHC.Maybe.Nothing @ Int;
--                    Just x_au0 [Occ=Once!] ->
--                      GHC.Maybe.Just
--                        @ Int
--                        (case x_au0 of { GHC.Types.I# x_aB9 [Occ=Once] ->
--                         GHC.Types.I# (GHC.Prim.+# 3# x_aB9)
--                         })
--                  }}]
-- incrm3
--   = \ (mx_avd [Dmd=<S,U>] :: Maybe Int) ->
--       case mx_avd of {
--         Nothing -> GHC.Maybe.Nothing @ Int;
--         Just x_au0 [Dmd=<L,U(U)>] ->
--           GHC.Maybe.Just
--             @ Int
--             (case x_au0 of { GHC.Types.I# x_aB9 ->
--              GHC.Types.I# (GHC.Prim.+# 3# x_aB9)
--              })
--       }
-- 
-- 
