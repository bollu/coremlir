```hs
-- The Glorious Glasgow Haskell Compilation System, version 8.10.2
-- compiled with -O3
module Foo(incr2, main) where

incr :: [Int] -> [Int]
incr xs = case xs of [] -> []; (y:ys) -> (y+1):incr ys

incr2 :: [Int] -> [Int]
incr2 xs = incr(incr xs)

main :: IO ()
main = do
 xs <- read <$> getLine
 let ys = incr2 xs
 print ys
```

- WTF, all I see is a `(incr (incr y_a1CP))`? Why haven't the loops fused?!
  I would have expected to see the loops fused?!


```
==================== Tidy Core ====================
2020-10-02 06:07:48.637195359 UTC

Result size of Tidy Core
  = {terms: 66, types: 68, coercions: 10, joins: 0/0}

-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
Foo.$trModule4 :: GHC.Prim.Addr#
[GblId,
 Caf=NoCafRefs,
 Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 20 0}]
Foo.$trModule4 = "main"#

-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
Foo.$trModule3 :: GHC.Types.TrName
[GblId,
 Caf=NoCafRefs,
 Str=m1,
 Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
Foo.$trModule3 = GHC.Types.TrNameS Foo.$trModule4

-- RHS size: {terms: 1, types: 0, coercions: 0, joins: 0/0}
Foo.$trModule2 :: GHC.Prim.Addr#
[GblId,
 Caf=NoCafRefs,
 Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 20 0}]
Foo.$trModule2 = "Foo"#

-- RHS size: {terms: 2, types: 0, coercions: 0, joins: 0/0}
Foo.$trModule1 :: GHC.Types.TrName
[GblId,
 Caf=NoCafRefs,
 Str=m1,
 Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 20}]
Foo.$trModule1 = GHC.Types.TrNameS Foo.$trModule2

-- RHS size: {terms: 3, types: 0, coercions: 0, joins: 0/0}
Foo.$trModule :: GHC.Types.Module
[GblId,
 Caf=NoCafRefs,
 Str=m,
 Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True, Guidance=IF_ARGS [] 10 30}]
Foo.$trModule = GHC.Types.Module Foo.$trModule3 Foo.$trModule1

Rec {
-- RHS size: {terms: 16, types: 11, coercions: 0, joins: 0/0}
incr [Occ=LoopBreaker] :: [Int] -> [Int]
[GblId, Arity=1, Caf=NoCafRefs, Str=<S,1*U>, Unf=OtherCon []]
incr
  = \ (xs_atx :: [Int]) ->
      case xs_atx of {
        [] -> GHC.Types.[] @ Int;
        : y_aty ys_atz ->
          GHC.Types.:
            @ Int
            (case y_aty of { GHC.Types.I# x_a1Gf ->
             GHC.Types.I# (GHC.Prim.+# x_a1Gf 1#)
             })
            (incr ys_atz)
      }
end Rec }

-- RHS size: {terms: 4, types: 2, coercions: 0, joins: 0/0}
incr2 :: [Int] -> [Int]
[GblId,
 Arity=1,
 Caf=NoCafRefs,
 Str=<S,1*U>,
 Unf=Unf{Src=InlineStable, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True,
         Guidance=ALWAYS_IF(arity=1,unsat_ok=True,boring_ok=False)
         Tmpl= \ (xs_auM [Occ=Once] :: [Int]) -> incr (incr xs_auM)}]
incr2 = \ (xs_auM :: [Int]) -> incr (incr xs_auM)

-- RHS size: {terms: 27, types: 30, coercions: 7, joins: 0/0}
Foo.main1
  :: GHC.Prim.State# GHC.Prim.RealWorld
     -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #)
[GblId,
 Arity=1,
 Str=<L,U>,
 Unf=Unf{Src=<vanilla>, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True, Guidance=IF_ARGS [0] 240 0}]
Foo.main1
  = \ (s_a1Gz :: GHC.Prim.State# GHC.Prim.RealWorld) ->
      case GHC.IO.Handle.Internals.wantReadableHandle_1
             @ String
             GHC.IO.Handle.Text.hGetLine4
             GHC.IO.Handle.FD.stdin
             (GHC.IO.Handle.Text.hGetLine2
              `cast` (<GHC.IO.Handle.Types.Handle__>_R
                      ->_R Sym (GHC.Types.N:IO[0] <String>_R)
                      :: (GHC.IO.Handle.Types.Handle__
                          -> GHC.Prim.State# GHC.Prim.RealWorld
                          -> (# GHC.Prim.State# GHC.Prim.RealWorld, String #))
                         ~R# (GHC.IO.Handle.Types.Handle__ -> IO String)))
             s_a1Gz
      of
      { (# ipv_a1GU, ipv1_a1GV #) ->
      ((GHC.IO.Handle.Text.hPutStr'
          GHC.IO.Handle.FD.stdout
          (case Text.Read.$wreadEither
                  @ [Int] GHC.Read.$fReadInt_$creadListPrec ipv1_a1GV
           of {
             Left x_a1CN ->
               case errorWithoutStackTrace @ 'GHC.Types.LiftedRep @ [Int] x_a1CN
               of wild1_00 {
               };
             Right y_a1CP ->
               GHC.Show.showList__
                 @ Int
                 GHC.Show.$fShowInt1
                 (incr (incr y_a1CP))
                 (GHC.Types.[] @ Char)
           })
          GHC.Types.True)
       `cast` (GHC.Types.N:IO[0] <()>_R
               :: IO ()
                  ~R# (GHC.Prim.State# GHC.Prim.RealWorld
                       -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #))))
        ipv_a1GU
      }

-- RHS size: {terms: 1, types: 0, coercions: 3, joins: 0/0}
main :: IO ()
[GblId,
 Arity=1,
 Str=<L,U>,
 Unf=Unf{Src=InlineStable, TopLvl=True, Value=True, ConLike=True,
         WorkFree=True, Expandable=True,
         Guidance=ALWAYS_IF(arity=0,unsat_ok=True,boring_ok=True)
         Tmpl= Foo.main1
               `cast` (Sym (GHC.Types.N:IO[0] <()>_R)
                       :: (GHC.Prim.State# GHC.Prim.RealWorld
                           -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #))
                          ~R# IO ())}]
main
  = Foo.main1
    `cast` (Sym (GHC.Types.N:IO[0] <()>_R)
            :: (GHC.Prim.State# GHC.Prim.RealWorld
                -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #))
               ~R# IO ())
```
