# Core-MLIR

## Concerns about this `Graph` version of region

The code that looks like below is considered as a non-dominated use. So it
checks **use-sites**, not **def-sites**.

```cpp
standalone.module { 
standalone.dominance_free_scope {
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-Blocking dominance_free_scope
    vvvv-DEF
    %fib = standalone.toplevel_binding {  
        // ... standalone.dominance_free_scope { standalone.return (%fib) } ...
        ... standalone.return (%fib) } ...
                           USE-^^^^
    }
} // end dominance_free_scope
```

On the other hand, the code below is considered a dominated use (well, the domaintion
that is masked by `standalone.dominance_free_scope`:

```cpp
standalone.module { 
// standalone.dominance_free_scope {

    vvvv-DEF
    %fib = standalone.toplevel_binding {  
        ... standalone.dominance_free_scope { standalone.return (%fib) } ...
   BLOCKING-^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                 USE-^^^^
        //... standalone.return (%fib) } ...
                                    
    }
//} // end dominance_free_scope
```

So, abstractly, the version below round-trips through MLIR. I will denote
this as `DEF(BLOCKING(USE))`:

```
DEF-MUTUAL-RECURSIVE
   BLOCKING----------->
     | USE-MUTUAL-RECURSIVE
     |
     v
```

The one below (denoted as `BLOCKING(DEF(USE)))`) does not
round-trip through MLIR; It gives domination errors:

```
BLOCKING----------->
|DEF-MUTUAL-RECURSIVE
|   USE-MUTUAL-RECURSIVE
|    
v    
```

- My mental model of the "blocking" was off! I thought it meant that
  everything inside this region can disobey SSA conditions. Rather,
  it means that everything **inside** this region can disobey SSA _with respect to_
  everything **outside** this region. [Maybe the other way round as well, I have not tried,
  nor do I have a good mental model of this].


- Unfortunately, for `Core`, it is the latter version with `BLOCKING (DEF (USE))` that 
  is of more use, since the `Core` encoding encodes the recursion as:

```
rec { -- BLOCKING
  fib :: Int -> Int
  
  {- Core Size{terms=23 types=6 cos=0 vbinds=0 jbinds=0} -}
  fib = -- DEF-MUTUAL-RECURSIVE
    λ i →
       ...
      (APP(Main.fib i)) -- USE-MUTUAL-RECURSIVE
      ...
}
```

So when we translate from `Core` into MLIR, we need to either
figure out which are the bindings that are `use-before-def` and then wrap them.
Or we participate in the discussion and petition for this kind of "lazy-region"
as well. Maybe both.

## Stuff discovered in this process about `ghc-dump`:

##### Fib for reference

```hs
{- Core Size{terms=23 types=6 cos=0 vbinds=0 jbinds=0} -}
fib = λ i → case i of wild {
    I# ds →
      case ds of ds {
        DEFAULT →
          APP(GHC.Num.+ @Int GHC.Num.$fNumInt // fib(i-1) + fib(i)
            (APP(Main.fib i)) // fib(i)
            (APP(Main.fib  -- fib(i - 1)
                    APP(GHC.Num.- @Int GHC.Num.$fNumInt i (APP(GHC.Types.I# 1#))))))) -- (i - 1)
        0# → APP(GHC.Types.I# 0#)
        1# → APP(GHC.Types.I# 1#)
      }
  }
```
I feel that `ghc-dump` does not preserve all the information we want. Hence
I started hand-writing the IR we want. I'll finish the translator after
I sleep and wake up. However, it's unclear to me how much extending `ghc-dump`
makes sense. I should maybe clean-slate from a Core plugin.


- Why? Because `ghc-dump` does not retain enough information. For example,
  it treats both `GHC.Num.$fNumInt` and `GHC.Types.I#` as variables; It has
  erased the fact that one is a typeclass dictionary and the other is
  a data constructor.

- Similarly, there is no way to query from within `ghc-dump` what `GHC.Num.-`
  is, and it's impossible to infer from context.

- In general, this is full of unknown-unknowns for me. I don't know enough 
  of the details of core to forsee what we will may need from GHC. Using
  `ghc-dump` is a bad idea because it's technical debt against a 
  _prettyprinter of core_ (fundamentally).

- Hence, we should really be reusing the
  [code in `ghc-dump` that traverses `Core` from within GHC](https://github.com/bgamari/ghc-dump/blob/master/ghc-dump-core/GhcDump/Convert.hs#L237).


# 1 July 2020

- [Reading GHC core sources paid off, the `CorePrep` invariants are documented here](https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/coreSyn/CorePrep.hs#L142)
- In particular we have `case <body> of`. So nested cases are legal, 
  which is something we need to flatten.
- Outside of nested cases, everything else seems "reasonable": laziness is
  at each point of `let`. We can lower `let var = e` as `%let_var = lazy(e)`
  for example.
- Will first transcribe our `fibstrict` example by hand, then write a small
  Core plugin to do this automagically.
- I don't understand WTF `cabal` is doing. In particular, why `cabal install --library`
  installs the library twice :(
- It _seems_ like the cabal documentation on [how to install a system library](https://downloads.haskell.org/~cabal/Cabal-latest/doc/users-guide/installing-packages.html#building-and-installing-a-system-package)
  should do the trick.

```hs
$ runghc Setup.hs configure --ghc
$ runghc Setup.hs build
$ runghc Setup.hs install
```

- OK, so the problem was that I somehow had `cabal` hidden in my package management.
  It turns that even `ghc-pkg` maintains a local and a global package directory,
  and I was exposing stuff in my _local_ package directory (which is in `~/.ghc/.../package.conf.d`),
  note the global one (which is in `/usr/lib/ghc-6.12.1/package.conf.d`).
- The solution is to ask `ghc-pkg --global expose Cabal` which exposes `cabal`,
  which contains `Distribution.Simple`, which is needed to run `Setup.hs`.
- `runghc` is some kind of wrapper around `ghc` runs a file directly without
  having to compile things.
- Of course, this is disjoint from `cabal`'s `exposed-modules`, which is a layer
  disjoint from `ghc-pkg`. I think cabal commands `ghc-pkg` to expose and hide
  what it needs. This is fucking hilarious if it weren't so complex.

- To quote the GHC manual on `Cabal`'s `Distribution.Simple`:

> This module isn't called "Simple" because it's simple. Far from it. It's
> called "Simple" because it does complicated things to simple software.
> The original idea was that there could be different build systems that all
> presented the same compatible command line interfaces. There is still a
> Distribution.Make system but in practice no packages use it.
> https://hackage.haskell.org/package/Cabal-3.2.0.0/docs/Distribution-Simple.html


Reading GHC sources can sometimes be unpleasant. There are many, many invariants
to be maintained. [This is from CorePrep.hs:1450](https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/coreSyn/CorePrep.hs#L1450):

> There is a subtle but important invariant ...
> The solution is CorePrep to have a miniature inlining pass...
> Why does the removal of 'lazy' have to occur in CorePrep? he gory details are in Note [lazyId magic]...
> We decided not to adopt this solution to keep the definition of 'exprIsTrivial' simple....
> There is ONE caveat however...
> the (hacky) non-recursive -- binding for data constructors...

- Brilliant, my tooling suddenly died thanks to https://github.com/well-typed/cborg/issues/242: GHC Prim
  and `cborg` started overlapping an export. 


- [`cabal install --lib` is not idempotent](https://github.com/haskell/cabal/issues/6394).
  Only haskellers would have issue citing a problem about **library installs**,
  while describing the issue as one of **idempotence**.

# 3 July 2020 (Friday)

Got the basic examples converted to SSA. Trying to do this in a GHC plugin.
Most of the translation code works. I'm stuck at a point, though. I need
to rename a variable `GHC.Num.-#` into something that can be named. Otherwise,
I try to create the MLIR:

```
%app_100  =  hask.apSSA(%-#, %i_s1wH)
```

where the `-#` refers to the variable name `GHC.Num.-#`. This is pretty
ludicrous. However, attempting to get a name from GHC seems quite complicated.
There are things like:

- `Id`
- `Var`
- `class NamedThing`
- `data OccName`

it's quite confusing as to what does what.

# 7 July 2020 (Tuesday)

- `mkUniqueGrimily`: great name for a function that creates data.
- OK, good, we now have MLIR that round-trips, in the sense that our
  MLIR gets verified. Now we have undeclared SSA variable problems:

```
tomlir-fibstrict.pass-0000.mlir:12:56: error: use of undeclared SSA value name
                                 %app_0  =  hask.apSSA(%var_minus_hash_99, %var_i_a12E)
                                                       ^
tomlir-fibstrict.pass-0000.mlir:12:76: error: use of undeclared SSA value name
                                 %app_0  =  hask.apSSA(%var_minus_hash_99, %var_i_a12E)
                                                                           ^
tomlir-fibstrict.pass-0000.mlir:25:72: error: use of undeclared SSA value name
                                                 %app_5  =  hask.apSSA(%var_plus_hash_98, %var_wild_X5)
                                                                       ^
tomlir-fibstrict.pass-0000.mlir:49:29: error: use of undeclared SSA value name
      %app_1  =  hask.apSSA(%var_TrNameS_ra, %lit_0)
                            ^
tomlir-fibstrict.pass-0000.mlir:50:29: error: use of undeclared SSA value name
      %app_2  =  hask.apSSA(%var_Module_r7, %app_1)
                            ^
tomlir-fibstrict.pass-0000.mlir:59:29: error: use of undeclared SSA value name
      %app_1  =  hask.apSSA(%var_fib_rwj, %lit_0)
                            ^
tomlir-fibstrict.pass-0000.mlir:65:37: error: use of undeclared SSA value name
              %app_3  =  hask.apSSA(%var_return_02O, %type_2)
                                    ^
tomlir-fibstrict.pass-0000.mlir:66:45: error: use of undeclared SSA value name
              %app_4  =  hask.apSSA(%app_3, %var_$fMonadIO_rob)
                                            ^
tomlir-fibstrict.pass-0000.mlir:69:45: error: use of undeclared SSA value name
              %app_7  =  hask.apSSA(%app_6, %var_unit_tuple_71)
                                            ^
tomlir-fibstrict.pass-0000.mlir:77:29: error: use of undeclared SSA value name
      %app_1  =  hask.apSSA(%var_runMainIO_01E, %type_0)
                            ^
makefile:4: recipe for target 'fibstrict' failed
make: *** [fibstrict] Error 1
```

Note that all of these names are GHC internals. We need to:
- Process all names, figure out what are our 'external' references.
- Code-generate 'extern' stubs for all of these.

There is also going to be the annoying "recursive call does not dominate use"
problem badgering us. We'll have to analyze Core to decide which use site is
recursive. This entire enterprise is messy, messy business.

The GHC sources are confusing. Consider `Util/Bag.hs`. We have `filterBagM` which
seems like an odd operation to have becuse a `Bag` is supposed to be unordered.
Nor does the function have any users at any rate. Spoke to Ben about it,
he said it's fine to delete the function, so I'll send a PR to do that once
I get this up and running...

# Wednesday, 8th july

- change my codegen so that regular variables are not uniqued, only wilds. This
  gives us stable names for things like `fib`, `runMain`, rather than names like `fib_X1` 
  or whatever. That will allow me to hardcode the preamble I need to build a
  vertical proptotype. This is also what Core seems to do:

```hs
Rec {
-- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
fib [Occ=LoopBreaker] :: Int# -> Int#
[LclId]
fib -- the name fib is not uniqued
  = \ (i_a12E :: Int#) ->  -- this lambda variable is uniqued
      case i_a12E of {
        __DEFAULT ->
          case fib (-# i_a12E 1#) of wild_00 { __DEFAULT ->
          (case fib i_a12E of wild_X5 { __DEFAULT -> +# wild_X5 }) wild_00
          };
        0# -> i_a12E;
        1# -> i_a12E
      }
end Rec }

-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
$trModule :: Module
[LclIdX]
$trModule = Module (TrNameS "main"#) (TrNameS "Main"#)

-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
main :: IO ()
[LclIdX]
main
  = case fib 10# of { __DEFAULT -> return @ IO $fMonadIO @ () () }

-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
main :: IO ()
[LclIdX]
main = runMainIO @ () main

```

Note that only parameters to lambdas and wilds are `unique`d. Toplevel names
are not. I need some sane way in the code to figure out what I should unique
and what I should not by reading the Core pretty printing properly.

- There is a bigger problem. Note that the Core appears to have ** two `main` ** 
  declarations. I have no idea WTF is the semantics of this.

- OK, names are now fixed. I call the underlying `Outputable` instance of `Var` that knows the 
  right thing to do in all contexts. I didn't do this earlier because it prints
  functions as `-#`, `+#`, `()`, etc. So I intercept these. The implementation
  is 4 lines, but figuring it out took half an hour :/. This entire enterprise
  is like this.

```hs
-- use the ppr of Var because it knows whether to print or not.
cvtVar :: Var -> SDoc
cvtVar v = 
	let name = unpackFS $ occNameFS $ getOccName v
	in if name == "-#" then  (text "%minus_hash")
  	   else if name == "+#" then (text "%plus_hash")
  	   else if name == "()" then (text "%unit_tuple")
  	   else text "%" >< ppr v 
```


##### Re-checking the dumps from `fibstrict.hs`

OK, so I decided to view the dump from the horse's mouth:

```hs
-- | fibstrict.hs
{-# LANGUAGE MagicHash #-}
import GHC.Prim
fib :: Int# -> Int#
fib i = case i of
        0# ->  i; 1# ->  i
        _ ->  (fib i) +# (fib (i -# 1#))
main :: IO (); main = let x = fib 10# in return ()
```

```Core
-- | generated from fibstrict.hs
==================== Desugar (after optimization) ====================
2020-07-08 16:31:29.998915479 UTC
...

-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
main :: IO ()
[LclIdX]
main
  = case fib 10# of { __DEFAULT ->
    return @ IO GHC.Base.$fMonadIO @ () GHC.Tuple.()
    }

-- | what is this :Main.main?
-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
:Main.main :: IO ()
[LclIdX]
:Main.main = GHC.TopHandler.runMainIO @ () main

```

Note that there is `main`, and then there is `:Main.main` [So there is an extra `:Main.`].
This appears to inform the difference. One of them is some kind of top handler
that is added automagically. I might have to strip this from my printing.
I need to see how to deal with this. Will first identify what adds this symbol
and if there's a clean way to disable this.

- TODO: figure out how to get the core dump that I print in my MLIR file
  to contain as much information as the GHC dump. for example,
  the GHC dump says:

```
-- ***GHC file fibstrict.dump-ds***
-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
:Main.main :: IO ()
[LclIdX]
:Main.main = GHC.TopHandler.runMainIO @ () main

```

```
-- ***my MLIR file with the Core appended to the end as a comment***
-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
main :: IO ()
[LclIdX]
main = runMainIO @ () main

```

- In particular, note that `fibstrict.dump-ds` says `:Main.main = GHC.TopHandler.runMainIO` while my
  MLIR file only says `main = runMainIO ...`. I want that full qualification
  in my dump as well. I will spend some time on this, because the upshot
  is **huge**: accurate debugging and names!