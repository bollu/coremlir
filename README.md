# Core-MLIR

Convert GHC Core to MLIR.


# Log:  [newest] to [oldest]

# Monday, 24 August 2020
- Nuked `HaskModuleOp`, `HaskDummyFinishOp` since I'm just using the regular `ModuleOp` now. I now understand
  why `ModuleOp` doesn't allow SSA variables in its body: these are not accessible from functions because of the
  `IsolatedFromAbove` constraint. So it only makes sense to have "true global data" in a `ModuleOp`. I really wish
  I didn't have to "learn their design choices" by reinventing the bloody wheel. Oh well, it was at least very
  instructive.
  
- Got full lowering down into LLVM. I now need to lower a program with `Int`, not just `Int#`.

- [Wow the names of data constructors are complicated](https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/basicTypes/DataCon.hs#L126)

> Note [Data Constructor Naming]
> ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
> Each data constructor C has two, and possibly up to four, Names associated with it:
- My god, GHC does love inflicting pain on those who decide to read its sources.

- I'm writing the simplest possible version of `fib` that compiles through the GHC toolchain:

```hs
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
import GHC.Prim
import GHC.Types(IO(..))
data SimpleInt = MkSimpleInt Int#

plus :: SimpleInt -> SimpleInt -> SimpleInt
plus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival +# jval)


minus :: SimpleInt -> SimpleInt -> SimpleInt
minus i j = case i of MkSimpleInt ival -> case j of MkSimpleInt jval -> MkSimpleInt (ival -# jval)


one :: SimpleInt; one = MkSimpleInt 1#
zero :: SimpleInt; zero = MkSimpleInt 0#

fib :: SimpleInt -> SimpleInt
fib i =
    case i of
       MkSimpleInt 0# -> zero
       MkSimpleInt 1# -> one
       n -> plus (fib n) (fib (minus n one))
main = IO (\s -> (# s, ()#))
```

```
/tmp/ghc1433_0/ghc_2.s:194:0: error:
     Error: symbol `Main_MkSimpleInt_info' is already defined
    |
194 | Main_MkSimpleInt_info:
    | ^

/tmp/ghc1433_0/ghc_2.s:214:0: error:
     Error: symbol `Main_MkSimpleInt_closure' is already defined
    |
214 | Main_MkSimpleInt_closure:
    | ^
```
- OK, interesting, my GHC plugin is somehow causing `Int` to be defined twice.
 
- I gave up. It seems to be because I run `CorePrep` myself manually, after which GHC
  also decides to run `CorePrep`. So I came up with the brilliant solution of killing `GHC`
  in a plugin pass after all of my scheduled passes run. This is so fucked up.

- I need to change `apSSA` to be capable of accepting the second parameter as a symbol
  as well.
```
tomlir-fib.pass-0000.mlir:82:39: error: expected SSA operand
        %app_24 = hask.apSSA(%app_23, @one)
```
# Log:  [oldest] to [newest]

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

- The GHC codebase is written with misery as a resource, it seems:

```hs
-- compiler/GHC/Rename/Env.hs
        -- We can get built-in syntax showing up here too, sadly.  If you type
        --      data T = (,,,)
        -- the constructor is parsed as a type, and then GHC.Parser.PostProcess.tyConToDataCon
        -- uses setRdrNameSpace to make it into a data constructors.  At that point
        -- the nice Exact name for the TyCon gets swizzled to an Orig name.
        -- Hence the badOrigBinding error message.
        --
        -- Except for the ":Main.main = ..." definition inserted into
        -- the Main module; ugh!
```

Ugh indeed. I have no idea how to check if the binder is `:Main.main`

- What I do know is that this is built here:

```
compiler/GHC/Tc/Module.hs
-- See Note [Root-main Id]
-- Construct the binding
--      :Main.main :: IO res_ty = runMainIO res_ty main
; run_main_id <- tcLookupId runMainIOName
; let { root_main_name =  mkExternalName rootMainKey rOOT_MAIN
                   (mkVarOccFS (fsLit "main"))
                   (getSrcSpan main_name)
; root_main_id = Id.mkExportedVanillaId root_main_name
                                      (mkTyConApp ioTyCon [res_ty])
```

After which I have no _fucking_ clue how to check that the binding
comes from this module.

The note reads:

```
Note [Root-main Id]
~~~~~~~~~~~~~~~~~~~
The function that the RTS invokes is always :Main.main, which we call
root_main_id.  (Because GHC allows the user to have a module not
called Main as the main module, we can't rely on the main function
being called "Main.main".  That's why root_main_id has a fixed module
":Main".)

This is unusual: it's a LocalId whose Name has a Module from another
module. Tiresomely, we must filter it out again in GHC.Iface.Make, less we
get two defns for 'main' in the interface file!
```

# Monday, 13th July 2020

- Added a new type `hask.untyped` to represent all things in my hask dialect.
  This was mostly to future proof and ensure that stuff is not
  accidentally wrecked by my use of `none`.

## how is `FuncOp` implemented?

How the funcOp gets parsed:

- Toplevel: It calls `parseFunctionLikeOp`. They use `PIMPL` style here for whatever
  reason.

- https://github.com/llvm/llvm-project/blob/74145d584126da2ce7a836d9b2240d56442f3ea1/mlir/lib/IR/Function.cpp

```cpp
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false,
                                   buildFuncType);
}
```

- the call to `parseFunctioLikeOp` does bog-standard stuff. The interesting
  bit is that it parses the function name as a _symbol_ (attribute). so the
  syntax `func foo` has `func` as a keyword, with `foo` being a symbol.

- Now I'm confused as to how this prevents "double declarations" of the same
  function. is this verified by the module after as a separate check, and
  not encoded as SSA? If so, that's fugly.

- https://github.com/llvm/llvm-project/blob/5eae715a3115be2640d0fd37d0bd4771abf2ab9b/mlir/lib/IR/FunctionImplementation.cpp#L160
```cpp
ParseResult
mlir::impl::parseFunctionLikeOp(OpAsmParser &parser, OperationState &result,
                                bool allowVariadic,
                                mlir::impl::FuncTypeBuilder funcTypeBuilder) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  auto signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (parseFunctionSignature(parser, allowVariadic, entryArgs, argTypes,
                             argAttrs, isVariadic, resultTypes, resultAttrs))
    return failure();

  std::string errorMessage;
  if (auto type = funcTypeBuilder(builder, argTypes, resultTypes,
                                  impl::VariadicFlag(isVariadic), errorMessage))
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));    
  else
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  return parser.parseOptionalRegion(
      *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes);
}
```

##### How `call` works:

- FML, tobias was right. I was hoping he was not. It is indeed true that the
  function name argument is a string :/ So then, how does one walk the
  use chain when one hits a function?
- https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/Dialect/StandardOps/IR/Ops.td#L632

```cpp
def CallOp : Std_Op<"call", [CallOpInterface]> {
  ...
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, FuncOp callee,"
    "ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder.getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  }]>, OpBuilder<
    "OpBuilder &builder, OperationState &result, SymbolRefAttr callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addTypes(results);
  }]>, OpBuilder<
    "OpBuilder &builder, OperationState &result, StringRef callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      build(builder, result, builder.getSymbolRefAttr(callee), results,
            operands);
  }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}
```

- What is a `FlatSymbolRefAttr` you ask? excellent question.
- https://github.com/llvm/llvm-project/blob/9db53a182705ac1f652c6ee375735bea5539272c/mlir/include/mlir/IR/Attributes.h#L551
- OK, so it's not a string! It's a `symbolName`, as parsed by `parseSymbolName`.


```cpp
 /// A symbol reference attribute represents a symbolic reference to another
 /// operation.
 class SymbolRefAttr
    : public Attribute::AttrBase<SymbolRefAttr, Attribute,
                                 detail::SymbolRefAttributeStorage> {
```

- symbols are explained in MLIR as follows, at the 'Symbols and symbol tables' doc: https://github.com/llvm/llvm-project/blob/9db53a182705ac1f652c6ee375735bea5539272c/mlir/docs/SymbolsAndSymbolTables.md

> A Symbol is a named operation that resides immediately within a region that
> defines a SymbolTable. The name of a symbol must be unique within the parent
> SymbolTable. This name is semantically similarly to an SSA result value, and
> may be referred to by other operations to provide a symbolic link, or use, to
> the symbol. An example of a Symbol operation is func. func defines a symbol
> name, which is referred to by operations like `std.call`.

- It continues, talking explicitly about SSA:

> Using an attribute, as opposed to an SSA value, has several benefits:
>
> If we were to use SSA values, we would need to create some mechanism in which
> to opt-out of certain properties of it such as dominance. Attributes allow
> for referencing the operations irregardless of the order in which they were
> defined.
>
> Attributes simplify referencing operations within nested symbol tables, which
> are traditionally not visible outside of the parent region.

- OK, nice, this is not fugly! Great `:D` I am so releived.

##### How `ret` works:

- https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/Dialect/StandardOps/IR/Ops.td#L2063

```cpp
 def ReturnOp : Std_Op<"return", [NoSideEffect, HasParent<"FuncOp">, ReturnLike,
                                 Terminator]> {
  ...

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
```

### Can we use the `recursive_ref` construct to encode `fib` more simply?

Yes we can. We can write, for example:

```mlir
hask.module { 
    %fib = hask.recursive_ref  {  
        %core_one =  hask.make_i32(1)
        %fib_call = hask.apSSA(%fib, %core_one) <- use does not dominate def
        hask.return(%fib_call)
    }
    hask.return(%fib)
}
```

and this "just works".

**EDIT**: Nope, NVM. I implemented this and found out that this _does not work_:

```mlir
hask.module { 
    %core_one =  hask.make_i32(1)

    // passes
    %flat = hask.recursive_ref  {  
        %fib_call = hask.apSSA(%flat, %core_one)
        hask.return(%fib_call)
    }

    // fails!
    %nested = hask.recursive_ref  {  
        %case = hask.caseSSA %core_one 
                ["default" -> { //default
                    // fails because the use is nested inside a region.
                    %fib_call = hask.apSSA(%nested, %core_one)
                    hask.return(%fib_call)
                }]
        hask.return(%case)
    }
    hask.dummy_finish
}
```

- In particular, note that the `%nested` use fails. This is because the use
  is wrapped inside a normal region of the `default` block. This normal
  region again establishes SSA rules.


### Email to GHC-devs about how to use names


I'm trying to understand how to query information about `Var`s from a
Core plugin. Consider the snippet of haskell:

```
{-# LANGUAGE MagicHash #-}
import GHC.Prim
fib :: Int# -> Int#
fib i = case i of 0# ->  i; 1# ->  i; _ ->  (fib i) +# (fib (i -# 1#))

main :: IO (); main = let x = fib 10# in return ()
```

That compiles to the following (elided) GHC Core, dumped right after desugar:

```mlir
Rec {
fib [Occ=LoopBreaker] :: Int# -> Int#
[LclId]
fib
  = \ (i_a12E :: Int#) ->
      case i_a12E of {
        __DEFAULT ->
          case fib (-# i_a12E 1#) of wild_00 { __DEFAULT ->
          (case fib i_a12E of wild_X5 { __DEFAULT -> +# wild_X5 }) wild_00
          };
        0# -> i_a12E;
        1# -> i_a12E
      }
end Rec }

Main.$trModule :: GHC.Types.Module
[LclIdX]
Main.$trModule
  = GHC.Types.Module
      (GHC.Types.TrNameS "main"#) (GHC.Types.TrNameS "Main"#)

-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
main :: IO ()
[LclIdX]
main
  = case fib 10# of { __DEFAULT ->
    return @ IO GHC.Base.$fMonadIO @ () GHC.Tuple.()
    }

-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
:Main.main :: IO ()
[LclIdX]
:Main.main = GHC.TopHandler.runMainIO @ () main
```

I've been  using `occNameString . getOccName :: Id -> String` to detect names from a `Var`
. I'm rapidly finding this insufficient, and want more information
about a variable. In particular, How to I figure out:

1. When I see the Var with occurence name `fib`, that it belongs to module `Main`?
2. When I see the Var with name `main`, whether it is `Main.main` or `:Main.main`?
3. When I see the Var with name `+#`, that this is an inbuilt name? Similarly
   for `-#` and `()`.
4. In general, given a Var, how do I decide where it comes from, and whether it is
   user-defined or something GHC defined ('wired-in' I believe is the term I am
   looking for)?
5. When I see a `Var`, how do I learn its type?
6. In general, is there a page that tells me how to 'query' Core/`ModGuts` from within a core plugin?


### Answers to email [Where to find name info]:

- I received one answer (so far) that told me to look at https://hackage.haskell.org/package/ghc-8.10.1/docs/Name.html#g:3
- In `haskell-code-explorer`: https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/basicTypes/Name.hs#L107

- The link seems to contain answers to some of my questions, but not others. I
  had tried some of the APIs among them, and didn't understand their semantics.
  But it's at least comforting to know that I was looking at the right place.

- Another  file that might be useful: https://hackage.haskell.org/package/ghc-8.10.1/docs/Module.html#t:Module
- In `haskell-code-exporer`: https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/basicTypes/Module.hs#L407

### Trying to use `SymbolAttr` for implementing functions

- The problem appears to be that something like `@foo` is not considered an SSA value, but a `SymbolAttr`
  So what is the "type" of my function `apSSA`? does it take first parameter an SSA value?
  or does it take first parameter `SymbolAttr`? I need both!

```mlir
// foo :: Int -> (Int -> Int)
func @foo() {
  ... 
  %ret = hask.ap(@foo, 1) // recursive call
  hask.ap(%ret, 1) // 
}
```

Possible solutions:
1. Make the call of two types: `callToplevel`, and `callOther`. This is against the ethos of haskell.
2. Continue using our `reucrsive_ref` hack that lets us treat toplevel bindings uniformly.
3. Use MLIR hackery to have `call(...)` take first parameter _either_ `FlatSymbolAttr` _or_ an SSA value.
   It seems that this is sub-optimal, which is why the `std` dialect seems to have both `call`
   and `indirect_call`.

- `call`: https://mlir.llvm.org/docs/Dialects/Standard/#stdcall-callop. Has attribute `callee`
  of type `::mlir::FlatSymbolRefAttr`

- `call_indirect`: https://mlir.llvm.org/docs/Dialects/Standard/#stdcall_indirect-callindirectop Has operand `callee` of type `function type`.

This will lead to pain, because we have a `SymbolAttr` and an SSA value with the
same name, like so:

```mlir
// This is hopeless, we can have SSA values and symbol table entries with
// the same name.
hask.func @function {
    %function = hask.make_i32(1)
    hask.return (%function)
}
```

This round trips through MLIR `:(`. Big sad.

### Hacked `apSSA`:

got `apSSA` to accept both `@fib` and `%value`. I don't see this as a good
solution, primarily because later on, when we are trying to write stuff
that rewrites the IR, we will need to handle the two cases separately. 

- Plus, it's not possible to stash this `SymbolAttr` which is the name of
  `@fib`, and the `mlir::Value` which is `%value` in the same `set/vector/container` data
  structure since they don't share a base class. 

- I guess the argument will be that we should store the _full_ `func @symbol = {... }`,
  which is an `Op`. But `Op` and `Value` don't share the same base class either?

## Tuesday, 14th July 2020

- added a `hask.force` to allow us to write `case e of ename { default -> ... }`
  as `ename = hask.force(e); …`

- This brings up another problem. Assume we have `y = case e of ename { default -> …; val = …; return val }`. We would
  like to make _emitting_ MLIR easy, so I took the decision to emit this as `hask.copy(...)`:

```mlir
//NEW
%ename = hask.force(%e)
...
%val = ...
%y = hask.copy(%val)
```                                                                      

Old (what we used to have):

```mlir
// old
%y = case %e of { ^default(%ename): …; %val = … ; return %val; }
```

- So we have a new instruction called `hask.copy`, which is necessary because one can't write `%y = %x`.
  It's a stupid hack around MLIR's (overly-restrictive) SSA form. It can be removed by a rewriter that replaces
  `%y = hask.copy(%x)` by replacing all uses of `%y` with `%x`.

### Another design for function calls

We can perhaps force all functions to be of the form:

```mlir
hask.func @fib {
  ...
  %fibref = constant @fib
  hask.apSSA(%fibref, %constant_one) // <- new proposal
  hask.apSSA(@fib, %constant_one) // <- current version
  This simplifies the use of the variable: We will always have an SSA variable
  as the called function.
}
```

### Can generate resonable code from Core:

```
// Main
// Core2MLIR: GenMLIR BeforeCorePrep
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
  %lambda_0 = hask.lambdaSSA(%i_a12E) {
    %case_1 = hask.caseSSA  %i_a12E
    ["default" ->
    {
    ^entry(%ds_d1jZ: !hask.untyped):
      # app_2 = (-# i_a123)
      %app_2 = hask.apSSA(%minus_hash, %i_a12E)
      # lit_3 = 1
      %lit_3 = hask.make_i32(1)
      # app_4 = (-# i_a123 1)
      %app_4 = hask.apSSA(%app_2, %lit_3)
      # app_5 = fib (-# i_a123 1)
      %app_5 = hask.apSSA(@fib, %app_4)
      # wild_00 = force(fib(-# i_a123 1))
      %wild_00 = hask.force (%app_5)
      # app_7 = fib(i)
      %app_7 = hask.apSSA(@fib, %i_a12E)
      # wild_X5 = force(fib(i))
      %wild_X5 = hask.force (%app_7)
      # app_7 = (+# force(fib(i)))
      %app_9 = hask.apSSA(%plus_hash, %wild_X5)
      # app_10 = (+# force(fib(i)) fib(-# i_a123 1))
      %app_10 = hask.apSSA(%app_9, %wild_00)
    hask.return(%app_10)
    }
    ]
    [0 ->
    {
    ^entry(%ds_d1jZ: !hask.untyped):
    hask.return(%i_a12E)
    }
    ]
    [1 ->
    {
    ^entry(%ds_d1jZ: !hask.untyped):
    hask.return(%i_a12E)
    }
    ]
    hask.return(%case_1)
  }
  hask.return(%lambda_0)
  }
hask.dummy_finish
}
// ============ Haskell Core ========================
//Rec {
//-- RHS size: {terms: 21, types: 4, coercions: 0, joins: 0/0}
//main:Main.fib [Occ=LoopBreaker]
//  :: ghc-prim-0.5.3:GHC.Prim.Int# -> ghc-prim-0.5.3:GHC.Prim.Int#
//[LclId]
//main:Main.fib
//  = \ (i_a12E :: ghc-prim-0.5.3:GHC.Prim.Int#) ->
//      case i_a12E of {
//        __DEFAULT ->
//          case main:Main.fib (ghc-prim-0.5.3:GHC.Prim.-# i_a12E 1#)
//          of wild_00
//          { __DEFAULT ->
//          (case main:Main.fib i_a12E of wild_X5 { __DEFAULT ->
//           ghc-prim-0.5.3:GHC.Prim.+# wild_X5
//           })
//            wild_00
//          };
//        0# -> i_a12E;
//        1# -> i_a12E
//      }
//end Rec }
//
//-- RHS size: {terms: 5, types: 0, coercions: 0, joins: 0/0}
//main:Main.$trModule :: ghc-prim-0.5.3:GHC.Types.Module
//[LclIdX]
//main:Main.$trModule
//  = ghc-prim-0.5.3:GHC.Types.Module
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "main"#)
//      (ghc-prim-0.5.3:GHC.Types.TrNameS "Main"#)
//
//-- RHS size: {terms: 7, types: 3, coercions: 0, joins: 0/0}
//main:Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main:Main.main
//  = case main:Main.fib 10# of { __DEFAULT ->
//    base-4.12.0.0:GHC.Base.return
//      @ ghc-prim-0.5.3:GHC.Types.IO
//      base-4.12.0.0:GHC.Base.$fMonadIO
//      @ ()
//      ghc-prim-0.5.3:GHC.Tuple.()
//    }
//
//-- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
//main::Main.main :: ghc-prim-0.5.3:GHC.Types.IO ()
//[LclIdX]
//main::Main.main
//  = base-4.12.0.0:GHC.TopHandler.runMainIO @ () main:Main.main
//
```

### Reading the rewriter/lowering documentation of MLIR
- https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/
- https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/
- https://github.com/bollu/musquared/blob/master/lib/LeanDialect.cpp#L824
- https://github.com/bollu/musquared/blob/master/include/LeanDialect.h#L220

### Updating `fibstrict`
- I managed to eliminate the need for `hask.copy` from the auto-generated code,
  but I don't really understand _how_. I need to think about this a bit more, and a bit more carefully.
  This stuff is subtle!

The new readable hand-written `fibstrict` (adapted from the auto-generated code) is:

```
// Main
// Core2MLIR: GenMLIR BeforeCorePrep
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      %retval = hask.caseSSA  %i
      ["default" -> { ^entry(%default_random_name: !hask.untyped): // todo: remove this defult
        %i_minus = hask.apSSA(%minus_hash, %i)
        %lit_one = hask.make_i32(1)
        %i_minus_one = hask.apSSA(%i_minus, %lit_one)
        %fib_i_minus_one = hask.apSSA(@fib, %i_minus_one)
        %force_fib_i_minus_one = hask.force (%fib_i_minus_one) // todo: this is extraneous!
        %fib_i = hask.apSSA(@fib, %i)
        %force_fib_i = hask.force (%fib_i) // todo: this is extraneous!
        %plus_force_fib_i = hask.apSSA(%plus_hash, %force_fib_i)
        %fib_i_plus_fib_i_minus_one = hask.apSSA(%plus_force_fib_i, %force_fib_i_minus_one)
        hask.return(%fib_i_plus_fib_i_minus_one) }]
      [0 -> { ^entry(%default_random_name: !hask.untyped):
        hask.return(%i) }]
      [1 -> { ^entry(%default_random_name: !hask.untyped):
        hask.return(%i) }]
      hask.return(%retval)
    }
    hask.return(%lambda)
  }
hask.dummy_finish
}
```
- It is quite unclear to me why GHC generates the extra `hask.force` around the fibs
  when it knows perfectly well that they are strict values. It is a bit weird I feel.
- Perhaps they later use demand analysis to learn these are strict. Not sure.

# Wednesday: 16th July 2020

- Decided I couldn't use the default `opt` stuff any longer, since I now need
  fine grained control over which passes are run how.

- Stole code from toy to do the printing. Unfortunately, toy only uses
  `module->dump()`.
- What I want to do is to print the module to `stdout`. `module->print()`
  needs an `OpAsmPrinter`. Kill me.
- Let's see how `MlirOptMain` prints to the output file.

```cpp
LogicalResult mlir::MlirOptMain(raw_ostream &os,
                                std::unique_ptr<MemoryBuffer> buffer,
                                const PassPipelineCLParser &passPipeline,
                                bool splitInputFile, bool verifyDiagnostics,
                                bool verifyPasses,
                                bool allowUnregisteredDialects) {
  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  if (splitInputFile)
    return splitAndProcessBuffer(
        std::move(buffer),
        [&](std::unique_ptr<MemoryBuffer> chunkBuffer, raw_ostream &os) {
          return processBuffer(os, std::move(chunkBuffer), verifyDiagnostics,
                               verifyPasses, allowUnregisteredDialects,
                               passPipeline);
        },
        os);

  return processBuffer(os, std::move(buffer), verifyDiagnostics, verifyPasses,
                       allowUnregisteredDialects, passPipeline);
}
```

- OK, so we need to know how `processBuffer` works:

```cpp
static LogicalResult processBuffer(raw_ostream &os,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   bool verifyDiagnostics, bool verifyPasses,
                                   bool allowUnregisteredDialects,
                                   const PassPipelineCLParser &passPipeline) {
  ...
  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return performActions(os, verifyDiagnostics, verifyPasses, sourceMgr,
                          &context, passPipeline);
  }
  ...
}
```
- Recursive into `performActions`:

```cpp
static LogicalResult performActions(raw_ostream &os, bool verifyDiagnostics,
                                    bool verifyPasses, SourceMgr &sourceMgr,
                                    MLIRContext *context,
                                    const PassPipelineCLParser &passPipeline) {
  ...
  // Print the output.
  module->print(os);
  os << '\n';
  ...
}
```

- WTF, so a `raw_ostream` satisfies an `OpAsmPrinter`? no way
- OK, I found the overloads. Weird that `VSCode`'s intellisense missed these
  and pointed me to the wrong location. I should stop trusting it:

```cpp
class ModuleOp
...
public:
...
  /// Print the this module in the custom top-level form.
  void print(raw_ostream &os, OpPrintingFlags flags = llvm::None);
  void print(raw_ostream &os, AsmState &state,
             OpPrintingFlags flags = llvm::None);
...
}
```

- Cool, so I can just say `module->print(llvm::outs())` and it's going to print
  it.

- OK, I now need to figure out how to get the MLIR framework to pick up
  my `ApSSARewriter`. Jesus, getting used to MLIR is a pain. I suppose
  some of it has to do with my refusal to use TableGen. But then again, TableGen
  just makes me feel more lost, so it's not a good style.

- Doing exactly what `toy ch3` suggests does not seem to work. OK, I guess I'll
  read what `mlir::createCanonicalizerPass` does, since that's what seems
  to be responsible for adding my rewriter in the code snippet:

```cpp
  if (enableOptimization) {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "Run of canonicalizer failed.\n";
      return 4;
    }
  }
```

- It's darkly funny to me that no snippet of MLIR has ever worked out of the box.
  Judging from past experience, I estimate an hour of searching and pain.

- OK, first peppered code with `assert`s to see how far it is getting:


```cpp
struct UncurryApplication : public mlir::OpRewritePattern<ApSSAOp> {
  UncurryApplication(mlir::MLIRContext *context)
      : OpRewritePattern<ApSSAOp>(context, /*benefit=*/1) {
          assert(false && "uncurry application constructed")
      }
  mlir::LogicalResult
  matchAndRewrite(ApSSAOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(false && "UncurryApplication::matchAndRewrite called");
    return failure();
  }
};

void ApSSAOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  assert(false && "ApSSAOp::getCanonicalizationPatterns called");
  results.insert<UncurryApplication>(context);
}
```

- FML, literally no `assert` fails. OK, I guess I actually do need to read 
  `mlir::createCanonicalizerPass`: https://github.com/llvm/llvm-project/blob/a5b9316b24ce1de54ae3ab7a5254f0219fee12ac/mlir/lib/Transforms/Canonicalizer.cpp#L41

```cpp
namespace {
/// Canonicalize operations in nested regions.
struct Canonicalizer : public CanonicalizerBase<Canonicalizer> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;

    // TODO: Instead of adding all known patterns from the whole system lazily
    // add and cache the canonicalization patterns for ops we see in practice
    // when building the worklist.  For now, we just grab everything.
    auto *context = &getContext();
    for (auto *op : context->getRegisteredOperations())
      op->getCanonicalizationPatterns(patterns, context); // <- this should be asserting!
    Operation *op = getOperation();
    applyPatternsAndFoldGreedily(op->getRegions(), patterns);
  }
};
} // end anonymous namespace
```

- OK, progress made. It's the difference between:

```cpp
// v this, as I understand it, runs only inside `mlir::FuncOp`.
pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass()); 
// v this runs on everything.
pm.addPass(mlir::createCanonicalizerPass());
```

- Of course, I need to understand this properly. So let's figure out WTF
  `addNestedPass` actually means: https://github.com/llvm/llvm-project/blob/6d15451b175293cc98ef1d0fd9869ac71904e3bd/mlir/include/mlir/Pass/PassManager.h#L77

```cpp
/// Add the given pass to a nested pass manager for the given operation kind
/// `OpT`.
template <typename OpT> void addNestedPass(std::unique_ptr<Pass> pass) {
  nest<OpT>().addPass(std::move(pass));
}
```

- What is `nest`? https://github.com/llvm/llvm-project/blob/6d15451b175293cc98ef1d0fd9869ac71904e3bd/mlir/include/mlir/Pass/PassManager.h#L65
```cpp
  /// Nest a new operation pass manager for the given operation kind under this
  /// pass manager.
  OpPassManager &nest(const OperationName &nestedName);
  OpPassManager &nest(StringRef nestedName);
  template <typename OpT> OpPassManager &nest() {
    return nest(OpT::getOperationName());
  }
```

- This file in MLIR about passes seems good: https://github.com/llvm/llvm-project/blob/master/mlir/docs/PassManagement.md

- Got nerd sniped by the devloping story of twitter being hacked: https://news.ycombinator.com/item?id=23851275#23852853.
  High profile accounts are asking folks to donate to a BTC address. Seems like a really weak use of 
  incredible amounts of power. Tinfoil hat theory: this is a demonstration.
- Twitter acknowledgement of the hack: https://twitter.com/TwitterSupport/status/1283518038445223936

- OK, I'm actually writing my pass now. Jesus, I realised that my implemtnation of `ApSSA` is really annoying and perhaps
  mildly broken.

- I allow the first parameter to be either a `Symbol` or a `Value`. Now that I need to rewrite stuff,
  how do I find out *what* the symbol is? The MLIR docs wax poetic about symbol tables:
   https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table

- I tried to give my `ModuleOp` the trait `OpTrait::SymbolTable`. All hell
  has broken loose.

- I now can't have a result from my module (makes sense). So I make it
  `OpTrait::ZeroResult` and remove the `hask.dummy_finish` thing I had
  hanging around.

- This somehow destroys the correctness of my module, with errors:

```
./fib-strict.mlir:4:18: error: block with no terminator
    %plus_hash = hask.make_data_constructor<"+#">
```

Here is my file:

```mlir
// Main
// Core2MLIR: GenMLIR BeforeCorePrep
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
...
```

- I confess, I do not know what it is talking about. What terminator? Why ought
  I terminate the block? Do I need to terminate it with an operator
  that returns zero results?. This to me seems the most reasonable explanation

- Whatever the explanation, that is an _atrocious_ place to put the error marker.
  Maybe I send a patch.

- Nice, I make progress. Now my printing of `ApSSA` is broken, my module compiles. I get the amazing backtrace:

```
(gdb) run
The program being debugged has been started already.
Start it from the beginning? (y or n) y
Starting program: /home/bollu/work/mlir/coremlir/build/bin/hask-opt ./fib-strict.mlir
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
- parse:197attribute: "+#"
- parse:198attribute ty: none
- parse:197attribute: "-#"
- parse:198attribute ty: none
- parse:197attribute: "()"
- parse:198attribute ty: none
-parse:454:%i
parse:396
parse:398
parse:413
parse:417
parse:420
parse:423
Module (no optimization):

module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    hask.func @fib {
      %3 = hask.lambdaSSA(%arg0) {
        %4 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          %5 = hask.apSSA(%5,hask-opt: /home/bollu/work/mlir/llvm-project/mlir/lib/IR/Value.cpp:22: mlir::Value::Value(mlir::Operation*, unsigned int): Assertion `op->getNumResults() > resultNo && "invalid result number"' failed.
```

- Had to add builder for `ApSSAOp`:

```cpp
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    Value fn, SmallVectorImpl<Value> params);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    FlatSymbolRefAttr fn, SmallVectorImpl<Value> params);
```

- I find it interesting that the builder API is specified entirely through
  mutation of the `OperationState`. I would love to have discussions
  with the folks who designed this API to undertand their ideas for why
  they did it this way.

- The API is getting fugly, because everywhere this dichotomy between having a `Value`
  and having a `SymbolRef` keeps showing up. Is it really this complicated? Mh.

- In LLVM, a `Function` is a `GlobalObject` is a `GlobalValue` is a `Constant` is a
  `User` which is a `Value`: https://llvm.org/doxygen/classllvm_1_1Function.html
- The reason it's able to break SSA is embedded in `Verifier`:https://github.com/llvm/llvm-project/blob/master/llvm/lib/IR/Verifier.cpp
- We only check that an instruction dominates all of its uses _of other instructions_:
  https://github.com/llvm/llvm-project/blob/master/llvm/lib/IR/Verifier.cpp#L4147;
  https://github.com/llvm/llvm-project/blob/master/llvm/lib/IR/Verifier.cpp#L4292

```cpp
...
// https://github.com/llvm/llvm-project/blob/master/llvm/lib/IR/Verifier.cpp#L4147
void Verifier::verifyDominatesUse(Instruction &I, unsigned i) {
  Instruction *Op = cast<Instruction>(I.getOperand(i));
  ...
  if (!isa<PHINode>(I) && InstsInThisBlock.count(Op))
    return;
  ...
  const Use &U = I.getOperandUse(i);
  Assert(DT.dominates(Op, U),
         "Instruction does not dominate all uses!", Op, &I);
}
...
// https://github.com/llvm/llvm-project/blob/master/llvm/lib/IR/Verifier.cpp#L4292
} else if (isa<Instruction>(I.getOperand(i))) {
  verifyDominatesUse(I, i);
} 

```

- On the other hand, if the instruction has a use of a _function_, then we check
  other (unrelated) properties:

```cpp
    if (Function *F = dyn_cast<Function>(I.getOperand(i))) {
      // Check to make sure that the "address of" an intrinsic function is never
      // taken.
      Assert(!F->isIntrinsic() ||
                 (CBI && &CBI->getCalledOperandUse() == &I.getOperandUse(i)),
             "Cannot take the address of an intrinsic!", &I);
      Assert(
          !F->isIntrinsic() || isa<CallInst>(I) ||
              F->getIntrinsicID() == Intrinsic::donothing ||
              F->getIntrinsicID() == Intrinsic::coro_resume ||
              F->getIntrinsicID() == Intrinsic::coro_destroy ||
              F->getIntrinsicID() == Intrinsic::experimental_patchpoint_void ||
              F->getIntrinsicID() == Intrinsic::experimental_patchpoint_i64 ||
              F->getIntrinsicID() == Intrinsic::experimental_gc_statepoint ||
              F->getIntrinsicID() == Intrinsic::wasm_rethrow_in_catch,
          "Cannot invoke an intrinsic other than donothing, patchpoint, "
          "statepoint, coro_resume or coro_destroy",
          &I);
      Assert(F->getParent() == &M, "Referencing function in another module!",
             &I, &M, F, F->getParent());
    } else if (BasicBlock *OpBB = dyn_cast<BasicBlock>(I.getOperand(i))) {
```

- So really, LLVM had a sort of *exception* for functions. Or, rather, it's
  notion of SSA was strictly for *instructions*, not for *all Ops* 
  (Ops in the MLIR sense of the word).

- OK, our code now looks like:

```
Module (+optimization):

module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    hask.func @fib {
      %3 = hask.lambdaSSA(%arg0) {
        %4 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          %5 = hask.apSSA(%1, %arg0) // <- dead!
          %6 = hask.make_i32(1 : i64)
          %7 = hask.apSSA(%1, %arg0, %6)
          %8 = hask.apSSA(@fib, %7)
          %9 = hask.force(%8)
          %10 = hask.apSSA(@fib, %arg0)
          %11 = hask.force(%10)
          %12 = hask.apSSA(%0, %11) // <- dead!
          %13 = hask.apSSA(%0, %11, %9)
          hask.return(%13)
        }]
 [0 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]
 [1 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]

        hask.return(%4)
      }
      hask.return(%3)
    }
    hask.dummy_finish
  }
}
```
- We do fuse away the applications. But I now have dead instructions. Need
  to find the pass that eliminates dead values. Looks like CSE takes
  care of this, so I'll just run CSE and see what output I get.


- Good reference to learn how to deal with symbols, the inliner: https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/lib/Transforms/Inliner.cpp
- InliningUtils that contains the actually useful function `inlineCall`:
  https://github.com/llvm/llvm-project/blob/22219cfc6a2a752c53238df4ceea342672392818/mlir/lib/Transforms/Utils/InliningUtils.cpp
- List of passes in MLIR: https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/include/mlir/Transforms/Passes.h 

- After CSE, we get the code:

```cpp
module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    hask.func @fib {
      %3 = hask.lambdaSSA(%arg0) {
        %4 = hask.caseSSA %arg0 ["default" ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          %5 = hask.make_i32(1 : i64)
          %6 = hask.apSSA(%1, %arg0, %5)
          %7 = hask.apSSA(@fib, %6)
          %8 = hask.force(%7)
          %9 = hask.apSSA(@fib, %arg0)
          %10 = hask.force(%9)
          %11 = hask.apSSA(%0, %10, %8)
          hask.return(%11)
        }]
 [0 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]
 [1 : i64 ->  {
        ^bb0(%arg1: !hask.untyped):  // no predecessors
          hask.return(%arg0)
        }]

        hask.return(%4)
      }
      hask.return(%3)
    }
    hask.dummy_finish
  }
}
```

- We now need to lower `force(apSSA(...))` and `apSSA(+#, … )`, `apSSA(-#, … )`,
  and `make_i32`. Time to learn the lowering infrastructure properly.

- Started thinking of how to lower to LLVM. There's a huge problem: I don't know the type of `fib`. Now what? `:(`.
  For now, I can of course assume that all parameters are `i32`. This is, naturally, not scalable.

# Thursday, 17th July 2020

- Of course the MLIR-LLVM dialect does not have switch case: https://reviews.llvm.org/D75433.
- I guess I should reduce my code to `scf` then? it's pretty unclear to me
  what the expectation is.
- Alternatively, I just emit a bunch of `cmp`s. This is really really annoying.
  Fuck it, SCF it is.
- First I work on lowering `hask.fib` and `hask.func` to standard, then
  I lower case to `SCF` with its if-then-else support. 
- If this mix of standard-and-SCF works, that will be great!

```cpp
/// This class provides a CRTP wrapper around a base pass class to define
/// several necessary utility methods. This should only be used for passes that
/// are not suitably represented using the declarative pass specification(i.e.
/// tablegen backend).
template <typename PassT, typename BaseT> class PassWrapper : public BaseT {
public:
  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Pass *pass) {
    return pass->getTypeID() == TypeID::get<PassT>();
  }

protected:
  PassWrapper() : BaseT(TypeID::get<PassT>()) {}

  /// Returns the derived pass name.
  StringRef getName() const override { return llvm::getTypeName<PassT>(); }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<PassT>(*static_cast<const PassT *>(this));
  }
};
```

Why do we need a `PassWrapper`? whatever. I defined my own pass as:

```cpp
namespace {
struct LowerHaskToStandardPass
    : public PassWrapper<LowerHaskToStandardPass, OperationPass<ModuleOp>> {
  void runOnOperation();
};
} // end anonymous namespace.
void LowerHaskToStandardPass::runOnOperation() {
  this->getOperation();
  assert(false && "running lower hask pass");
}
std::unique_ptr<mlir::Pass> createLowerHaskToStandardPass() {
  return std::make_unique<LowerHaskToStandardPass>();
}
```

which of course, greets me with the delightful error:

```
Module (no optimization):hask-opt: /home/bollu/work/mlir/llvm-project/mlir/lib/Pass/Pass.cpp:275:
mlir::OpPassManager::OpPassManager(mlir::OperationName, bool):
Assertion `name.getAbstractOperation()->hasProperty( OperationProperty::IsolatedFromAbove) &&
"OpPassManager only supports operating on operations marked as " "'IsolatedFromAbove'"' failed.
Aborted (core dumped)
../build/bin/hask-opt ./fib-strict-roundtrip.mlir
Module (no optimization):

module {
}hask-opt: /home/bollu/work/mlir/llvm-project/mlir/lib/Pass/Pass.cpp:275:
mlir::OpPassManager::OpPassManager(mlir::OperationName, bool):
Assertion `name.getAbstractOperation()->hasProperty( OperationProperty::IsolatedFromAbove) &&
"OpPassManager only supports operating on operations marked as " "'IsolatedFromAbove'"' failed.
```

- Now I need to read what `IsolatedFromAbove` is. IIRC, it can't
  use values that are defined outside/ above it in terms of depth?

The MLIR docs say:
> Passes are expected to not modify operations at or above the current
> operation being processed.
> If the operation is not isolated,
> it may inadvertently
> modify the use-list of an operation it is not supposed to modify.

- Indeed, the question is precisely _what_ and _why_ am I "not supposed to modify".
- So I made the `ModuleOp` `IsolatedFromAbove`.
- I now realise that I'm confused. I need to change both my functions from `hask.func` to the regular `std.func`
  while simultaneously changing my call instructions from `apSSA` to `std.call`.
  So the IR in between will be illegal [indeed, "nonsensical"]?
  We shall see how this goes.

- OK, I see, so we are expected to replace the _root_ operation in a conversion pass. 
  So this:

```cpp
namespace {
struct LowerHaskToStandardPass
    : public PassWrapper<LowerHaskToStandardPass, OperationPass<ModuleOp>> {
  void runOnOperation();
};
} // end anonymous namespace.

void LowerHaskToStandardPass::runOnOperation() {
    ConversionTarget target(getContext());
  OwningRewritePatternList patterns;
  patterns.insert<HaskFuncOpLowering>(&getContext());
  patterns.insert<HaskApSSAOpLowering>(&getContext());

  if (failed(applyPartialConversion(this->getOperation(), target, patterns))) {
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    llvm::errs() << "fn\nvvvv\n";
    getOperation().dump() ;
    llvm::errs() << "\n^^^^^\n";
    signalPassFailure();
    assert(false);
  }
```
dies with:

```
Module (no optimization):Module: lowering to standard+SCF...hask-opt:
/home/bollu/work/mlir/llvm-project/mlir/lib/Transforms/DialectConversion.cpp:1504:
mlir::LogicalResult
  {anonymous}::OperationLegalizer
  ::legalizePatternResult(mlir::Operation*,
      const mlir::RewritePattern&,
      mlir::ConversionPatternRewriter&,
      {anonymous}::RewriterState&):
  Assertion `(replacedRoot || updatedRootInPlace()) &&
  "expected pattern to replace the root operation"' failed.
```

So it appears that in a `ModuleOp`, I _must_ replace a module. So I guess
the "correct" thing to do is to have _separate_ conversion passes for 
each of my `HaskFuncOpLowering`, `HaskApSSAOpLowering`? I really don't
understand what the hell the invariants

- What is the rationale of `ConversionPattern : RewritePattern`? What new
  powers does `ConversionPattern` confer on me? `:(` I am generally sad panda
  because I have no idea why I need this tower of abstraction, it's not
  well motivated.

- Ookay, so I decided to replace my module with Standard. It dies with:
```
Module (no optimization):Module: lowering to standard+SCF...
hask-opt: /home/bollu/work/mlir/llvm-project/mlir/lib/IR/PatternMatch.cpp:142:
void mlir::PatternRewriter::replaceOpWithResultsOfAnotherOp(mlir::Operation*, mlir::Operation*):
Assertion `op->getNumResults() == newOp->getNumResults() &&
"replacement op doesn't match results of original op"' failed.
```

But that's ludicrous! 

```cpp


class ModuleOp : public Op<ModuleOp, OpTrait::ZeroResult, OpTrait::OneRegion, OpTrait::SymbolTable, OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hask.module"; };
  ...
};
```

```cpp
class ModuleOp
    : public Op<
          ModuleOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
          OpTrait::IsIsolatedFromAbove, OpTrait::AffineScope,
          OpTrait::SymbolTable,
          OpTrait::SingleBlockImplicitTerminator<ModuleTerminatorOp>::Impl,
          SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  static StringRef getOperationName() { return "module"; }
```

- Both of these have zero results! What drugs is the assert on?

- OK WTF?
- Ah I see:

```cpp
class ModuleOpLowering : public ConversionPattern {
public:
  explicit ModuleOpLowering(MLIRContext *context)
      : ConversionPattern(ApSSAOp::getOperationName(), 1, context) {}
                          // ^ I see, so I made a mistake here. 
```

- Damn, I am sleepy or something, this is quite obvious.
- OK, now my pass isn't even running:

```cpp
class ModuleOpLowering : public ConversionPattern {
public:
  explicit ModuleOpLowering(MLIRContext *context)
      : ConversionPattern(ModuleOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "vvvvvvvvvvvvvvvvvvvvvv\nop: " << *op << "^^^^^^^^^^^^^^\n";
    rewriter.replaceOpWithNewOp<mlir::ModuleOp>(op);
    assert(false); // should crash

    return success();
  }
};
```

- So it runs, but it seems to double my module? WTF is going on:

```cpp
class ModuleOpLowering : public ConversionPattern {
public:
  explicit ModuleOpLowering(MLIRContext *context)
      : ConversionPattern(ModuleOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ModuleOp>(op);
    return success();
  }
};
```
```
vvvvvvvvvvvvvvvvvvvvvvvvvvvv
Module (+optimization), lowered to Standard+SCF:


module {
  module {
    hask.module {
      %0 = hask.make_data_constructor<"+#">
      %1 = hask.make_data_constructor<"-#">
      %2 = hask.make_data_constructor<"()">
      hask.func @fib {
        %3 = hask.lambdaSSA(%arg0) {
          %4 = hask.caseSSA %arg0 ["default" ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            %5 = hask.make_i32(1 : i64)
            %6 = hask.apSSA(%1, %arg0, %5)
            %7 = hask.apSSA(@fib, %6)
            %8 = hask.force(%7)
            %9 = hask.apSSA(@fib, %arg0)
            %10 = hask.force(%9)
            %11 = hask.apSSA(%0, %10, %8)
            hask.return(%11)
          }]
 [0 : i64 ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            hask.return(%arg0)
          }]
 [1 : i64 ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            hask.return(%arg0)
          }]

          hask.return(%4)
        }
        hask.return(%3)
      }
      hask.dummy_finish
    }
  }
  module {
    hask.module {
      %0 = hask.make_data_constructor<"+#">
      %1 = hask.make_data_constructor<"-#">
      %2 = hask.make_data_constructor<"()">
      hask.func @fib {
        %3 = hask.lambdaSSA(%arg0) {
          %4 = hask.caseSSA %arg0 ["default" ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            %5 = hask.make_i32(1 : i64)
            %6 = hask.apSSA(%1, %arg0, %5)
            %7 = hask.apSSA(@fib, %6)
            %8 = hask.force(%7)
            %9 = hask.apSSA(@fib, %arg0)
            %10 = hask.force(%9)
            %11 = hask.apSSA(%0, %10, %8)
            hask.return(%11)
          }]
 [0 : i64 ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            hask.return(%arg0)
          }]
 [1 : i64 ->  {
          ^bb0(%arg1: !hask.untyped):  // no predecessors
            hask.return(%arg0)
          }]

          hask.return(%4)
        }
        hask.return(%3)
      }
      hask.dummy_finish
    }
  }
}^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

- I have no fucking clue WTF is happening `:(`
- Right, I built an `ApSSAOp` which I can't use because it's not `IsolatedFromAbove`. OK, I really don't understand
  the semantics of lowering.

- Read the dialect conversion document: https://mlir.llvm.org/docs/DialectConversion/

- OK, now I understand why we need `ConversionPattern`:

> When type conversion comes into play, the general Rewrite Patterns can no
> longer be used. This is due to the fact that the operands of the operation
> being matched will not correspond with the operands of the correct type as
> determined by TypeConverter. The operation rewrites on type boundaries must
> thus use a special pattern, the ConversionPattern

Also:

> If a pattern matches, it must erase or replace the op it matched on.
>  Operations can not be updated in place.
> Match criteria should not be based on the IR outside of the op itself. The
> preceding ops will already have been processed by the framework (although it
> may not update uses), and the subsequent IR will not yet be processed. This can
> create confusion if a pattern attempts to match against a sequence of ops (e.g.
> rewrite A + B -> C). That sort of rewrite should be performed in a separate
> pass.

- So it seems to me that my rewrite of `ApSSA(@plus_hash, %1, %2) -> addi %1 %2`
  should be a separate Pass? and cannot reuse the infrastructure?


- To convert the types of block arguments within a Region, a custom hook 
  on the `ConversionPatternRewriter` must be invoked; `convertRegionTypes`

- I guess I should be using the more general `PatternRewriter` and `applyPatternsAndFoldGreedily`?
  Or can I not, because I need a `ConversionPattern`? Argh, this is so poorly
  documented.

-`VectorToSCF.cpp`: https://github.com/llvm/llvm-project/blob/master/mlir/lib/Conversion/VectorToSCF/VectorToSCF.cpp
- `VectorToSCF.h`: https://github.com/llvm/llvm-project/blob/master/mlir/lib/Conversion/VectorToSCF/VectorToSCF.cpp

# Tuesday, 18th August 2020

- Lowering our IR down to LLVM. Currently hacking the shit out of it,
  assuming all our types are int, etc. We then fix it gradually as we
  get more online, as per our iteration strategy.
- Currently, I'm getting annoyed at the non-existence of a `RegionTrait`
  called `SingleBlockExplicitTerminator`: this is precisely what my `func` is:
  it should just create a `lambda` and then return the `lambda`. Hm, perhaps
  I should put this information in an attribute. Not sure. Oh well.
- What is going on with `LLVMTypeConverter`? Why does it exist?
- For whatever reason, any IR I generate from the legalization pass
  mysteriously vanishes after being generated. I presume I'm being really
  stupid and missing something extremely obvious. 
- Got annoyed at the MLIR documentation, so spent some time messing with doxygen
  to get both (1) very detailed doxygen pages, and also (2) man pages. Hopefully
  this helps me search for things faster when it comes to the sprawling MLIR
  sources.
- There are lots of design concerns that I'm basically giving up on for the
  first iteration. Non-exhaustive list: (1) we need to at least know the types
  of functions when we lower to MLIR, to the granularity of int-value-or-boxed-value.
  I currently assume everything is `int`. (2) We need to go through the usual
  pain in converting from the nice lazy representation to the `void*` mess that
  is representing closures inside LLVM. This too needs to know types to know
  how much space to allocate. Completely ignore these issues as well.

# Thursday, 20th August 2020
- Yay, more kludge to get MLIR to behave how I want:

```cpp
llvm::errs() << "debugging? " << ::llvm::DebugFlag << "\n";
LLVM_DEBUG({ assert(false && "llvm debug exists"); });
::llvm::DebugFlag = true; 
```

- I **manually** turn on debugging. This is, of course, horrible. On the other
  hand, I'm really not sure what the best practice is. When it came to developing
  with LLVM, since we would always run with `opt`, things "just worked". This
  time around, I'm not sure how we are expected to allow the `llvm::CommandLine`
  machinery to kick in, without explicitly invoking said machinery.

- In my `hask-opt.cpp` file, I used to use:

```cpp
  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects))) {
    return 1;
  }
```

But I then saw that the toy tutorials themselves don't do this. They use:

```cpp
mlir::registerAsmPrinterCLOptions();
mlir::registerMLIRContextCLOptions();
mlir::registerPassManagerCLOptions();

cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
```

So I presume that this `ParseCommandLineOptions` is going to launch the LLVM
machinery.

- Anyway, here's what the debug info of the legalizer spits out:

```cpp
    ** Erase   : 'hask.func'(0x560b1f99f5d0)

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x560b1f99f640) {
      * Fold {
      } -> FAILURE : unable to fold
    } -> FAILURE : no matched legalization pattern
    //===-------------------------------------------===//
  } -> FAILURE : operation 'func'(0x0000560B1F99F640) became illegal after block action
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//
```

- I don't understand 'became illegal after block action'. Time to read the
  sources.

- OK, so we can do the simplest thing known to man: delete the entire `hask.func`

```
// Input
// Debugging file: Can do anything here.
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      hask.return(%unit_tuple)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
```

```
// Output
module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    hask.dummy_finish
  }
}
```


- So to generate a `FuncOp`, I apparently need to **explicitly call**
  `target.addLegalOp<FuncOp>()`, even though I have a
  `target.addLegalDialect<mlir::StandardOpsDialect>()`.

```cpp
target.addLegalDialect<mlir::StandardOpsDialect>();
// Why do I need this? Isn't adding StandardOpsDialect enough?
target.addLegalOp<FuncOp>(); <- WHY?
```

- I really don't understand what's happening `:(`. I want to understand why
  `FuncOp` is not considered legal-by-default on marking `std` legal. Either
  (i) `FuncOp` does not, in fact, belong to `std`, or (ii) there is some
  kind of precedence in the way in which the `addLegal*` rules kick in, where
  somehow `FuncOp` is becoming illegal? I don't even know.

- Anyway, we can now lower the `play.mlir` file from an empty `hask.func`
  to an empty `func`:

##### input
```
// INPUT
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      hask.return(%unit_tuple)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
```
##### lowered

```
// LOWERED
module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    func @fib_lowered()
    hask.dummy_finish
  }
}
```

- `LinalgToStandard` creates new functions with `FuncOp`: 
```cpp
//LinalgToStandard.cpp
// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
template <typename LinalgOp>
static FlatSymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr = rewriter.getSymbolRefAttr(fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(extractOperandTypes<LinalgOp>(op));
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = FunctionType::get(inputTypes, {}, rewriter.getContext());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  FuncOp funcOp =
      rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType,
                              ArrayRef<NamedAttribute>{});
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp.setAttr("llvm.emit_c_interface", UnitAttr::get(op->getContext()));
  return fnNameAttr;
}
...
void ConvertLinalgToStandardPass::runOnOperation() {
   auto module = getOperation();
   ConversionTarget target(getContext());
   target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect>();
   target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ReturnOp>();
   ...
```

- They too add `FuncOp` as legal manually. Man I wish I understood this.
  What dialect does `FuncOp, ModuleOp`, etc belong to?


- OK, we can now lower a dummy `hask.func` into a dummy `FuncOp`:

##### input

```cpp
// INPUT
hask.module {
  // vvvv unusedvvv
  %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      %foo = hask.make_data_constructor<"foo">
      hask.return(%foo)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
```

##### lowered
```cpp
 // LOWERED
 module {
  hask.module {
    %0 = hask.make_data_constructor<"+#">
    %1 = hask.make_data_constructor<"-#">
    %2 = hask.make_data_constructor<"()">
    func @fib_lowered(%arg0: !hask.untyped) {
      %3 = hask.make_data_constructor<"foo">
      hask.return(%3)
    }
    hask.dummy_finish
  }
}
```

- What I am actually interested is to have our function return a `%unit_tuple`,
  but that does not seem to be allowed because `FuncOp` has a `IsolatedFromAbove`
  trait. This is very strange: how do I use global data?

- I think I should be using a symbol, so my signature should read something
  like `hask.make_data_constructor @"+#"` or something like that to mark
  the data constructor as a global piece of information. Let me try and check
  that a `Symbol` is what I need.


- Fun fact: LLVM out-of-memorys if you hand it an uninitialized OperandType.

```cpp
OpAsmParser::OperandType scrutinee;
if(parser.resolveOperand(scrutinee, 
    parser.getBuilder().getType<UntypedType>(), 
    results)) { return failure(); } // BOOM! out of memory
```

- OK, we can now lower references to `make_data_constructor`:

##### input
```
hask.module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"

  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      // %foo_ref = constant @XXXX : () -> ()
      %f = hask.ref(@"+#")
      hask.return(%f)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
```

##### output

```
module {
  hask.module {
    hask.make_data_constructor +#
    hask.make_data_constructor -#
    hask.make_data_constructor ()
    vvv is a std func with a real argument.
    func @fib_lowered(%arg0: !hask.untyped) {
      %0 = hask.ref (@"+#")
      hask.return(%0)
    }
    hask.dummy_finish
  }
}                                                                                                                                      
```

- This `Symbol` thing is prone to breakage, I feel. For example, consider:

```
hask.func @fib {
  %lambda = hask.lambdaSSA(%i) {
      ...
      %fib_i = hask.apSSA(@fib, %i)
      ...
  }
}
```

- Upon lowering, if I generate a function called `@fib_lowered`, the code 
  [which passes verification] becomes:

```
func @fib_lowered(%arg0: !hask.untyped) {
      ...
      %fib_i = hask.apSSA(@fib, %i) <- still called fib!
      ...
  }
}
```

- The thing really, truly is a god damm symbol table, with a danging symbol
  of `@fib`. Is there some way to verify that we do not have a dangling `Symbol`
  in a module?

# Friday, 21 August 2020

- `ConversionPatternRewriter::mergeBlocks` is not defined in my copy of MLIR.
  Time to pull and waste a whole bunch of time in building `:(`
  my MLIR commit is [`7ddee0922fc2b8629fa12392e61801a8ad96b7af`](https://github.com/llvm/llvm-project/commit/7ddee0922fc2b8629fa12392e61801a8ad96b7af)
  `Tue Jun 23 16:07:44 2020 +0300`, with message `[NFCI][CostModel] Add const to Value*`
- I'm going to get the stuff other than `case` working before I pull and
  waste an hour or two compiling MLIR.

- Great, the type related things changed. Before, one created an non-parametric
  type using

```cpp
namespace HaskTypes {
  enum Types {
    // TODO: I don't really understand how this works. In that,
    //       what if someone else has another 
    Untyped = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  };
};

class UntypedType : public mlir::Type::TypeBase<UntypedType, mlir::Type,
                                               TypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == HaskTypes::Untyped; }
  static UntypedType get(MLIRContext *context) { return Base::get(context, HaskTypes::Types::Untyped); } 
};
```

- Now, I have no idea, this seems to not be the solution anymore :(

- It seems that in `Toy`, the stopped using the tablegen'd version of the
  dialect: [they define the dialect in C++](https://github.com/llvm/llvm-project/blob/e1cd7cac8a36608616d515b64d12f2e86643970d/mlir/examples/toy/Ch7/include/toy/Dialect.h#L54).
  I switched to doing this as well --- I prefer the C++ version at any rate.

- Making progress with  my pile-of-hacks. I replace the `case` with the
  body of the default, and I get this:

```
./playground.mlir:14:28: error: 'std.call' op 'fib' does not reference a valid function
        %fib_i_minus_one = hask.apSSA(@fib, %i_minus_one)
                           ^
./playground.mlir:14:28: note: see current operation: %1 = "std.call"(%0) {callee = @fib} : (i32) -> i32
===Lowering failed.===
===Incorrectly lowered Module to Standard+SCF:===


module {
  hask.module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
    func @fib_lowered(%arg0: i32) {
      %c1_i32 = constant 1 : i32
      %0 = subi %arg0, %c1_i32 : i32
      %1 = call @fib(%0) : (i32) -> i32
      %2 = call @fib(%arg0) : (i32) -> i32
      %3 = addi %2, %1 : i32
      return %3 : i32
    }
    hask.dummy_finish
  }
}
```

- I am not sure why `fib` does not reference a valid function! What on earth
  is it talking about?

```cpp
static LogicalResult verify(CallOp op) {
  // Check that the callee attribute was specified.
  auto fnAttr = op.getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");
  auto fn =
      op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
  if (!fn)
    return op.emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid function";
```

- So I think the problem is that it doesn't have a `parentOfType<ModuleOp>`?


- I now generate this:

```cpp
module {
  module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
    func @fib(%arg0: i32) -> i32 {
      %c0_i32 = constant 0 : i32
      %0 = cmpi "eq", %c0_i32, %arg0 : i32
      scf.if %0 {
      ^bb1(%6: !hask.untyped):  // no predecessors
        return %arg0 : i32
      }
      %c1_i32 = constant 1 : i32
      %1 = cmpi "eq", %c1_i32, %arg0 : i32
      scf.if %1 {
      ^bb1(%6: !hask.untyped):  // no predecessors
        return %arg0 : i32
      }
      %c1_i32_0 = constant 1 : i32
      %2 = subi %arg0, %c1_i32_0 : i32
      %3 = call @fib(%2) : (i32) -> i32
      %4 = call @fib(%arg0) : (i32) -> i32
      %5 = addi %4, %3 : i32
      return %5 : i32
    }
  }
}
```

which fails legalization with:

> ./playground.mlir:9:17: error: 'scf.if' op expects region #0 to have 0 or 1 blocks

Not sure which region is `region #0`. Need to read the code where the
error comes from.


- The MLIR API sucks with 32 bit numbers `:(` The problem is that `IntegerAttr`
  is parsed as 64-bit by default. So to get to 32 bit values, one needs
  to juggle a decent amount. By switching to 64-bit as the
  default, I got quite a bit of code cleanup:

```patch
-            IntegerAttr lhsVal = caseop.getAltLHS(i).cast<IntegerAttr>();
-            mlir::IntegerAttr lhsI32 =
-                mlir::IntegerAttr::get(rewriter.getI32Type(),lhsVal.getInt());
             mlir::ConstantOp lhsConstant =
-                rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), lhsI32);
-
-            llvm::errs() << "- lhs constant: " << lhsConstant << "\n";
+                rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(),
+                                                  caseop.getAltLHS(i));
```

# Monday, 24 August 2020
- See under "Newest to Oldest". I changed the organization strategy to keep the 
  newest log at the top.