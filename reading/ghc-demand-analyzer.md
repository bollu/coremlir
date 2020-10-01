# Demand analyser in GHC

**[This page was taken from the page on the GHC wiki](https://gitlab.haskell.org/ghc/ghc/-/wikis/commentary/compiler/demand/)**


This page explains basics of the so-called demand analysis in GHC,
comprising strictness and absence analyses. Meanings of demand
signatures are explained and examples are provided. Also, components
of the compiler possibly affected by the results of the demand
analysis are listed with explanations provided.

- The [demand-analyser draft paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/03/demand-jfp-draft.pdf)
  is as yet unpublished, but gives the most accurate overview of the
  way GHC's demand analyser works.

---

## Demand signatures


Let us compile the following program with `-O2 -ddump-stranal` flags:

```wiki
f c p = case p 
          of (a, b) -> if c 
                       then (a, True) 
                       else (True, False)
```


The resulting demand signature for function `f` will be the following one:

```wiki
Str=DmdType <S,U><S,U(UA)>m
```


This should be read as "`f` puts stricts demands on both its arguments
(hence, `S`); `f` might use its first and second arguments. but in the
second argument (which is a product), the second component is
ignored". The suffix `m` in the demand signature indicates that the
function returns **CPR**, a constructed product result (for more
information on CPR see the JFP paper 
[Constructed Product Result Analysis for Haskell](http://research.microsoft.com/en-us/um/people/simonpj/Papers/cpr/index.htm)).


Current implementation of demand analysis in Haskell performs
annotation of all binders with demands, put on them in the context of
their use. For functions, it is assumed, that the result of the
function is used strictly. The analysis infers strictness and usage
information separately, as two components of a cartesian product
domain. The same analysis also performs inference CPR and bottoming
properties for functions, which can be read from the suffix of the
signature. Demand signatures of inner definitions may also include
*demand environments* that indicate demands, which a closure puts to
its free variables, once strictly used, e.g. the signature

```wiki
Str=DmdType <L,U> {skY-><S,U>}
```


indicates that the function has one parameter, which is used lazily
(hence `<L,U>`), however, when its result is used strictly, the free
variable `skY` in its body is also used strictly.

### Grammar

This a simple grammar extracted from the `Outputable` instances as of GHC 8.11:
```python
DmdType    := JointDmd* Divergence

# Joint strictness/usage demand
JointDmd   := '<' StrDmd ',' UseDmd '>'

####################################
# Strictness demands
####################################

StrDmd     := 'B'                          # HyperStr: Diverges if forced (bottom of lattice)
            | 'C' '(' StrDmd ')'           # SCall: Call demand
            | 'S' '(' ArgStr* ')'          # SProd: Product demand
            | 'S'                          # HeadStr: Forced only to WHNF

# Argument strictness
ArgStr     := 'L'                          # Lazy: Argument not necessarily demanded
            | StrDmd                       # Strict: Places given strictness demand on argument

####################################
# Usage demands
####################################

UseDmd     := 'U'                          # Used: Top of lattice
            | 'U' '(' (ArgUse ',')* ')'    # UProd: Used only for values of product type
            | 'C' Count '(' UseDmd ')'     # UCall: Used only for values of function type
            | 'H'                          # UHead: Used, but only to WHNF; components definitely not used

# Argument usage
ArgUse     := 'A'                          # Abs: Definitely unused (bottom of lattice)
            | '1*' UseDmd                  # Use Once: Used with the given usage demand exactly once
            | UseDmd                       # Use Many: Used with the given usage demand more than once

# Usage cardinality
Count      := '1'                          # Once
            | ''                           # Many times

# Divergence
Divergence := 'b'                          # Diverges: Definitely divergences but *doesn't* throw a precise exception.
            | 'x'                          # ExnOrDiv: Definitely diverges or throws a precise exception.
            | ''                           # Dunno: May or may not diverge

####################################
# Constructed Product Result types
####################################

CprType    := Arity CprResult              # The arity is the number of value arguments necessary
                                           # for the expression to reduce to CprResult.

CprResult  := ''                           # NoCPR: No CPR information (top of lattice)
            | 'm' ConTag                   # ConCPR: The result is the constructor identified by ConTag
            | 'b'                          # BotCPR: Evaluation bottoms (bottom of lattice)
```

### Demand descriptions



Strictness demands


- `B` -- a *hyperstrict* demand. The expression `e` puts this demand
  on its argument `x` if every evaluation of `e` is guaranteed to
  diverge, regardless of the value of the argument. We call this
  demand *hyperstrict* because it is safe to evaluate `x` to arbitrary
  depth before evaluating `e`. This demand is polymorphic with respect
  to function calls and can be seen as `B = C(B) = C(C(B)) = ...` for
  an arbitrary depth.


  


- `L` -- a *lazy* demand. If an expression `e` places demand `L` on a
  variable `x`, we can deduce nothing about how `e` uses `x`. `L` is
  the completely uninformative demand, the top element of the lattice.

- `S` -- a *head-strict* demand.  If `e` places demand `S` on `x` then
  `e` evaluates `x` to at least head-normal form; that is, to the
  outermost constructor of `x`.  This demand is typically placed by
  the `seq` function on its first argument. The demand `S(L ... L)`
  places a lazy demand on all the components, and so is equivalent to
  `S`; hence the identity `S = S(L ... L)`. Another identity is for
  functions, which states that `S = C(L)`. Indeed, if a function is
  certainly called, it is evaluated at lest up to the head normal
  form, i.e., *strictly*. However, its result may be used lazily.

- `S(s1 ... sn)` -- a structured strictness demand on a product.  It
  is at least head-strict, and perhaps more.

- `C(s)` -- a *call-demand*, when placed on a binder `x`, indicates
  that the value is a function, which is always called and its result
  is used according to the demand `s`.


Absence/usage demands

- `A` -- when placed on a binder `x` it means that `x` is definitely
  unused.

- `U` -- the value is used on some execution path.  This demand is a
  top of usage domain.

- `H` -- a *head-used* demand. Indicates that a product value is used
  itself, however its components are certainly ignored. This demand is
  typically placed by the `seq` function on its first argument. This
  demand is polymorphic with respect to products and functions. For a
  product, the head-used demand is expanded as `U(A, ..., A)` and for
  functions it can be read as `C(A)`, as the function is called (i.e.,
  evaluated to at least a head-normal form), but its result is
  ignored.

- `U(u1 ... un)` -- a structured usage demand on a product. It is at
  least head-used, and perhaps more.

- `C(u)` -- a *call-demand* for usage information. When put on a
  binder `x`, indicates that `x` in all executions paths where `x` is
  used, it is *applied* to some argument, and the result of the
  application is used with a demand `u`.


Additional information (demand signature suffix)

- `m` -- the function returns a
         [constructed product result](http://research.microsoft.com/en-us/um/people/simonpj/Papers/cpr/index.htm).

- `b` -- the function definitely diverges.

- `x` -- the function catches exceptions. For instance, consider
  `catch undefined g`: naturally, `catch` is strict in its first
  argument and therefore one would usually think that this expression
  would bottom. However, `catch` has special semantics: it catches
  exceptions. Consequently we give the first argument a demand of
  `C(L)x` to indicate that an application of `catch` to bottom can't
  be assumed to be itself bottom.

## Worker-Wrapper split


Demand analysis in GHC drives the *worker-wrapper transformation*,
which exposes specialised calling conventions to the rest of the
compiler.  In particular, the worker-wrapper transformation implements
the unboxing optimisation.


The worker-wrapper transformation splits each function `f` into a
*wrapper*, with the ordinary calling convention, and a *worker*, with
a specialised calling convention.  The wrapper serves as an
impedance-matcher to the worker; it simply calls the worker using the
specialised calling convention.  The transformation can be expressed
directly in GHC's intermediate language.  Suppose that `f` is defined
thus:

```wiki
  f :: (Int,Int) -> Int
  f p = <rhs>
```


and that we know that `f` is strict in its argument (the pair, that
is), and uses its components.  What worker-wrapper split shall we
make? Here is one possibility:

```wiki
 f :: (Int,Int) -> Int
  f p = case p of
          (a,b) -> $wf a b

  $wf :: Int -> Int -> Int
  $wf a b = let p = (a,b) in <rhs>
```


Now the wrapper, `f`, can be inlined at every call site, so that
the caller evaluates `p`, passing only the components to the worker 
`$wf`, thereby implementing the unboxing transformation.


But what if `f` did not use `a`, or `b`?  Then it would be silly to
pass them to the worker `$wf`.  Hence the need for absence
analysis.  Suppose, then, that we know that `b` is not needed. Then
we can transform to:

```wiki
  f :: (Int,Int) -> Int
  f p = case p of (a,b) -> $wf a

  $wf :: Int -> Int
  $wf a = let p = (a,error "abs") in <rhs>
```


Since `b` is not needed, we can avoid passing it from the wrapper to
the worker; while in the worker, we can use `error "abs"` instead of
`b`.


In short, the worker-wrapper transformation allows the knowledge
gained from strictness and absence analysis to be exposed to the rest
of the compiler simply by performing a local transformation on the
function definition.  Then ordinary inlining and case elimination will
do the rest, transformations the compiler does anyway.

## Discussion


There's ongoing discussion about improvements to the demand analyser.

- Inspired by Call Arity's Co-Call graphs, 
  [this page](commentary/compiler/demand/let-up) discusses how to make the LetUp rule more flow sensitive

## Relevant compiler parts


Multiple parts of GHC are sensitive to changes in the nature of demand
signatures and results of the demand analysis, which might cause
unexpected errors when hacking into demands. [This list](commentary/compiler/demand/relevant-parts) enumerates the parts
of the compiler that are sensitive to demand, with brief summaries of
how so.

## Instrumentation


For the [Journal version of the demand analysis paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/03/demand-jfp-draft.pdf) we created some instrumentation

- to measure how often a thunk is entered (to see if the update code was useful), and also
- to find out why a thunk is expected to be entered multiple times.


The code adds significant complexity to the demand analyser and the code generator, so we decided not to merge it into master (not even hidden behind flags), but should it ever have to be resurrected, it can be found in the branch `wip/T10613`. View the [full diff](http://git.haskell.org/ghc.git/commitdiff/refs/heads/wip/T10613?hp=930a525a5906fdd65ab0c3e804085d5875517a20) (at least as long as the link is valid, as it hard-codes the base commit).


# Relevant GHC parts for demand analysis results

**[This page was taken from the page on the GHC wiki](https://gitlab.haskell.org/ghc/ghc/-/wikis/commentary/compiler/demand/relevant-parts)**

- `compiler/basicTypes/Demand.lhs` -- contains all information about demands and operations on them, as well as about serialization/deserialization of demand signatures. This module is supposed to be changed whenever the demand nature should be enhanced;

- `compiler/stranal/DmdAnal.lhs` -- the demand analysis itself. Check multiple comments to figure out main principles of the algorithm.

- `compiler/stranal/WorkWrap.lhs` -- a worker-wrapper transform, main client of the demand analysis. The function split is performed in `worthSplittingFun` basing on demand annotations of a function's parameters. 

- `compiler/stranal/WwLib.lhs` -- a helper module for the worker-wrapper machinery. The "deep" splitting of a product type argument makes use of the strictness info and is implemented by the function `mkWWstr_one`. The function `mkWWcpr` makes use of the CPR info.

- `compiler/basicTypes/Id.lhs` -- implementation of identifiers contains a number of utility functions to check/set demand annotations of binders. All of them are just delegating to appropriate functions/fields of the `IdInfo` record;

- `compiler/basicTypes/IdInfo.lhs` -- `IdInfo` record contains all information about demand and strictness annotations of an identifier. `strictnessInfo` contains a representation of an abstract two-point demand transformer of a binder, considered as a reference to a value. `demandInfo` indicates, which demand is put to the identifier, which is a function parameter, if the function is called in a strict/used context. `seq*`-functions are invoked to avoid memory leaks caused by transforming new ASTs by each of the compiler passes (i.e., no thunks pointing to the parts of the processed trees are left). 

- `compiler/basicTypes/MkId.lhs` -- A machinery, responsible for generation of worker-wrappers makes use of demands. For instance, when a signature for a worker is generated, the following strictness signature is created:

  ```wiki
    wkr_sig = mkStrictSig (mkTopDmdType (replicate wkr_arity top) cpr_info)
  ```

  In words, a non-bottoming demand type with `N` lazy/used arguments (`top`) is created for a worker, where `N` is just a worker's pre-computed arity. Also, particular demands are used when creating signatures for dictionary selectors (see `mkDictSelId`). 

- `compiler/prelude/primops.txt.pp` -- this file defines demand signatures for primitive operations, which are inserted by `cpp` pass on the module `compiler/basicTypes/MkId.lhs`;

- `compiler/coreSyn/CoreArity.lhs` -- demand signatures are used in order to compute the unfolding info of a function: bottoming functions should no be unfolded. See `exprBotStrictness_maybe` and `arityType`.

- `compiler/coreSyn/CoreLint.lhs` -- the checks are performed (in `lintSingleBinding`): 

  - whether arity and demand type are consistent (only if demand analysis already happened);
  - if the binder is top-level or recursive, it's not demanded (i.e., its demand is not strict).

- `compiler/coreSyn/CorePrep.lhs` -- strictness signatures are examining before converting expression to A-normal form.

- `compiler/coreSyn/MkCore.lhs` -- a bottoming strictness signature created for `error`-like functions (see `pc_bottoming_Id`).

- `compiler/coreSyn/PprCore.lhs` -- standard pretty-printing machinery, should be modified to change PP of demands.

- `compiler/iface/IfaceSyn.lhs`  -- serialization, grep for `HsStrictness` constructors.

- `compiler/iface/MkIface.lhs`  -- a client of `IfaceSyn`, see usages of `HsStrictness`.

- `compiler/iface/TcIface.lhs` -- the function `tcUnfolding` checks if an identifier binds a bottoming function in order to decide if it should be unfolded or not

- `compiler/main/TidyPgm.lhs` -- Multiple checks of an identifier to bind a bottoming expression, running a cheap-an-cheerful bottom analyser. See `addExternal` and occurrences of `exprBotStrictness_maybe`.

- `compiler/simplCore/SetLevels.lhs` -- It is important to zap demand information, when an identifier is moved to a top-level (due to let-floating), hence look for occurrences of `zapDemandIdInfo`.

- `compiler/simplCore/SimplCore.lhs` -- this module is responsible for running the demand analyser and the subsequent worker-wrapper split passes. 

- `compiler/simplCore/SimplUtils.lhs`  -- is a new arity is less than the arity of the demand type, a warning is emitted; check `tryEtaExpand`.

- `compiler/specialise/SpecConstr.lhs` -- strictness info is used when creating a specialized copy of a function, see `spec_one` and `calcSpecStrictness`.
