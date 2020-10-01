# Notes on 'theory and practice of demand analysis in haskell'

#### Their (confusing) notation:

- `S`: evaluated to head normal form [why `S`?]
- `A`: not evaulated **A**t all
- 

#### Example of computation that is divergent but still uses argument (2.2)

```hs
f x y = error x
```

should **not** be transformed into:

```hs
fw = let x = error "x"; y = error "y" in error x
f x y = fw 
```

We need to use the _original_ x.

#### Handling seq (2.2)

- Since they don't have an example, I need to create my own. I *think* this is
  right.

- This example shows that due to the presence of `seq` which can witness the
  difference between a bottom and a non-bottom, eta-expansion (going from `f` to `(\x -> f x)`)
  is not always legal:

```hs
bot :: a; bot = bot
bot2 :: a -> b -> c; bot2 a = bot

bot2 a /= (\b -> bot2 a b)
bot2 a = bottom
(\b -> bot2 a b) != bottom
```


#### 2.3

- old analyzer is in some paper (Peyton Jones 1996) where the reference is
  broken in the PDF. Will hunt this down. Used to use abstract interpretation.
- given `f x y z = <rhs>` the analyzer would analyze the strictness of `f`
  by trying `(f ⊥ T T)`, `(f T ⊥ T)`, `(f T T ⊥)`. It would then iterate
  this this analysis. This makes it exponential in the depth of the function.
- Backward analysis is (supposedly) much more efficient, as this paper will
  demonstrate.

#### 3: Backwards analysis

1. Strictness: how much an expression is guaranteed to be evaluated
2. Usage/absence: what parts of the expression is used.
- Strictness is different from usage as witnessed by `error`. We can also
  have things like `(f !x y = y)`. `f` does not _use_ x, but it is _strict_ in `x`
  due to the `!`-pattern.

#### 4: Projections

- In a CPO, a projection is an idempotent operator that removes
information: `p(x) <= x; p(p(x)) = x`.
- Written point-free, we have `p <= id; p . p = p`
- A demand that evaluates certain parts of data structures is modeled as a function
  that (1) **does not touch** evaluated parts, (2) **smashes** un-evaluated parts to `⊥`.
- So the demand of `fst (x, y) = x` is modeled by `fstp (x,y) = (x, ⊥)`.
- If this is correct, then we must have that `fst = fst . p`. That is, we can have
  `p` throw away information; `fst` cannot notice.

#### 4.2 Properties of projections
- complete lattice under point-wise `<=` ordering.
- LUB is defined point-wise for projections.
- Wadler&Hughes 1987: If `P` is a set of projections, the `LUB(P)` exists and is a projection.

##### Lemma 4.1: if `p1 <= p2` then `p1 . p2 = p1`.
**Proof**: We have that:

```
p1 
= p1 . p1 [defn of projection]
<= p1 . p2 [p1 <= p2; pointwise ordering]
```

```
p2 <= id
p1 . p2 <= p1 . id 
p1 . p2 <= p1
```

- Combining, we have `p1 <= p1 . p2; p1 . p2 <= p1` giving us `p1 . p2 = p1`.

- Projections are used to infer **how much information** can be removed.
- Formally, given a projection `p` and an element `d`, we ask: What is the
  best projection `q(p, d)` such that `p . q(p, d) $ d = p d`. 
- `q` tells us how much more information from `d` we can throw out in the
  context of `p`.

#### Context strengthening lemma

- If (`phigh d = phigh (q d)`) and (`plow <= phigh`), then (`plow d = plow (q d)`).
- Intuitively, `phigh` "needs more arugments" that `plow`. `q` removes data
  that's not used by `phigh`. `plow` uses _less data_ than `phigh`. So `plow`
  can't use data that's now used by `phigh`. Whatever `q` removes is safe for `plow`.


#### 4.4 Useful projections
- `id = \x -> x`
- `bot = \x -> ⊥`
- `(p, q)`: product of projections
- `p -> q`: higher order projection.

#### Product of projections

```
(p, q)(d, d') = (p d, q d')
(p, q)(_) = ⊥
```
##### Lemma: product of projections is a projection

```
(p, q)(d, e) = (p d, q e) <= (d, e) [since p, q are projections]
(p, q) . (p, q) = (p . p, q . q) = (p, q)
```

#### higher order projection

```
(p -> q)(f) = q . f . p [if f is a fn]
(p -> q)(_) = ⊥ otherwise
```

##### Lemma: higher order projections are projections

- If we apply `p -> q` on something that's not a function, the proof is trivial
  since we just get `⊥`.

```
q <= id
q.h <= [id.f = f]
q.h <= h

p <= id
g.p <= [g.id = g]
g.p <= p

let g = f
  [g.p <= p] => [f.p <= p]

let h = (f.p)
  [q.h <= h] => [q.(f.p) <= f.p <= p]
```

- Hence `q.f.p <= p <= id`. So, `q.f.p` must either preserve arguments or
  smash arguments to `⊥`, since it's less that `id`.

#### Projection Environments

- We have value environments which are partial functions `ValEnv = Var -o-> D` where `D`
  is the domain of values. [we denote partial functions as `-o->`: an arrow with a hole.
  [Notation stolen from Davey and Priestly].
- Define projection environments: projections on value environments.
- `type ProjEnv = (Var -o-> Projection, Projection)`
- `ProjEnv = (φ, r)` where `φ: Var -o-> Projection`; `r: Projection` is a default
  projection. Projection environments are denoted by `θ`.

```
@ : ProjEnv -> ValEnv -> ValEnv
((φ, r) @ ρ)(x) = φ(x)(ρ(x)) | if x ∈ Dom(φ)
((φ, r) @ ρ)(x) = r(ρ(x))  | otherwise
```

- That is, build a new environment which (i) looks up `ρ` (2) projects the lookup results.
  if `x` is in the domain, then project the result of looking up `x` using `φ(x)`.
  Otherwise, project the result of the lookup with the default projection `r`.
- The union of projection domains is computed pointwise.

```
((φ1, r1) U (φ2, r2)) = (φ1 U φ2, r1 U r2)
  (φ1 U φ2)(x) = φ1(x) U φ2(x)
```

##### Theorem: Projection environments are projections on value environments.

#### 4.6: Projection Types & Projection Transformers

- `type ProjTy = (Proj, ProjEnv)`. Represents the result of a projection analysis
  for an open expression: so we need a `ProjEnv` for the free variables of the
  expression, and a `Proj` for the expression itself.
- `type ProjTransform = ProjTy -> ProjTy`.  Continuous function from projection types
  to projection types.  An open expression is modeled by a projection transform.

```hs
-- Example 4.2
g :: (Int, Int, Int) -> [a] -> (Int, Bool)
g (a, b, c) = 
  case a of
    0 -> error "urk"
    _ -> \y -> case b of
                0 -> (c, null y)
                _ -> (c, False)
-- null returns if a list is empty or not.
```

Projection transformer of `g`:
- `id` maps to `id`.
- `id -> id` maps to `id -> id`.
- `id -> id -> (⊥, id)` maps to `(id, id, ⊥) -> id -> id`. That is, we 
  know that in the output, we can have the first component of the output as `⊥`.
  Given this information, if we look at `g`, the first component of the output
  is always `c` or `error "urk"`. The second component does not use `c`.
  Hence, if we want to allow the first component to be `⊥`, we can have `c = ⊥`.
- `id -> id -> (id, ⊥)` maps to `id -> ⊥ -> id`. See that the second component of
  the output is `False`, `null y`, or `error "urk"`. The first component
  does not use `y`. Thus, if we want the second component to be `⊥`, we can safely set `y = ⊥`.

##### How to approximate projection transformers

- Want to approximate concrete `T: ProjTransform` with abstract `T#`.
- Use a finite sequence of input projections `p[1] <= p[2] <= ... <= p[n]`.

```
T#(q) = t[i]# | if p[i-1] <= q <= pi
T#(q) = t[0]# | if q <= p0
T#(q) = T     | otherwise
```

In practice, we use `n=1`:
```
T#(q) = t#    | q <= p
T#(q) = T     | otherwise
```

- so we maintain a pair `(t#, p)` where `p` is the threshold upto which the 
  `t#` is legal.

TODO: write about "update", think about the types

#### 4.7: Safety of projection based analysis

- We can think of a function `PP: Expr -> ProjTransform`. Recall that
  `ProjTransform = ProjTy -> ProjTy`. A `ProjTy = (ProjEnv, Proj)`, where the
   first `ProjEnv` modeled the demand on the free variables of the expression, and the
   second `Proj` modeled the demand on the expression itself. Recall that A
   `ProjEnv = (Var -o-> Proj, Proj)`, where the first `Var -o-> Proj`
   was the known projections for variables, and the second `Proj` was the
   default projection used for unknown variables.


- If we have `PP(e)(p) = (θ, q)`, then we want that for all environments `ρ`,
   we have the equality:

```
p([e](ρ)) = p(q([e](θ@ρ)))
```

- That is, if `e` is demanded with projection `p`, then we can safely demand 
  `e` with an addition projection `q`, and demand the environment that `e` is
  being evaluated with a projection environment `θ`.
- The problem with the above definition is that it is not modular (?)
  [SID: I don't understand this]. We may already have information for some of
  the variables in `ρ`, which we want our analysis to take into account.

#### Definition 4.5: Transformer environments

```hs
type TransformerEnv = Var -o-> ProjTransform
~= Var -o-> (ProjTy -> ProjTy)
~= Var -o-> ((ProjEnv, Proj) -> (ProjEnv, Proj))
~= Var -o-> (((Var -o-> Proj, Proj), Proj) -> ((Var -o-> Proj, Proj), Proj))
```

We have a partial order on transformer environments since a projection
transform is a continuous function with a partial order on it.

#### Example of projection analysis: Page 15

```hs
let x = 42
  in let y = x + y
    in (y, 1)
```

  


#### Wild notes by Sid
- Why do we use a lattice of projections/idempotents? can we just use a boolean
  ring? does this give us something more?
