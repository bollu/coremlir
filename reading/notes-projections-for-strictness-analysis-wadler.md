# Projections for Strictness Analysis

- "context analysis reaches places abstract interpretation cannot reach": Most interesting!
- context analysis can find both head and tail strictness. abstract interpretation can find
  tail strictness, and head-strictness when paired with tail strictness, but not head
  strictness alone.
- Still first order, _not_ higher order.

- `before`:

```hs
-- | take elements until first 0
before xs = 
  case xs of
    []     -> [];
    (y:ys) -> if y == 0 then [] else y:before ys

```
- `length`:

```hs
length xs = 
  case xs of
    []     -> 0
    (y:ys) -> 1 + length ys
```


- `doubles`:
```hs
-- | double the elements of the list.
doubles xs = 
  case xs of
    []   -> []
    y:ys -> (2*y):doubles ys
  
```

# Head, tail strictness.

- `:` is the list constructor.
- `:h` is a list constructor that is strict in the head.
- `H` is the function that replaces each `:` by a `:h`. So,

```
H(1:2:⊥:3:[])
= 1 :h 2 :h ⊥ :h 3 :h []
= 1 :h 2 :h ⊥
```

- `:t` is the list constructor that is strict in the tail. 
- `T` is the function that replaces each `:` by a `:t`.

- `f` is **head-strict** if we can replace `:` by `:h` in the arguments to `f`.
  That is, `f = f . H`. `f` is tail-strict if `f = f . T`.

##### `before` is head strict
because it scrutinizes the head of the list: it has a `(y == 0)`.

Alternatively, case analyze on the relative position between a `⊥` and the
first `0`.

- If a bottom occurs before the first zero, then we will
  take the list till the bottom:
```hs
before(1:⊥:0:[]) = 1:⊥

before(H(1:⊥:0:[]))
  = before(1:⊥) 
  = 1:⊥
```
- If a bottom occurs after the first zero, then we will take the list till
  the first zero:

```hs
before(1:2:0:⊥:3:[]) = 1:2:[]

before(H(1:2:0:⊥:3:[]))
= before(1:2:0:⊥)
= 1:2:[]
```

##### `length` is tail strict
- Because `length` recurses on the tail and it analyzes its input, it will force
  the tail.
- If any tail field contains `⊥` then `length` is undefined:

```hs
length(1:2:3:⊥) = 3 + length ⊥ = ⊥
T(1:2:3:⊥) = ⊥
```

- The heads can contain `⊥` because `length` does not inspect elements:

```hs
length(1:⊥:3:[]) = 3
T(1:⊥:3:[]) = 1:⊥:3:[]
```

#### Where this is useful

- If we have `f . g` and we know that `f` is head-strict, we can replace `f`
  with `f . H . g`. So we replace `g` with `gH`, where every use of `:` in `g`
  is replaced by `:h`.

#### Contexts beget contexts.

- `doubles` is not head-strict:

```
doubles(1:⊥:3:[]) = 2:⊥:6:[] 
doubles(H(1:⊥:3:[])) = doubles(1:⊥) = 2:⊥
```
- The two results are not equal, so `doubles` is not head-strict.
- On the other hand, we can see that `H . doubles = H . doubles . H`. 
- Since `before` is head-strict, we can optimize:

```
before . doubles
= before . H . doubles [before head-strict]
= before . H . doubles . H [doubles head-strict in H context
```

#### Tail strictness v/s Head strictness
- A strict function is tail-strict  iff `f u = ⊥` whenever `tail(u) = ⊥`.
  If any tail of `u` is bottom, the `T(u) = ⊥`. So:
```
f u 
= f (T u)  [f is tail-strict]
= f ⊥  [assumption]
= ⊥ [f is tail-strict; tail of ⊥ = ⊥]
```

- It is **not true** that a function is head-strict only if `f u = ⊥`
  whenever some head of u is `⊥`.  The intuition is that we can have a function `f`
  that forces the _list_, while not having to force _elements of the list_. 
  For example, how `length` works.
- The context approach can describe such functions, while `f u = ⊥` cannot.

#### 2. Projections
- `p(u) <= u`, and ` p (p(u)) = p(u)` is a projection.
- We can also rephrase the above as: `p <= id`, `p . p = p`.
- Projections form a complete lattice with `id` at the top, `bot` at
  the bottom, where `id u = u`, and `bot u = ⊥`.

##### Definition: function being strict in context

a function `f` is `s`-strict in context `c` [`c`, `s` are projections] 
if `c . f = c . f . s`. We write `f: c => s`. For example, we have seen that
`before: id => H`, and `doubles: H => H`.

##### Proposition: `f: a => b` iff `a . f <= f . b`

**Forward proof**

We have that `f: a => b`. that is, `a . f = a . f . b`.  We must show
that `a . f <= f . b`.

```
f.b = f.b 
a.f.b <= f.b  [a is projection, pulls values downwards]
a.f <= f.b [a.f = a.f.b]
```

**Backward proof**

We have that `a.f <= f.b`. We must show that

`f: a => b`. that is, `a.f = a.f.b`.

- Step 1: show that `a.f <= a.f.b`.

```
a(f(x)) <= f(b(x)) [assumption]
a(a(f(x)) <= a(f(b(x))) [a is monotone; x < y => a(x) <= a(y)]
a(f(x)) <= a(f(b(x))) [a is projection; a.a = a]
a.f <= a.f. b [pointfree]
```
- Step 2: show that `a.f >= a.f.b`: Since `id >= b`, hence `a.f.id >= a.f.b`.
  Spelled out:
```
x >= b(x) [b is a projection]
y >= z => a(f(y)) >= a(f(z)) [a.f is monotone]
a(f(x)) >= a(f(b(x))) [set y=x, z=b(x)]
a.f >= a.f.b
```

- Step 3: since `a.f <= a.f.b` and `a.f >= a.f.b` we have that `a.f = a.f.b`.

##### Proposition: if `f: a => b` and `g: b => c` then `f . g : a => c`
- From above, `a.f <= f.b`
- From above, `b.g <= g.c`

```
(a.f).g
<= (f.b).g
= f.(b.g)
<= f.(g.c)
```

- From above, since `a.f.g <= f.g.c` we have that `f.g: a => c`.


##### Propositions: If `A` is a set of projections then `union(A)` exists and is a projection.

- For every `a ∈ A` we have that `a <= id`.
- In a DCPO, every bounded set has a least upper bound (LUB) which is at most the LUB
  [TODO: find reference]
- Thus the element `u = union(A)` exists since `A` is bounded since we are within
  a DCPO. Also, `u <= id`.
- We need to show that `u` is a projection.
- We must have that `u . u <= [u.id = u]` because `u <= id`.
- It remains to be shown that `u.u >= u`, thereby giving `u.u = u`.
- The core idea is to expand `u` in terms of its union, and then exploiting
  that it is built from a union of projections.

```
(u.u)(x)
= u(union({ a(x) : a ∈ A}))   [u = union(A)]
>= union({ u(a(x)) : a ∈ A})) [u <= id]
>= union({ a(a(x)) : a ∈ A})) [a <= u, since u is LUB]
>= union({ a(u) : a ∈ A}))    [a is projection]
= u(x)
```

- Hence `u.u >= u`. Combining `u.u <= u` we have shown that `u` is a projection.

##### Proposition: intersection of projections need not always be projection

```
f(c)=c|c---c --   |g(c)=b
      |        \  |
f(b)=a|  --b----b |g(b)=b
      | /	      |
f(a)=a|a---a---- a|g(a)=a
```

- Call the intersection of `f` and `g` as `h`. We define `h(x) = intersect(f(x), g(x))`.
-  Now, `h(a) = a` since both `f, g` map `a` to `a`.
- `h(b) = a` since `f` pulls `b` down to `a` will `g` keeps `b` at `b`.
- `h(c) = b` since `g` pulls `c` down to `b` while `f` keeps `c` at `c`.
- Now note that `h(h(c)) = a` while `h(c) = b`. Thus `h` is not a projection!


##### Proposition: defining intersection of projections

- We define the intersection of two projections `p, q` to be the largest projection `r`
  such that `r <= p` and `r <= q`. Formally, it is:
  
```
intersect(A) = union(lower-bounds(A))
= union_a({ p ∈ Proj: ∀ a ∈ A, p <= a})
```
- This must exist because unions exist because bounded set of DCPO.

#### 3: Strictness and absence

- We have `⊥` which tells us that if forced, we will get a divergent computation.
  We need a way to talk about divergence itself.
- Add a new element `⑂` (lightning bolt) which means "divergence". This is less 
  that `⊥`

##### Unacceptable values
- A value `u` is **unacceptable** to a projection `p` iff `p(u) = ⑂`.
- If `f: a => b`  and `u` is unacceptable to `b`, then `f(u)` is
  unacceptable to `a`

```
a.f = a.f.b [by defn. f:a => b]
a(f(u) = a(f(b(u)))
a(f(u)) = a(f(⑂)) [u is unacceptable to b]
a(f(u)) = ⑂ [⑂ propagates]
```
- Since `a(f(u)) = ⑂`, `f(u)` is unacceptable to `a`.

##### `STR` projection

Define:

```
STR(⑂) = ⑂
STR(⊥) = ⑂
STR(x) = x
```

##### `FAIL` projection

- `FAIL(_) = ⑂`. This is the new bottom of the lattice.

##### `f: STR => FAIL` iff `f` is divergent

- The intuition is that if we evaluate the return value of `f` [the context `STR=>`],
  then this is as good as failing [the context `=> FAIL`]. That is, `f` returns a bottom value
  on **all** inputs when the output is forced.
  
```
STR(f(x)) = STR(f(fail(x)))
STR(f(x)) = STR(f(⑂)) [defn of fail]
STR(f(x)) = ⑂ [propagate ⑂]
f(x) = ⑂ or f(x) = ⊥ [defn of STR]
```

- Hence, we have that `f(x)` does not produce a useful value on any input.

##### `f: FAIL => FAIL` for all functions

- The intuition is that if we want to fail after `f` returns [the context `FAIL =>`],
  we might as well fail before evaluating `f` [the context `=> FAIL`].
  
```
FAIL(f(x)) = FAIL(y) = ⑂
FAIL(f(FAIL(x))) = FAIL(z) = ⑂
FAIL(f(x)) = FAIL(f(FAIL(x)))
```

- Hence, if `f` is divergent, so is `f . g` for any `g`, because failing after
  evaluating `f` is the same as failing before evaluating `f`.
  
##### `ABS` projection

- `ABS` detects the absence of values. It squashes all values to `⊥` while leaving
  divergent as divergent:
  
```
ABS(⑂) = ⑂
ABS(⊥) = ⊥
ABS(x) = ⊥
```


##### Ignoring arguments
- Say that **f ignores its argument** if `f(u) = f(⊥)` for all `u`.

##### `f: STR => ABS` iff `f` ignores its argument



##### UB versus abort

The semantcs of `abort` is such that the program simply errors out. If we want
to optimize programs, it seems that we can replace these semantics with that
of undefined behaviour. This gives us a pretty precise definition of what
UB is and how it can propagate.


##### Domain of projections

We get a lattice:

```   ID
  ABS    STR
     FAIL
```
which is a _subdomain_ of all projections of the full domain `D`. Here, the
different labels mean:
- `FAIL`: no value returned by `e` is acceptable. We can implement `e` by 
  immediately aborting.
- `ABS`: the value of `e` is ignored. We don't need to pass the parameter `e`.
- `STR`: e is forced. We can evaluate `e` immediately.
- `ID`: `e` is either forced, or not used. We must construct a thunk
  for `e`.

##### 4 Finite domains

For primitive types, we can build a domain with a projection operator called
`EQUAL(x0): D -> D` which operates as:

```
EQUAL(x0)(x) = x0 [if x = x0]
EQUL(x0)(_) = ⑂  [otherwise]
```

- Thus, if we have `f: EQUAL(y) => EQUAL(x)` then we must have that `f(x) = y`.
- This is generally "too precise". We don't need this much of information about
  a datatype.
  
##### 5 Finite domains for lists
- `LIST D` is the domain whose elements come from the domain `D`. we've already
seen `H` and `T`. here we see two more projections called `NIL` and `CONS`
which strictly expect to see a `NIL` or a `CONS` node:

```
NIL :: LIST(D) -> LIST(D)
NIL (⑂) = ⑂
NIL (⊥) = ⑂
NIL ([]) = []
NIL (x:xs) = ⑂
```

- a `CONS` is a _projection transformer_. It takes two projections, one for the
  head, a `px: Proj(D)` and one for the tail, a projection for the rest of the 
  list, `pxs: Proj(LIST(D))`, and it applies these if it finds a cons otherwise.
  Otherwise, it `⑂`s out.

```
CONS :: Proj(D) -> Proj(LIST(D)) -> LIST(D) -> LIST(D)
CONS px pxs (⑂) = ⑂
CONS px pxs (⊥) = ⑂
CONS px pxs ([]) = ⑂
CONS px pxs (x:xs) = (px x):(pxs xs)
```

We can describe the shape of lists exactly using these. For example, the projection:

```
CONS ID (CONS EQUAL(0) NIL))
```

Interestingly, we can also describe exactly finite and infinite lists with
these recursive definitions:

```
FIN (a) = NIL U CONS a (FIN a)
INF (a) = NIL U CONS a (ABS U INF a)
```

- `Fin` needs both the head and the tail to be strictly evaluatable. This is 
  only possible if the list is finite. Consider `let xs = 1:xs`. We can't evaluate
  the tail of the list reucrsively in finite time!
- `INF` allows the tail to be absent, thereby allowing us to pass off
  infinite lists.
- We have `length: STR => FIN ABS`, which means that if we demand `length`,
  then we need the input to be a finite list that ignores its input.

#### 6 Context Analysis

Given a function `f` and a projection `α` we want to find a projection `β` such that
`f: α => β`. Recall that this condition is phrased in two equivalent ways:
- (original): `α . f = α . f . β`:
  it is safe to elide `β` more data from the input if all we want is `α` amount of data in the output
- (equivalent): `α . f <= f . β`: The output from `f` after removing `β` amount of information 
  is still larger than what is perceived by `α` with _no_ information removed. It is thus safe to
  remove `β` amount of information.

It is equivalent to halting to find the smallest `β` for a given `α`: We have that
`f: STR ⇒ FAIL` only holds iff `f` diverges for each argument. Thus, we can check
if a function halts or not on all inputs by asking `f: STR ⇒ ?`

##### LANGUAGE

```
e := x [variable]
   | k [constant]
   | f(e1, ..., en) [function application]
   | if e0 then e1 else e2 [conditionals]
   | case e0 of [] => e1 | y:ys => e2
```

#### 6.2: Projection transformers

##### Projection transformers for functions

For a given demand `α (f(u1, ..., un))`, we define `f[i](α)` to be the safe
demand on the input `ui`. That is:

```
α(f(u1, ..., ui, ... , un)) <= f(u1, ..., f[i](α)[ui], ..., un)
```

Equivalently this can be phrased as:

```
α(f(u1, ..., ui, ... , un)) = α(f(u1, ..., f[i](α)[ui], ..., un))
```

This tells us that `f[i](α)` is a safe demand on the input `ui` when the demand
on the output is `α`. So formally, if `f: T1 -> .. -> Ti -> .. -> Tn -> R`, 
then we must have `α: PROJ(R)`, `β = f[i](α): PROJ(Ti)`, this we must have
`f[i]: PROJ(R) -> PROJ(Ti)`: A projection transformer that transforms a
projection on the return type to a projection on the `i`th input.

##### Projection transformers for expressions

Similarly, for a given expression `e : Te` and  a variable  `x: Tx`
we define `e[x]: PROJ(Te) -> PROJ(Tx)` which transforms a projection on the
output to a projection on the variable `Tx`, defined by the relation:

```
α(e) <= substitute(expr=e, var=x, replacement=e[x](α)(x))
```

That is, given a demand of `α` on `e`, we can safely replace the variable `x`
with the variable `e[x](α)(x)`.

#### Definition of `e[x](α)`:

```
x[y](α) = x == y ? α : ABS [variable]
k[y](α) = ABS [constant]
e[x](α) = α |> e[x](α') [α' is the strict version of α]

α' STRICT:
----------

(f(e1, ..., en))[x](α) = [function application]
  e1[x](f[1](α)) U! ... U! en[x](f[n](α))

(if e0 then e1 else e2)[x](α) =  [conditionals]
   e0[x](STR) U! (e1[x](α) U e2[x](α))

(case e0 of [] => e1 | y:ys => e2)[x](α) = [case]
  (e0[x](NIL) U! e1[x](α)) U
  (e0[x](CONS(e2[y](α), e2[ys](α) U! e2[x](α))
```

- NOTE: I rename the `&` operator as the `U!` operator because I feel that this
  actually captures the spirit of what the operator is trying to do.


#### 6.3: the `|>` [guard] operation:

Notice that if we have a projection that is lazy in our four point domain
`{FAIL, ABS, STR, ID}` , then we can decompose its 
effect into two parts: the part that is `ABS` and the "strict part". For example,
- `ABS = ABS U FAIL`.
- `ID = ABS U STR`.

All projections that are built out of the 4-point domain (such as `CONS ABS ABS`)
can again be reucrsively decomposed into the "strict" part and the "lazy" part.
So we build an operator `|>` whose entire job is to help us decompose the domain
into the lazy part and the strict part. so it has the rules:

```
FAIL |> β = FAIL
ABS |> β = ABS
α |> β = β [if α is strict and α != FAIL]
```

So, if we have some kind of lazy operator `α`, we can write its effect as:

```
α |> α'
```

where `α'` is the strictified version of `α`. So we use this to define the
effect of `e[x](α)` for only strict and non-fail operators, and use the `|>`
operator to compose the effect of laziness / failure.

#### 6.4: function application

#### 6.5: conditional expression
#### 6.6: case 
#### 6.7: primitives


