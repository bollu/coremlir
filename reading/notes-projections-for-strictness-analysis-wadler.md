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
- Projections form a complete lattice with `id` at the top, `bot` at the bottom,
  where `id u = u`, and `bot u = ⊥`.

##### Definition: function being strict in context

a function `f` is `s`-strict in context `c` [`c`, `s` are projections] 
if `c . f = c . f . s`. We write `f: c => s`. For example, we have seen that
`before: id => H`, and `doubles: H => H`.

##### Proposition: `f: a => b` iff `a . f <= f . b`

