# Core2MLIR

Core plugin to convert Core into MLIR.

- [GHC 8.10](https://github.com/ghc/ghc/tree/ghc-8.10)
- [ANF](https://gitlab.haskell.org/ghc/ghc/blob/master/compiler/GHC/CoreToStg/Prep.hs)


To run these, first install the GHC plugin from `ghc-dump`. This installs
both the plugin that is loaded with `-fplugin GhcDump.Plugin` [The package `ghc-dump-core`]
as well as the utility tool for dumping these called `ghc-dump` [The package `ghc-dump-util`].

```
# install tool and compiler plugin 
$ cd ghc-dump && cabal install --lib all && cabal install all
```


Then run the command:

```
ghc -fplugin Core2MLIR.Plugin -dumpdir=dump -O fib.hs -o fib.out
```


## More detailed instructions.

GHC can only load plugins that are registered against its package database,
hence all this song-and-dance. 
- The first time we run `cabal install --lib --overwrite-policy=always`, cabal writes a bunch of stuff
  into a bunch of global locations, including the path `/home/bollu/.ghc/x86_64-linux-8.8.3/environments/default`.

- From now on, **only run `cabal build && cabal install --lib --overwrite-policy=always`**. If one re-runs `cabal install --lib`
  _without_ `--ovewrite-policy=always`,
  it goes and writes __a second entry__ into the aforementioned `environments/default` file.
  _This will succeed_. It will fail when one attempts to run `ghc -fplugin Core2MLIR.Plugin`
  with an error: `Ambiguous module name ‘Core2MLIR.Plugin'`.

- Indeed, _eat flaming death_ seems like a totally valid response to this state
  of affairs. Why do pure functional programmers build _more horribly stateful_,
  fragile systems than the UNIX folks who wrote in (gasp) C?

- This will still fucking _not work_ because the latest versions of Cabal
  build stuff in "environments". So we need to append a `--global`, or remember
  sandbox names. I wish this transition of `cabal` tooling was documented someplace.

- Also, it looks like `cabal install --lib` may soon be moved into a separate command
  because it "behaves very differently". You don't say. 

- Unfortunately, we _need to register_ a Core plugin with the GHC package
  registration system: [Compiler plugins](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/extending_ghc.html#compiler-plugins) states:

> Plugins can be added on the command line with the -fplugin=⟨module⟩ option
> where **⟨module⟩ is a module in a registered package that exports the plugin**.

Ergo, one seems to be hosed.

#### `cabal` warts one may run into:
- [Reinstall of package breaks things](https://github.com/haskell/cabal/issues/6391)
- [cabal install `--lib` is not idempotent](https://github.com/haskell/cabal/issues/6394)


#### References

- [GHC plugins](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/extending_ghc.html#annotation-pragmas)
- [GHC Source code viewer](https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/main/Finder.hs)
- [GHCPlugins module: `CoreSyn`](https://hackage.haskell.org/package/ghc-8.2.1/docs/CoreSyn.html)
- [GHCPlugins module: `CoreMonad`](https://hackage.haskell.org/package/ghc-8.2.1/docs/CoreMonad.html)

