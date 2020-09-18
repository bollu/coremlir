// Check that constructors let us build left and right.
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 42
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  hask.adt @EitherBox [#hask.data_constructor<@Left[@Box]>, #hask.data_constructor<@Right[@Box]>]
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt[@"Int#"]>]



  hask.func @one {
      %lam = hask.lambdaSSA() {
          %v = hask.make_i64(1)
          %boxed = hask.construct(@SimpleInt, %v:!hask.value) :!hask.adt<@SimpleInt>
          hask.return(%boxed): !hask.adt<@SimpleInt>
      }
      hask.return(%lam): !hask.fn<() -> !hask.adt<@SimpleInt>>
  }

  hask.func @leftOne {
    %lam = hask.lambdaSSA() {
        %ofn = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
        %o = hask.apSSA(%ofn  : !hask.fn<() -> !hask.adt<@SimpleInt>>)

        %l = hask.construct(@Left, %o: !hask.thunk<!hask.adt<@SimpleInt>>) :!hask.adt<@Either>
        hask.return(%l) :!hask.adt<@Either>
    }
    hask.return(%lam) :!hask.fn<() -> !hask.adt<@Either>>
  }

  hask.func @rightLeftOne {
    %lam = hask.lambdaSSA() {
      %lfn = hask.ref(@leftOne): !hask.fn<() -> !hask.adt<@Either>>
      %l_t = hask.apSSA(%lfn: !hask.fn<() -> !hask.adt<@Either>>)

      %r = hask.construct(@Right, %l_t :!hask.thunk<!hask.adt<@Either>>) :!hask.adt<@Either>
      %r_t = hask.thunkify(%r :!hask.adt<@Either>):!hask.thunk<!hask.adt<@Either>>
      hask.return(%r):!hask.adt<@Either>
    }
    hask.return(%lam)  :!hask.fn<() -> !hask.adt<@Either>>
  }

  // 1 + 2 = 3
  hask.func@main {
    %lam = hask.lambdaSSA(%_: !hask.thunk<!hask.value>) {
      %input = hask.ref(@rightLeftOne) : !hask.fn<() -> !hask.adt<@Either>>
      %input_closure = hask.apSSA(%input: !hask.fn<() -> !hask.adt<@Either>>)

      %v = hask.make_i64(42)
      %output_v = hask.construct(@SimpleInt, %v:!hask.value) :!hask.adt<@SimpleInt>
      hask.return(%output_v) : !hask.adt<@SimpleInt>
    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.adt<@SimpleInt>>
  }
}
