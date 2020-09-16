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



  hask.global @one {
      %v = hask.make_i64(1)
      %boxed = hask.construct(@MkSimpleInt, %v:!hask.value)
      hask.return(%boxed): !hask.untyped
  }

  hask.global @leftOne {
    %o = hask.ref(@one) : !hask.untyped
    %l = hask.construct(@Left, %o:!hask.untyped)
    hask.return(%l) :!hask.untyped
  }

  hask.func @rightLeftOne {
    %lam = hask.lambdaSSA() {
      %l = hask.ref(@leftOne): !hask.untyped
      %l_t = hask.thunkify(%l: !hask.untyped):!hask.thunk<!hask.untyped>
      %r = hask.construct(@Right, %l_t :!hask.thunk<!hask.untyped>)
      %r_t = hask.thunkify(%r :!hask.untyped):!hask.thunk<!hask.untyped>
      hask.return(%r):!hask.untyped
    }
    hask.return(%lam) :!hask.fn<() -> !hask.untyped>
  }

  // 1 + 2 = 3
  hask.func@main {
    %lam = hask.lambdaSSA(%_: !hask.thunk<!hask.untyped>) {
      %input = hask.ref(@rightLeftOne) : !hask.fn<() -> !hask.untyped>
      %input_closure = hask.apSSA(%input: !hask.fn<() -> !hask.untyped>) 

      %v = hask.make_i64(42)
      %output_v = hask.construct(@MkSimpleInt, %v:!hask.value)
      %output_t = hask.thunkify(%output_v :!hask.untyped):!hask.thunk<!hask.untyped>
      hask.return(%output_t) : !hask.thunk<!hask.untyped>
    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }
}
