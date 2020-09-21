// Check that constructors let us build left and right.
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s
// CHECK: 1
module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  hask.adt @EitherBox [#hask.data_constructor<@Left[@Box]>, #hask.data_constructor<@Right[@Box]>]
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt[@"Int#"]>]


  // extract e = case e of @Right e2-> case e2 of @Left v -> v
  hask.func @extract {
    %lam = hask.lambda(%t: !hask.thunk<!hask.adt<@Either>>) {
       %v = hask.force(%t) : !hask.adt<@Either>
       %ret = hask.case @Either %v 
           [@Right -> { ^entry(%t2: !hask.thunk<!hask.adt<@Either>>):
               %v2 = hask.force(%t2) : !hask.adt<@Either>
               %ret2 = hask.case @Either %v2
                   [@Left -> { ^entry(%t3: !hask.thunk<!hask.value>):
                       %v3 = hask.force(%t3) : !hask.value
                       hask.return(%v3):!hask.value
                   }]
              hask.return(%ret2):!hask.value
           }]
        hask.return(%ret):!hask.value
    } 
    hask.return(%lam):  !hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>
  }

  hask.func @one {
      %lam = hask.lambda() {
          %v = hask.make_i64(1)
          %boxed = hask.construct(@SimpleInt, %v:!hask.value) :!hask.adt<@SimpleInt>
          hask.return(%boxed): !hask.adt<@SimpleInt>
      }
      hask.return(%lam): !hask.fn<() -> !hask.adt<@SimpleInt>>
  }

  hask.func @leftOne {
    %lam = hask.lambda() {
        %ofn = hask.ref(@one) : !hask.fn<() -> !hask.adt<@SimpleInt>>
        %o = hask.ap(%ofn  : !hask.fn<() -> !hask.adt<@SimpleInt>>)

        %l = hask.construct(@Left, %o: !hask.thunk<!hask.adt<@SimpleInt>>) :!hask.adt<@Either>
        hask.return(%l) :!hask.adt<@Either>
    }
    hask.return(%lam) :!hask.fn<() -> !hask.adt<@Either>>
  }

  hask.func @rightLeftOne {
    %lam = hask.lambda() {
      %lfn = hask.ref(@leftOne): !hask.fn<() -> !hask.adt<@Either>>
      %l_t = hask.ap(%lfn: !hask.fn<() -> !hask.adt<@Either>>)

      %r = hask.construct(@Right, %l_t :!hask.thunk<!hask.adt<@Either>>) :!hask.adt<@Either>
      %r_t = hask.thunkify(%r :!hask.adt<@Either>):!hask.thunk<!hask.adt<@Either>>
      hask.return(%r):!hask.adt<@Either>
    }
    hask.return(%lam)  :!hask.fn<() -> !hask.adt<@Either>>
  }

  // 1 + 2 = 3
  hask.func@main {
    %lam = hask.lambda(%_: !hask.thunk<!hask.value>) {
      %rlo = hask.ref(@rightLeftOne): !hask.fn<() -> !hask.adt<@Either>>
      %input = hask.ap(%rlo : !hask.fn<() -> !hask.adt<@Either>>)
      %extract = hask.ref(@extract) :!hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>

      %extract_t = hask.ap(%extract:!hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>, %input)
      %extract_v = hask.force(%extract_t) :!hask.value
      hask.return(%extract_v):!hask.value

    }
    hask.return (%lam) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
  }
}
